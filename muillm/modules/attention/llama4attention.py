# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Callable, Optional, Tuple

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.rope.ropeops import apply_complex_rotary_emb
from muillm.modules.attention.temperaturetuning import _MuiTemperatureTuning
from muillm.modules.kvcache.cache_utils import MuiCache, MuiHybridChunkedCache
from muillm.modules.linear import MuiLinear
from muillm.modules.module import MuiModule
import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextAttention,
)

from muillm.modules.norm.qkl2norm import MuiQKL2Norm
from muillm.replacement.replacementcontext import MuiReplacementContext

import muillm_ext

logger = logging.get_logger(__name__)


def apply_temperature_tuning(
    query_states: torch.Tensor,
    cache_position: torch.LongTensor,
    attn_scale: float,
    floor_scale: float,
) -> torch.Tensor:
    if (query_states.is_cuda) and (
        (query_states.dtype == torch.float16) or (query_states.dtype == torch.bfloat16)
    ):
        # can dispatch to the custom kernel
        return _MuiTemperatureTuning.apply(
            query_states,
            cache_position,
            attn_scale,
            floor_scale,
        )
    else:
        bsz, num_attention_heads, q_len, head_dim = query_states.shape

        attn_scales = (
            torch.log(torch.floor((cache_position.float() + 1.0) / floor_scale) + 1.0)
            * attn_scale
            + 1.0
        )
        attn_scales = attn_scales.view((1, 1, q_len, 1)).expand(
            (bsz, 1, q_len, 1)
        )  # batch size > 1
        query_states = (query_states * attn_scales).to(query_states.dtype)

        return query_states


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) / math.sqrt(
        module.head_dim
    )
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class _MuiLlama4AttentionFullForward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module,
        cache_module,
        q,
        k,
        v,
        m,
        residual,
        position_embeddings,
        cache_positions,
    ):
        output = muillm_ext.muillm_llama4_attention_module_rope_forward(
            module,
            cache_module,
            q,
            k,
            v,
            m,
            residual,
            position_embeddings,
            cache_positions,
        )

        ctx.save_for_backward(q, k, v, m)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")


class _MuiLlama4Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, q, k, v, m, residual):
        output = muillm_ext.muillm_llama4_attention_module_forward(
            module,
            q,
            k,
            v,
            m,
            residual,
        )

        ctx.save_for_backward(q, k, v, m)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")


class MuiLlama4TextAttention(MuiModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        prev_module: Llama4TextAttention,
        qk_norm: Optional[MuiQKL2Norm],
        o_proj: MuiLinear,
    ):
        super().__init__(engine_config=engine_config)

        self.cpp_engine = engine_config.cpp_engine
        # the cpp module will be created at the end of all layer replacements
        # (set the field here before potential OOM errors so that it can still be manipulated in
        # the destructor)
        self.cpp_module = None

        self.config = prev_module.config
        self.layer_idx = prev_module.layer_idx
        self.head_dim = prev_module.head_dim
        self.num_attention_heads = prev_module.num_attention_heads
        self.num_key_value_groups = prev_module.num_key_value_groups
        self.num_key_value_heads = prev_module.num_key_value_heads
        self.scaling = prev_module.scaling
        self.attn_scale = prev_module.attn_scale
        self.floor_scale = prev_module.floor_scale
        self.attn_temperature_tuning = prev_module.attn_temperature_tuning
        self.attention_dropout = prev_module.attention_dropout
        self.is_causal = True
        self.use_rope = prev_module.use_rope
        self.o_proj = o_proj
        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = qk_norm

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        self.dispatchable = self.o_proj.dispatchable

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        if self.cpp_module is not None:
            muillm_ext.muillm_llama4_attention_module_deinit(self.cpp_module)

        use_qk_norm = hasattr(self, "qk_norm")
        use_temperature_tuning = self.attn_temperature_tuning and not self.use_rope

        self.cpp_module = muillm_ext.muillm_llama4_attention_module_init(
            self.cpp_engine,
            self.o_proj.cpp_module,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            bool(self.use_rope),
            use_qk_norm,
            self.qk_norm.variance_epsilon if use_qk_norm else 0.0,
            use_temperature_tuning,
            self.attn_scale,
            self.floor_scale,
            self.layer_idx,
        )

    def finalize_deinit(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_llama4_attention_module_deinit(self.cpp_module)
            self.cpp_module = None

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_module: Llama4TextAttention,
    ) -> "MuiLlama4TextAttention":
        engine_config = replacement_context.engine_config
        device = replacement_context.device
        qk_norm = None
        if hasattr(prev_module, "qk_norm"):
            qk_norm = MuiQKL2Norm.replace(
                replacement_context,
                prev_module.qk_norm,
            )

        new_o_proj = MuiLinear.replace(
            replacement_context,
            prev_module.o_proj,
        )

        return MuiLlama4TextAttention(
            engine_config=engine_config,
            prev_module=prev_module,
            qk_norm=qk_norm,
            o_proj=new_o_proj,
        )

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = query_states.size()

        if (q_len == 1) and self.dispatchable:
            if isinstance(past_key_value, MuiHybridChunkedCache):
                attn_output = _MuiLlama4AttentionFullForward.apply(
                    self.cpp_module,
                    past_key_value.cpp_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    residual,
                    position_embeddings,
                    cache_position,
                )
            else:
                # as q_len == 1, we can avoid the transposes
                query_states = query_states.view(
                    bsz, self.num_attention_heads, q_len, self.head_dim
                )
                key_states = key_states.view(
                    bsz, self.num_key_value_heads, q_len, self.head_dim
                )
                value_states = value_states.view(
                    bsz, self.num_key_value_heads, q_len, self.head_dim
                )

                # (rope and qk_norm commute as rope is a rotation)
                if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
                    query_states, key_states = self.qk_norm(query_states, key_states)

                # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
                if self.attn_temperature_tuning and not self.use_rope:
                    query_states = apply_temperature_tuning(
                        query_states,
                        cache_position,
                        self.attn_scale,
                        self.floor_scale,
                    )

                cache_kwargs = {
                    "cache_position": cache_position,
                }
                if self.use_rope and isinstance(past_key_value, MuiCache):
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    query_states, key_states, value_states = past_key_value.rope_update(
                        query_states,
                        key_states,
                        value_states,
                        position_embeddings,
                        self.layer_idx,
                        cache_kwargs,
                        complex_rope=True,
                    )
                else:
                    if self.use_rope:
                        query_states, key_states = apply_complex_rotary_emb(
                            query_states,
                            key_states,
                            position_embeddings,
                        )

                    if past_key_value is not None:
                        # sin and cos are specific to RoPE models; cache_position needed for the static cache
                        key_states, value_states = past_key_value.update(
                            key_states, value_states, self.layer_idx, cache_kwargs
                        )

                attn_output = _MuiLlama4Attention.apply(
                    self.cpp_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    residual,
                )

            attn_weights = None
        else:
            query_states = query_states.view(
                bsz, q_len, self.num_attention_heads, self.head_dim
            )
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            # (rope and qk_norm commute as rope is a rotation)
            if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
                query_states, key_states = self.qk_norm(query_states, key_states)

            # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
            if self.attn_temperature_tuning and not self.use_rope:
                query_states = apply_temperature_tuning(
                    query_states,
                    cache_position,
                    self.attn_scale,
                    self.floor_scale,
                )

            cache_kwargs = {
                "cache_position": cache_position,
            }
            if self.use_rope and isinstance(past_key_value, MuiCache):
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                query_states, key_states, value_states = past_key_value.rope_update(
                    query_states,
                    key_states,
                    value_states,
                    position_embeddings,
                    self.layer_idx,
                    cache_kwargs,
                    complex_rope=True,
                )
            else:
                if self.use_rope:
                    query_states, key_states = apply_complex_rotary_emb(
                        query_states,
                        key_states,
                        position_embeddings,
                    )

                if past_key_value is not None:
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )

            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get(
                    "output_attentions", False
                ):
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                        'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[
                        self.config._attn_implementation
                    ]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = self.o_proj(attn_output, residual=residual)

        return attn_output, attn_weights
