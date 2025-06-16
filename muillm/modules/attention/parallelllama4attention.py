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
from typing import Callable, List, Optional, Tuple, Union

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.attention.causaltransformerdecoding import (
    mui_causally_decode,
    mui_causally_decode_masked,
)
from muillm.modules.attention.llama4attention import (
    apply_rotary_emb,
    apply_temperature_tuning,
    eager_attention_forward,
)

from muillm.modules.kvcache.cache_utils import MuiHybridChunkedCache
from muillm.modules.module import MuiModule
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextAttention,
)

from muillm.modules.parallellinear import MuiParallelLinear
from muillm.modules.parallelmultilinear import MuiParallelMultiLinear
from muillm.modules.norm.qkl2norm import MuiQKL2Norm


import muillm_ext

logger = logging.get_logger(__name__)


class _MuiParallelLlama4AttentionFullForward(torch.autograd.Function):
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
        output = muillm_ext.muillm_parallel_llama4_attention_module_rope_forward(
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


class _MuiParallelLlama4Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, q, k, v, m, residual):
        output = muillm_ext.muillm_parallel_llama4_attention_module_forward(
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


class MuiParallelLlama4TextAttention(MuiModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        prev_module: Llama4TextAttention,
        o_proj: MuiParallelLinear,
        qk_norm: Optional[MuiQKL2Norm],
    ):
        super().__init__(engine_config=engine_config)

        self.cpp_engine = engine_config.cpp_engine
        # the cpp module will be created at the end of all layer replacements
        # (set the field here before potential OOM errors so that it can still be manipulated in
        # the destructor)
        self.cpp_module = None
        self.comms = engine_config.comms
        self.tensor_parallelism = engine_config.tensor_parallelism

        self.config = prev_module.config
        self.layer_idx = prev_module.layer_idx

        self.head_dim = prev_module.head_dim

        self.num_attention_heads = prev_module.num_attention_heads
        self.num_tp_attention_heads = (
            self.num_attention_heads // self.tensor_parallelism
        )

        self.num_key_value_groups = prev_module.num_key_value_groups
        self.num_key_value_heads = prev_module.num_key_value_heads
        self.num_tp_key_value_heads = (
            self.num_key_value_heads // self.tensor_parallelism
        )

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
            muillm_ext.muillm_parallel_llama4_attention_module_deinit(self.cpp_module)

        use_qk_norm = hasattr(self, "qk_norm")
        use_temperature_tuning = self.attn_temperature_tuning and not self.use_rope

        self.cpp_module = muillm_ext.muillm_parallel_llama4_attention_module_init(
            self.cpp_engine,
            self.comms.comms,
            self.o_proj.cpp_module,
            self.num_tp_attention_heads,
            self.num_tp_key_value_heads,
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
            muillm_ext.muillm_parallel_llama4_attention_module_deinit(self.cpp_module)
            self.cpp_module = None

    @staticmethod
    def replace(
        prev_module: Llama4TextAttention, engine_config: MuiEngineConfig, device=None
    ) -> "MuiParallelLlama4TextAttention":
        qk_norm = None
        if hasattr(prev_module, "qk_norm"):
            qk_norm = MuiQKL2Norm.replace(
                prev_module.qk_norm, engine_config, device=device
            )

        new_o_proj = MuiParallelLinear.replace(
            prev_module.o_proj,
            engine_config=engine_config,
            device=device,
        )

        return MuiParallelLlama4TextAttention(
            engine_config=engine_config,
            prev_module=prev_module,
            o_proj=new_o_proj,
            qk_norm=qk_norm,
        )

    def parallel_forward(
        self,
        query_states: Union[torch.Tensor, List[torch.Tensor]],
        key_states: Union[torch.Tensor, List[torch.Tensor]],
        value_states: Union[torch.Tensor, List[torch.Tensor]],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        List[torch.Tensor], List[Optional[torch.Tensor]], Optional[Tuple[torch.Tensor]]
    ]:
        # unwrap if needed
        if isinstance(query_states, list):
            query_states = query_states[0]
        else:
            raise ValueError("sharding not implemented")
        if isinstance(key_states, list):
            key_states = key_states[0]
        else:
            raise ValueError("sharding not implemented")
        if isinstance(value_states, list):
            value_states = value_states[0]
        else:
            raise ValueError("sharding not implemented")

        bsz, q_len, _ = query_states.size()

        if (q_len == 1) and self.dispatchable:

            if isinstance(past_key_value, MuiHybridChunkedCache):
                attn_output = _MuiParallelLlama4AttentionFullForward.apply(
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
                    bsz, self.num_tp_attention_heads, q_len, self.head_dim
                )
                key_states = key_states.view(
                    bsz, self.num_tp_key_value_heads, q_len, self.head_dim
                )
                value_states = value_states.view(
                    bsz, self.num_tp_key_value_heads, q_len, self.head_dim
                )

                if (
                    self.use_rope
                ):  # the 16E model skips rope for long context on certain layers
                    query_states, key_states = apply_rotary_emb(
                        query_states,
                        key_states,
                        position_embeddings,
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

                query_states = query_states
                key_states = key_states

                if past_key_value is not None:
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    cache_kwargs = {"cache_position": cache_position}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )

                attn_output = _MuiParallelLlama4Attention.apply(
                    self.cpp_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    residual=residual,
                )

            attn_weights = None
        else:
            query_states = query_states.view(
                bsz, q_len, self.num_tp_attention_heads, self.head_dim
            )
            key_states = key_states.view(
                bsz, q_len, self.num_tp_key_value_heads, self.head_dim
            )
            value_states = value_states.view(
                bsz, q_len, self.num_tp_key_value_heads, self.head_dim
            )

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            if (
                self.use_rope
            ):  # the 16E model skips rope for long context on certain layers
                query_states, key_states = apply_rotary_emb(
                    query_states,
                    key_states,
                    position_embeddings,
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

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"cache_position": cache_position}
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
            attn_output = self.o_proj.parallel_forward(
                [attn_output],
                residual=residual,
            )[0]

        return [attn_output], [attn_weights]

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        if self.tensor_parallelism > 1:
            attn_outputs, attn_weights = self.parallel_forward(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                residual=residual,
                **kwargs,
            )

            return attn_outputs[0], attn_weights[0]

        raise ValueError("Only parallel inference is supported")
