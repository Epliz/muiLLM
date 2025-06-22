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
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.attention.causaltransformerdecoding import (
    mui_causally_decode,
    mui_causally_decode_masked,
)
from muillm.modules.attention.rotaryembedding import _MuiComplexRotaryNoCache
from muillm.modules.attention.temperaturetuning import _MuiTemperatureTuning
from muillm.modules.module import MuiModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextAttention,
    Llama4TextL2Norm,
)
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

from muillm.modules.multilinear import MuiMultiLinear
from muillm.modules.norm.qkl2norm import MuiQKL2Norm

logger = logging.get_logger(__name__)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (xq.dtype == torch.float16) and (xq.is_cuda):
        # can dispatch to the custom kernel
        return _MuiComplexRotaryNoCache.apply(
            xq,
            xk,
            freqs_cis,
        )
    else:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xq_out = torch.view_as_real(xq_ * freqs_cis[:, None, :, :]).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis[:, None, :, :]).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)


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


class MuiLlama4TextAttention(MuiModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        prev_module: Llama4TextAttention,
        qk_norm: Optional[MuiQKL2Norm],
    ):
        super().__init__(engine_config=engine_config)
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
        self.o_proj = prev_module.o_proj
        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = qk_norm

    @staticmethod
    def replace(
        prev_module: Llama4TextAttention, engine_config: MuiEngineConfig, device=None
    ) -> "MuiLlama4TextAttention":
        qk_norm = None
        if hasattr(prev_module, "qk_norm"):
            qk_norm = MuiQKL2Norm.replace(
                prev_module.qk_norm, engine_config, device=device
            )
        return MuiLlama4TextAttention(
            engine_config=engine_config,
            prev_module=prev_module,
            qk_norm=qk_norm,
        )

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = query_states.size()

        if (
            (q_len == 1)
            and (query_states.dtype == torch.float16)
            and (query_states.is_cuda)
        ):
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

            if attention_mask is not None:
                attn_output = mui_causally_decode_masked(
                    query_states, key_states, value_states, attention_mask
                )
            else:
                attn_output = mui_causally_decode(
                    query_states, key_states, value_states
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
        attn_output = self.o_proj(attn_output, residual=residual)
        return attn_output, attn_weights
