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
from muillm.modules.attention.llama4attention import (
    apply_rotary_emb,
    apply_temperature_tuning,
    eager_attention_forward,
)

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

from muillm.modules.parallelmultilinear import MuiParallelMultiLinear

logger = logging.get_logger(__name__)


class MuiParallelLlama4TextAttention(MuiModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        prev_module: Llama4TextAttention,
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
            self.qk_norm = prev_module.qk_norm

    @staticmethod
    def replace(
        prev_module: Llama4TextAttention, engine_config: MuiEngineConfig, device=None
    ) -> "MuiParallelLlama4TextAttention":

        return MuiParallelLlama4TextAttention(
            engine_config=engine_config,
            prev_module=prev_module,
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
                    position_embeddings.to(query_states.device),
                )

            # (rope and qk_norm commute as rope is a rotation)
            if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
                query_states = self.qk_norm(query_states)
                key_states = self.qk_norm(key_states)

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
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_output = mui_causally_decode_masked(
                    query_states, key_states, value_states, causal_mask
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

            if (
                self.use_rope
            ):  # the 16E model skips rope for long context on certain layers
                query_states, key_states = apply_rotary_emb(
                    query_states,
                    key_states,
                    position_embeddings.to(query_states.device),
                )

            # (rope and qk_norm commute as rope is a rotation)
            if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
                query_states = self.qk_norm(query_states)
                key_states = self.qk_norm(key_states)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

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
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
