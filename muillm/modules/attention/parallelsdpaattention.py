import math
from typing import List, Optional, Tuple, Union
import warnings
from muillm.modules.attention.parallelbaseattention import _MuiParallelAttention, _MuiParallelAttentionRope, MuiParallelBaseAttention
from muillm.modules.kvcache.cache_utils import MuiCache
from muillm.modules.parallellinear import MuiParallelLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers.utils.logging as logging
from transformers.cache_utils import Cache
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.configuration_llama import LlamaConfig

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.attention.rotaryembedding import MuiRotaryEmbedding
from muillm.modules.attention.kvcache import repeat_kv

logger = logging.get_logger(__name__)


class MuiParallelSdpaAttention(MuiParallelBaseAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    @staticmethod
    def _create_rotary_embeddings(engine_config: MuiEngineConfig, config: Union[LlamaConfig, MistralConfig], layer_idx:int, device=None, dtype=None) -> MuiRotaryEmbedding:

        rotary_emb = MuiRotaryEmbedding(
            engine_config,
            config,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        return rotary_emb

    @staticmethod
    def replace(prev_module: Union[LlamaAttention, MistralAttention], engine_config: MuiEngineConfig, device=None) -> "MuiParallelSdpaAttention":
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module.o_proj, MuiParallelLinear):
            device = prev_module.o_proj.weights[0].device if device is None else device
            dtype = prev_module.o_proj.weights[0].dtype
        else:
            device = prev_module.o_proj.weight.device if device is None else device
            dtype = prev_module.o_proj.weight.dtype

        layer_idx=prev_module.layer_idx
        config=prev_module.config

        rotary_emb = MuiParallelSdpaAttention._create_rotary_embeddings(engine_config, config, layer_idx, device, dtype)

        new_module = MuiParallelSdpaAttention(engine_config=engine_config, config=config, rotary_emb=rotary_emb, layer_idx=layer_idx, device=device, dtype=dtype)

        new_module.o_proj.copy_module(prev_module=prev_module.o_proj, device=device)

        return new_module

    def parallel_forward(
        self,
        query_states: Union[torch.Tensor, List[torch.Tensor]],
        key_states: Union[torch.Tensor, List[torch.Tensor]],
        value_states: Union[torch.Tensor, List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        all_ones_mask: Optional[bool] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
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

        if all_ones_mask is None:
            # if not specified, assume it might not have just ones
            all_ones_mask = False

        bsz, q_len, _ = query_states.size()

        # at this point, we have the following shapes:
        #  q: [B, num_q_heads, T, embed_dim]
        #  k: [B, num_k_heads, S, embed_dim]
        #  v: [B, num_v_heads, S, embed_dim]

        if (q_len == 1) and (query_states.dtype == torch.float16):
            #
            # if all_ones_mask or (attention_mask is None):
            #     attn_output = mui_causally_decode(query_states, key_states, value_states)
            # else:
            #     # The mask has shape:
            #     # M: [B, 1, S, T]
            #     # It contains 0 where OK, min_dtype where padded
            #     # min_dtype obtained with torch.finfo(dtype).min
            #     attn_output = mui_causally_decode_masked(query_states, key_states, value_states, attention_mask)

            # attn_output = self.o_proj.parallel_forward([attn_output], residual=residual)[0]

            # The mask has shape:
            # M: [B, 1, S, T]
            
            if self.dispatchable and isinstance(past_key_value, MuiCache):
                # can use the C++ module for doing rope + cache write + attention
                attn_output = _MuiParallelAttentionRope.apply(
                    self.cpp_module,
                    past_key_value.cpp_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    residual,
                    position_ids,
                    position_embeddings,
                    cache_position,
                )
            else:
                # as q_len is 1, we can avoid the transpose
                query_states = query_states.view(bsz, self.num_tp_heads, q_len, self.head_dim)
                key_states = key_states.view(bsz, self.num_tp_key_value_heads, q_len, self.head_dim)
                value_states = value_states.view(bsz, self.num_tp_key_value_heads, q_len, self.head_dim)

                query_states, key_states, value_states = self.rotary_emb.apply_rotary_pos_emb_write_kv_cache(
                    query_states,
                    key_states,
                    position_ids,
                    position_embeddings,
                    value_states,
                    past_key_value,
                    cache_position
                )
                attn_output = _MuiParallelAttention.apply(self.cpp_module, query_states, key_states, value_states, attention_mask, residual)
        else:
            if (q_len == 1):
                # as q_len is 1, we can avoid the transpose
                query_states = query_states.view(bsz, self.num_tp_heads, q_len, self.head_dim)
                key_states = key_states.view(bsz, self.num_tp_key_value_heads, q_len, self.head_dim)
                value_states = value_states.view(bsz, self.num_tp_key_value_heads, q_len, self.head_dim)
            else:
                query_states = query_states.view(bsz, q_len, self.num_tp_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_tp_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_tp_key_value_heads, self.head_dim).transpose(1, 2)

            query_states, key_states, value_states = self.rotary_emb.apply_rotary_pos_emb_write_kv_cache(
                query_states,
                key_states,
                position_ids,
                position_embeddings,
                value_states,
                past_key_value,
                cache_position
            )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            causal_mask = attention_mask

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )

            # from shape [B, num_q_heads, T, embed_dim], go to [B, T, num_q_heads, embed_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            # from shape [B, T, num_q_heads, embed_dim] go to [B, T, hidden_size]
            attn_output = attn_output.view(bsz, q_len, self.tp_hidden_size)

            attn_output = self.o_proj.parallel_forward([attn_output], residual=residual)[0]

        return [attn_output]

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        all_ones_mask: Optional[bool] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.tensor_parallelism > 1:
            attn_outputs = self.parallel_forward(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                all_ones_mask=all_ones_mask,
                residual=residual,
                **kwargs,
            )

            return attn_outputs[0]

        raise ValueError("Only parallel inference is supported")
