import math
from typing import List, Optional, Tuple, Union
import warnings
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.attention.baseattention import MuiBaseAttention
from muillm.modules.attention.parallelcausaltransformerdecoding import mui_parallel_causally_decode, mui_parallel_causally_decode_masked
from muillm.modules.parallelmultilinear import MuiParallelMultiLinear
import torch
import torch.nn as nn

import transformers.utils.logging as logging
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.llama.modeling_llama import LlamaAttention

from muillm.modules.attention.rotaryembedding import apply_rotary_pos_emb
from muillm.modules.attention.causaltransformerdecoding import mui_causally_decode
from muillm.modules.attention.kvcache import repeat_kv
from muillm.modules.attention.parallelbaseattention import MuiParallelBaseAttention

logger = logging.get_logger(__name__)

class MuiParallelSdpaAttention(MuiParallelBaseAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    @staticmethod
    def replace(prev_module: Union[MistralAttention, LlamaAttention, MuiBaseAttention], engine_config: MuiEngineConfig) -> "MuiParallelSdpaAttention":
        if isinstance(prev_module, MistralAttention) or isinstance(prev_module, LlamaAttention):
            device = prev_module.o_proj.weight.device
            dtype = prev_module.o_proj.weight.dtype
        else:
            raise ValueError(f"unsupported module type: {type(prev_module)}")


        layer_idx=prev_module.layer_idx
        config=prev_module.config

        rotary_embs = MuiParallelBaseAttention._create_rotary_embeddings(engine_config, config, layer_idx, device, dtype)

        new_module = MuiParallelSdpaAttention(engine_config=engine_config, config=config, rotary_embs=rotary_embs, layer_idx=layer_idx, device=device, dtype=dtype)
        new_module.o_proj.copy_module(prev_module=prev_module.o_proj)

        return new_module

    # Adapted from LlamaAttention.forward
    def parallel_forward(
        self,
        query_statess: Union[torch.Tensor, List[torch.Tensor]],
        key_statess: Union[torch.Tensor, List[torch.Tensor]],
        value_statess: Union[torch.Tensor, List[torch.Tensor]],
        attention_masks: Optional[List[torch.Tensor]] = None,
        position_idss: Optional[List[torch.LongTensor]] = None,
        past_key_values: Optional[List[Cache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_positions: Optional[List[torch.LongTensor]] = None,
        position_embeddingss: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,  # will become mandatory in v4.45
        all_ones_mask: Optional[bool] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[Cache]]]:
        if (attention_masks is not None) and (not isinstance(attention_masks, list)):
            raise ValueError("attention mask needs to be a list")
        if (position_idss is not None) and (not isinstance(position_idss, list)):
            raise ValueError("position_ids needs to be a list")
        if (past_key_values is not None) and (not isinstance(past_key_values, list)):
            raise ValueError("must pass list of caches")

        if output_attentions:
            raise ValueError("Not supported")

        sharded_inputs = isinstance(query_statess, list)
        if not sharded_inputs:
            query_statess = self.__shard_inputs(query_statess)
            key_statess = self.__shard_inputs(key_statess)
            value_statess = self.__shard_inputs(value_statess)
        else:
            # already sharded
            pass

        # q has the shape [B, NEW_T, num_q_heads * embed_dim]
        bsz, q_len, _ = query_statess[0].size()

        num_head_groups = len(query_statess)

        attn_outputs = []
        for d in range(num_head_groups):
            if (q_len == 1):
                # the transposition doesn nothing, so we can just avoid it to avoid the CPU cost of the operation
                query_states = query_statess[d].view(bsz, self.num_tp_heads, q_len, self.head_dim)
                key_states = key_statess[d].view(bsz, self.num_tp_key_value_heads, q_len, self.head_dim)
                value_states = value_statess[d].view(bsz, self.num_tp_key_value_heads, q_len, self.head_dim)
            else:
                query_states = query_statess[d].view(bsz, q_len, self.num_tp_heads, self.head_dim).transpose(1, 2)
                key_states = key_statess[d].view(bsz, q_len, self.num_tp_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_statess[d].view(bsz, q_len, self.num_tp_key_value_heads, self.head_dim).transpose(1, 2)

            pos_embeds = None
            if position_embeddingss is not None:
                coses, sines = position_embeddingss
                cos, sin = coses[d], sines[d]
                pos_embeds = cos, sin

            # Now the shapes are:
            # q: [B, num_q_heads, NEW_T, embed_dim]
            # k: [B, num_kv_heads, NEW_T, embed_dim]
            # v: [B, num_kv_heads, NEW_T, embed_dim]

            past_key_value = past_key_values[d] if past_key_values is not None else None
            query_states, key_states, value_states = self.rotary_embs[d].apply_rotary_pos_emb_write_kv_cache(
                query_states,
                key_states,
                position_idss[d],
                pos_embeds,
                value_states,
                past_key_value,
                cache_positions[d]
            )

            # Now the shapes are:
            # q: [B, num_q_heads, NEW_T, embed_dim]
            # k: [B, num_kv_heads, T, embed_dim]
            # v: [B, num_kv_heads, T, embed_dim]

            query_statess[d] = query_states
            key_statess[d] = key_states
            value_statess[d] = value_states

        if (q_len == 1) and (query_states[0].dtype == torch.float16):
            if all_ones_mask:
                attn_outputs = mui_parallel_causally_decode(query_statess, key_statess, value_statess)
            else:
                # The mask has shape:
                # M: [B, 1, NEW_T, T]
                # It contains 0 where OK, min_dtype where padded
                # min_dtype obtained with torch.finfo(dtype).min
                seq_len = key_statess[0].shape[-2]
                causal_masks = [causal_mask[:, :, :, : seq_len] for causal_mask in attention_masks]
                attn_outputs = mui_parallel_causally_decode_masked(query_statess, key_statess, value_statess, causal_masks)

            # from shape [B, num_q_heads, NEW_T, embed_dim], go to [B, NEW_T, num_q_heads, embed_dim]
            # then from shape [B, NEW_T, num_q_heads, embed_dim] go to [B, NEW_T, hidden_size]

            # q_len is 1 so we can remove the transposition
            attn_outputs = [attn_output.reshape(bsz, q_len, self.tp_hidden_size) for attn_output in attn_outputs]
        else:
            for d in range(num_head_groups):
                query_states = query_statess[d]
                key_states = key_statess[d]
                value_states = value_statess[d]

                # Before, key_states has the shape [B, num_k_heads, T, embed_dim]
                # after key_states has the shape [B, num_q_heads, T, embed_dim]
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                causal_mask = None
                if attention_masks is not None:
                    # The mask has shape:
                    # M: [B, 1, NEW_T, T]
                    # It contains 0 where OK, min_dtype where padded
                    # min_dtype obtained with torch.finfo(dtype).min
                    causal_mask = attention_masks[d] 
                    causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

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

                # Before, the shapes are:
                # A: [B, num_q_heads, NEW_T, embed_dim]
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(bsz, q_len, -1)

                # Now the shapes are:
                # A: [B, NEW_T, num_q_heads * embed_dim]
                attn_outputs.append(attn_output)

        attn_outputs = self.o_proj.parallel_forward(attn_outputs, residual=residual)

        return attn_outputs, None, past_key_values
