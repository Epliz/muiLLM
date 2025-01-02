import math
from typing import List, Optional, Tuple, Union
import warnings
from muillm.layers.attention.mistral.baseattention import MuiMistralAttention
from muillm.layers.attention.mistral.parallelcausaltransformerdecoding import mui_parallel_causally_decode
import torch
import torch.nn as nn

import transformers.utils.logging as logging
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import MistralAttention

from muillm.engineconfig import MuiEngineConfig
from muillm.layers.attention.mistral.rotaryembedding import apply_rotary_pos_emb
from muillm.layers.attention.mistral.causaltransformerdecoding import mui_causally_decode
from muillm.layers.attention.mistral.kvcache import repeat_kv
from muillm.layers.attention.mistral.parallelbaseattention import MuiParallelMistralAttention

logger = logging.get_logger(__name__)

class MuiParallelMistralSdpaAttention(MuiParallelMistralAttention):
    """
    Mistral attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MistralAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    @staticmethod
    def replace(prev_module: Union[MistralAttention, MuiMistralAttention], engine_config: MuiEngineConfig) -> "MuiParallelMistralSdpaAttention":
        if isinstance(prev_module, MistralAttention):
            device = prev_module.o_proj.weight.device
            dtype = prev_module.o_proj.weight.dtype
        elif isinstance(prev_module, MuiMistralAttention):
            device = prev_module.o_proj.device
            dtype = prev_module.o_proj.dtype
        else:
            raise ValueError(f"unsupported module type: {type(prev_module)}")


        new_module = MuiParallelMistralSdpaAttention(engine_config=engine_config, config=prev_module.config, layer_idx=prev_module.layer_idx, device=device, dtype=dtype)

        new_module.o_proj.copy_module(prev_module=prev_module.o_proj)

        return new_module

    # Adapted from MistralAttention.forward
    def parallel_forward(
        self,
        query_states: Union[torch.Tensor, List[torch.Tensor]],
        key_states: Union[torch.Tensor, List[torch.Tensor]],
        value_states: Union[torch.Tensor, List[torch.Tensor]],
        attention_masks: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[List[torch.LongTensor]] = None,
        past_key_values: Optional[List[Cache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        all_ones_mask: Optional[bool] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[Cache]]]:
        if (attention_masks is not None) and (not isinstance(attention_masks, list)):
            raise ValueError("attention mask needs to be a list")
        if (position_ids is not None) and (not isinstance(position_ids, list)):
            raise ValueError("position_ids needs to be a list")
        if (past_key_values is not None) and (not isinstance(past_key_values, list)):
            raise ValueError("must pass list of caches")

        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MistralModel is using MistralSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_masks=attention_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                all_ones_mask=all_ones_mask,
            )

        sharded_inputs = isinstance(query_states, list)
        if not sharded_inputs:
            query_states = self.__shard_inputs(query_states)
            key_states = self.__shard_inputs(key_states)
            value_states = self.__shard_inputs(value_states)
        else:
            # already sharded
            query_states = query_states
            key_states = key_states
            value_states = value_states

        if all_ones_mask is None:
            # if not specified, assume it might not have just ones
            all_ones_mask = False

        num_head_groups = len(query_states)

        bsz, q_len, _ = query_states[0].size()

        query_states = [query_state.view(bsz, q_len, self.num_tp_heads, self.head_dim).transpose(1, 2) for query_state in query_states]
        key_states = [key_state.view(bsz, q_len, self.num_tp_key_value_heads, self.head_dim).transpose(1, 2) for key_state in key_states]
        value_states = [value_state.view(bsz, q_len, self.num_tp_key_value_heads, self.head_dim).transpose(1, 2) for value_state in value_states]

        kv_seq_len = key_states[0].shape[-2]
        if past_key_values is not None:
            kv_seq_len += past_key_values[0].get_usable_length(kv_seq_len, self.layer_idx)

        for d in range(len(query_states)):
            # do the rotary embeddings on each head group
            pos_ids = position_ids[d] if position_ids is not None else None
            past_key_value = past_key_values[d] if past_key_values is not None else None
            query_states[d], key_states[d], value_states[d] = self.rotary_embs[d].apply_rotary_pos_emb_write_kv_cache(query_states[d], key_states[d], pos_ids, kv_seq_len, value_states[d], past_key_value)

        # at this point, we have the following shapes:
        #  q: [B, num_q_heads, T, embed_dim]
        #  k: [B, num_k_heads, NEW_T, embed_dim]
        #  v: [B, num_v_heads, NEW_T, embed_dim]
        if (bsz == 1) and (q_len == 1) and all_ones_mask and (query_states[0].dtype == torch.float16):
            #
            attn_outputs = mui_parallel_causally_decode(query_states, key_states, value_states)
        else:
            attn_outputs = []
            for d in range(num_head_groups):
                key_state = repeat_kv(key_states[d], self.num_key_value_groups)
                value_state = repeat_kv(value_states[d], self.num_key_value_groups)
                query_state = query_states[d]

                attention_mask = None
                if attention_masks is not None:
                    attention_mask = attention_masks[d]
                    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                        )

                # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
                # Reference: https://github.com/pytorch/pytorch/issues/112577.
                if query_state.device.type == "cuda" and attention_masks is not None:
                    query_state = query_state.contiguous()
                    key_state = key_state.contiguous()
                    value_state = value_state.contiguous()

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_state,
                    key_state,
                    value_state,
                    attn_mask=attention_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
                    is_causal=self.is_causal and attention_mask is None and q_len > 1,
                )

                attn_outputs.append(attn_output)

        # from shape [B, num_q_heads, T, embed_dim], go to [B, T, num_q_heads, embed_dim]
        # then from shape [B, T, num_q_heads, embed_dim] go to [B, T, hidden_size]
        attn_outputs = [attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.tp_hidden_size) for attn_output in attn_outputs]

        attn_outputs = self.o_proj.parallel_forward(attn_outputs, residual=residual)

        return attn_outputs, None, past_key_values
        