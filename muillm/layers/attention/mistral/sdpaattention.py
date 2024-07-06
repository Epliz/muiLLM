import math
from typing import Optional, Tuple
import warnings
import torch
import torch.nn as nn

import transformers.utils.logging as logging
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import MistralSdpaAttention

from muillm.layers.attention.mistral.rotaryembedding import apply_rotary_pos_emb
from muillm.layers.attention.mistral.causaltransformerdecoding import mui_causally_decode
from muillm.layers.attention.mistral.kvcache import repeat_kv
from muillm.layers.attention.mistral.baseattention import MuiMistralAttention

logger = logging.get_logger(__name__)

class MuiMistralSdpaAttention(MuiMistralAttention):
    """
    Mistral attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MistralAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    @staticmethod
    def replace(prev_module: MistralSdpaAttention) -> "MuiMistralSdpaAttention":
        device = prev_module.q_proj.weight.device
        dtype = prev_module.q_proj.weight.dtype

        new_module = MuiMistralSdpaAttention(config=prev_module.config, layer_idx=prev_module.layer_idx, device=device, dtype=dtype)

        new_module.o_proj.copy_module(prev_module=prev_module.o_proj)

        return new_module

    # Adapted from MistralAttention.forward
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
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        query_states, key_states, value_states = self.rotary_emb.apply_rotary_pos_emb_write_kv_cache(query_states, key_states, position_ids, kv_seq_len, value_states, past_key_value)

        # at this point, we have the following shapes:
        #  q: [B, num_q_heads, T, embed_dim]
        #  k: [B, num_k_heads, NEW_T, embed_dim]
        #  v: [B, num_v_heads, NEW_T, embed_dim]
        if (bsz == 1) and (q_len == 1) and (attention_mask is None) and (query_states.dtype == torch.float16):
            #
            attn_output = mui_causally_decode(query_states, key_states, value_states)
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and attention_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
                is_causal=self.is_causal and attention_mask is None and q_len > 1,
            )

        # from shape [B, num_q_heads, T, embed_dim], go to [B, T, num_q_heads, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # from shape [B, T, num_q_heads, embed_dim] go to [B, T, hidden_size]
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        # when non-batched, could push the o_proj into v?
        # TODO: could be made 2x faster?
        attn_output = self.o_proj(attn_output, residual=residual)

        return attn_output, None, past_key_value