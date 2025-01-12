import math
from typing import Optional, Tuple, Union
import warnings
import torch
import torch.nn as nn

import transformers.utils.logging as logging
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import MistralSdpaAttention
from transformers.models.llama.modeling_llama import LlamaSdpaAttention

from muillm.engineconfig import MuiEngineConfig
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
    def replace(prev_module: Union[LlamaSdpaAttention, MistralSdpaAttention], engine_config: MuiEngineConfig) -> "MuiMistralSdpaAttention":
        device = prev_module.q_proj.weight.device
        dtype = prev_module.q_proj.weight.dtype

        layer_idx=prev_module.layer_idx
        config=prev_module.config

        rotary_emb = MuiMistralAttention._create_rotary_embeddings(engine_config, config, layer_idx, device, dtype)

        new_module = MuiMistralSdpaAttention(engine_config=engine_config, config=config, rotary_emb=rotary_emb, layer_idx=layer_idx, device=device, dtype=dtype)

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
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        all_ones_mask: Optional[bool] = None,
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
                cache_position=cache_position,
                all_ones_mask=all_ones_mask,
            )

        if all_ones_mask is None:
            # if not specified, assume it might not have just ones
            all_ones_mask = False


        bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        query_states, key_states, value_states = self.rotary_emb.apply_rotary_pos_emb_write_kv_cache(
            query_states,
            key_states,
            position_ids,
            position_embeddings,
            kv_seq_len,
            value_states,
            past_key_value,
            cache_position
        )

        # at this point, we have the following shapes:
        #  q: [B, num_q_heads, T, embed_dim]
        #  k: [B, num_k_heads, NEW_T, embed_dim]
        #  v: [B, num_v_heads, NEW_T, embed_dim]
        if (bsz == 1) and (q_len == 1) and all_ones_mask and (query_states.dtype == torch.float16):
            #
            attn_output = mui_causally_decode(query_states, key_states, value_states)
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)


            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )

        # from shape [B, num_q_heads, T, embed_dim], go to [B, T, num_q_heads, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # from shape [B, T, num_q_heads, embed_dim] go to [B, T, hidden_size]
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        # when non-batched, could push the o_proj into v?
        # TODO: could be made 2x faster?
        attn_output = self.o_proj(attn_output, residual=residual)

        return attn_output, None, past_key_value