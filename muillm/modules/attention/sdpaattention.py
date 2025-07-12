import math
from typing import Optional, Tuple, Union
import warnings
import torch
import torch.nn as nn

import transformers.utils.logging as logging
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.llama.modeling_llama import LlamaAttention

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.attention.rotaryembedding import apply_rotary_pos_emb
from muillm.modules.attention.causaltransformerdecoding import (
    mui_causally_decode,
    mui_causally_decode_masked,
)
from muillm.modules.attention.kvcache import repeat_kv
from muillm.modules.attention.baseattention import MuiBaseAttention
from muillm.modules.linear import MuiLinear

logger = logging.get_logger(__name__)


# Modified to avoid a sync
def _ignore_causal_mask_sdpa(
    attention_mask: Optional[torch.Tensor],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
    all_ones_mask: Optional[bool] = None,
    is_training: bool = False,
) -> bool:
    """
    Detects whether the optional user-specified attention_mask & the automatically created causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the `attention_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is passed).
    """

    _, query_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
    key_value_length = query_length + past_key_values_length

    is_tracing = (
        torch.jit.is_tracing()
        or isinstance(inputs_embeds, torch.fx.Proxy)
        or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
    )

    ignore_causal_mask = False

    if attention_mask is None:
        # TODO: When tracing with TorchDynamo with fullgraph=True, the model is recompiled depending on the input shape, thus SDPA's `is_causal` argument is rightfully updated (see https://gist.github.com/fxmarty/1313f39037fc1c112508989628c57363). However, when using `torch.export` or
        # or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is hard-coded. If a user exports a model with q_len > 1, the exported model will hard-code `is_causal=True` which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108).
        # Thus, we only set `ignore_causal_mask = True` if the model is set to training.
        #
        # Besides, jit.trace can not handle the `q_len > 1` condition for `is_causal` (`TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not Tensor`).
        if (
            (is_training or not is_tracing)
            and (query_length == 1 or key_value_length == query_length)
            and (sliding_window is None or key_value_length < sliding_window)
        ):
            ignore_causal_mask = True
    elif sliding_window is None or key_value_length < sliding_window:
        if len(attention_mask.shape) == 4:
            return False
        elif (is_training or not is_tracing) and (
            ((all_ones_mask is not None) and all_ones_mask == True)
            or ((all_ones_mask is None) and torch.all(attention_mask == 1))
        ):
            if query_length == 1 or key_value_length == query_length:
                # For query_length == 1, causal attention and bi-directional attention are the same.
                ignore_causal_mask = True

            # Unfortunately, for query_length > 1 and key_value_length != query_length, we cannot generally ignore the attention mask, as SDPA causal mask generation
            # may be wrong. We will set `is_causal=False` in SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
            # Reference: https://github.com/pytorch/pytorch/issues/108108
            # TODO: maybe revisit this with https://github.com/pytorch/pytorch/pull/114823 in PyTorch 2.3.

    return ignore_causal_mask


class MuiSdpaAttention(MuiBaseAttention):
    """
    Mistral attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MistralAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    @staticmethod
    def replace(
        prev_module: Union[LlamaAttention, MistralAttention],
        engine_config: MuiEngineConfig,
        device=None,
    ) -> "MuiSdpaAttention":
        if device is None:
            raise ValueError("device was None")

        device = prev_module.o_proj.weight.device if device is None else device
        dtype = prev_module.o_proj.weight.dtype

        layer_idx = prev_module.layer_idx
        config = prev_module.config

        rotary_emb = MuiBaseAttention._create_rotary_embeddings(
            engine_config, config, layer_idx, device, dtype
        )

        new_o_proj = MuiLinear.replace(
            prev_module.o_proj,
            engine_config=engine_config,
            device=device,
        )

        new_module = MuiSdpaAttention(
            engine_config=engine_config,
            config=config,
            rotary_emb=rotary_emb,
            o_proj=new_o_proj,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )

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
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.45
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

        # TODO: optimization avoiding transpose for q_len==1
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # TODO: make sure it is restricted to seen tokens?
        query_states, key_states, value_states = (
            self.rotary_emb.apply_rotary_pos_emb_write_kv_cache(
                query_states,
                key_states,
                position_ids,
                position_embeddings,
                value_states,
                past_key_value,
                cache_position,
            )
        )

        # at this point, we have the following shapes:
        #  q: [B, num_q_heads, T, embed_dim]
        #  k: [B, num_k_heads, NEW_T, embed_dim]
        #  v: [B, num_v_heads, NEW_T, embed_dim]
        if (q_len == 1) and (
            (query_states.dtype == torch.float16)
            or (query_states.dtype == torch.bfloat16)
        ):
            #
            if all_ones_mask or (attention_mask is None):
                attn_output = mui_causally_decode(
                    query_states, key_states, value_states
                )
            else:
                # The mask has shape:
                # M: [B, 1, NEW_T, T]
                # It contains 0 where OK, min_dtype where padded
                # min_dtype obtained with torch.finfo(dtype).min
                attn_output = mui_causally_decode_masked(
                    query_states, key_states, value_states, attention_mask
                )

            # q_len is 1 so we can remove the transposition
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        else:
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
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        # when non-batched, could push the o_proj into v?
        # TODO: could be made 2x faster?
        attn_output = self.o_proj(attn_output, residual=residual)

        return attn_output, None, past_key_value
