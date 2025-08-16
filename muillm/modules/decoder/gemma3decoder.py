from typing import Optional, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.attention.gemma3attention import MuiGemma3Attention
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
from muillm.modules.module import MuiModule

from transformers.cache_utils import Cache
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
)
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

import torch

from muillm.modules.multilinear import MuiMultiLinear
from muillm.modules.norm.rmsnorm import MuiRMSNorm
from muillm.replacement.replacementcontext import MuiReplacementContext


class MuiGemma3DecoderLayer(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        config: Gemma3TextConfig,
        qkv_proj: MuiMultiLinear,
        self_attn: MuiGemma3Attention,
        mlp: MuiGateUpDownMLP,
        post_attention_layernorm: MuiRMSNorm,
        post_feedforward_layernorm: MuiRMSNorm,
        layer_idx: int,
    ):
        super().__init__(engine_config=engine_config)

        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.qkv_proj = qkv_proj
        self.self_attn = self_attn
        self.mlp = mlp
        self.post_attention_layernorm = post_attention_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm

        self.is_sliding = self.self_attn.is_sliding
        self.sliding_window = config.sliding_window

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_module: Union[Gemma3DecoderLayer, "MuiGemma3DecoderLayer"],
    ) -> "MuiGemma3DecoderLayer":
        engine_config = replacement_context.engine_config
        device = replacement_context.device

        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiGemma3DecoderLayer):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_module

        config = prev_module.config
        layer_idx = prev_module.layer_idx

        prev_attn = prev_module.self_attn

        qkv_proj = None
        self_attn = None
        if isinstance(prev_attn, Gemma3Attention):
            # When using tensor parallelism, we shard the attention by head, so we need to shard qkv by rows
            qkv_proj = MuiMultiLinear.replace(
                replacement_context,
                prev_modules=[prev_attn.q_proj, prev_attn.k_proj, prev_attn.v_proj],
                prev_layernorm_module=prev_module.input_layernorm,
            )
            self_attn = MuiGemma3Attention.replace(
                replacement_context,
                prev_module.self_attn,
            )
        else:
            raise ValueError(f"Not supported {type(prev_module.self_attn)}")

        post_attention_layernorm = MuiRMSNorm.replace(
            replacement_context,
            prev_module.post_attention_layernorm,
        )

        mlp = MuiGateUpDownMLP.replace(
            replacement_context,
            prev_module.mlp,
            prev_layernorm_module=prev_module.pre_feedforward_layernorm,
        )

        post_feedforward_layernorm = MuiRMSNorm.replace(
            replacement_context,
            prev_module.post_feedforward_layernorm,
        )

        new_decoder = MuiGemma3DecoderLayer(
            engine_config=engine_config,
            config=config,
            qkv_proj=qkv_proj,
            self_attn=self_attn,
            mlp=mlp,
            post_attention_layernorm=post_attention_layernorm,
            post_feedforward_layernorm=post_feedforward_layernorm,
            layer_idx=layer_idx,
        )

        # delete the previous modules to free memory
        del prev_module.self_attn
        del prev_module.mlp
        del prev_module.input_layernorm
        del prev_module.post_attention_layernorm
        del prev_module.pre_feedforward_layernorm
        del prev_module.post_feedforward_layernorm

        trigger_gc()

        return new_decoder

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if self.is_sliding and (attention_mask is not None):
            if sliding_window_mask is not None:
                # we have a sliding window mask, so we can use it directly
                attention_mask = sliding_window_mask
            else:
                # efficient SDPA and no padding
                # In prefill, we may be larger than sliding window
                effective_seq_len = max(cache_position.shape[0], self.sliding_window)
                # For FA2, the mask is 2D and is of shape [bs, processed_tokens] (not [bs, max_cache_len]),
                # thus we must slice from the right (at most `effective_seq_len` elements)
                if self.config._attn_implementation == "flash_attention_2":
                    attention_mask = attention_mask[:, -effective_seq_len:]
                # Otherwise, the mask is 4D of shape [bs, 1, query_len, max_cache_len] thus we must slice
                # from the left, with an offset if we are beyond the sliding window
                else:
                    min_dtype = torch.finfo(attention_mask.dtype).min
                    sliding_window_mask = torch.tril(
                        torch.ones_like(attention_mask, dtype=torch.bool),
                        diagonal=-self.sliding_window,
                    )
                    attention_mask = torch.where(
                        sliding_window_mask, min_dtype, attention_mask
                    )
                    # In case we are beyond the sliding window, we need to correctly offset the mask slicing
                    offset = cache_position[-1] - effective_seq_len + 1
                    # Should only be used when beyond the sliding window (i.e. offset > 0)
                    offset = torch.clamp(offset, min=0)
                    # equivalent to: `attention_mask = attention_mask[:, :, :, offset : offset + effective_seq_len]`,
                    # but without data-dependent slicing (i.e. torch.compile friendly)
                    mask_indexes = torch.arange(
                        min(effective_seq_len, attention_mask.shape[-1]),
                        device=attention_mask.device,
                    )
                    mask_indexes += offset
                    attention_mask = attention_mask[:, :, :, mask_indexes]

        residual = hidden_states

        # Transform q, k, v
        # input layer norm is fused
        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        hidden_states, self_attn_weights = self.self_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        # the pre_feedforward_layernorm is fused in the MLP
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
