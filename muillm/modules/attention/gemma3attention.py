import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, Union

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.attention.causaltransformerdecoding import (
    mui_causally_decode,
    mui_causally_decode_masked,
)
from muillm.modules.attention.rotaryembedding import apply_rotary_pos_emb
from muillm.modules.linear import MuiLinear
from muillm.modules.module import MuiModule
from muillm.modules.multilinear import MuiMultiLinear
from muillm.modules.norm.rmsnorm import MuiRMSNorm

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
)

from muillm.replacement.replacementcontext import MuiReplacementContext

logger = logging.get_logger(__name__)


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
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # Gemma 3 doesn't use softcapping, so we don't support it in MuiLLM
    if softcap is not None:
        raise ValueError("Softcapping is not supported.")

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class MuiGemma3Attention(MuiModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        prev_module: Gemma3Attention,
        q_norm: MuiRMSNorm,
        k_norm: MuiRMSNorm,
        o_proj: MuiLinear,
    ):
        super().__init__(engine_config=engine_config)

        config = prev_module.config
        layer_idx = prev_module.layer_idx

        self.is_sliding = bool((layer_idx + 1) % config.sliding_window_pattern)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        self.o_proj = o_proj

        self.attn_logit_softcapping = self.config.attn_logit_softcapping

        # Gemma 3 doesn't use softcapping, so we don't support it in MuiLLM
        if self.attn_logit_softcapping is not None:
            raise ValueError("Softcapping is not supported.")

        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = q_norm
        self.k_norm = k_norm

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext, prev_module: Gemma3Attention
    ) -> "MuiGemma3Attention":
        engine_config = replacement_context.engine_config

        # replace q and k norms
        q_norm = MuiRMSNorm.replace(
            replacement_context,
            prev_module.q_norm,
        )
        k_norm = MuiRMSNorm.replace(
            replacement_context,
            prev_module.k_norm,
        )

        o_proj = MuiLinear.replace(
            replacement_context,
            prev_module.o_proj,
        )

        return MuiGemma3Attention(
            engine_config=engine_config,
            prev_module=prev_module,
            q_norm=q_norm,
            k_norm=k_norm,
            o_proj=o_proj,
        )

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = query_states.size()

        if (
            (q_len == 1)
            and (
                (query_states.dtype == torch.float16)
                or (query_states.dtype == torch.bfloat16)
            )
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

            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

            cos, sin = position_embeddings
            # TODO: Make it use kernel
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                    "sliding_window": self.sliding_window,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            if attention_mask is not None:
                # TODO: try to remove it by guaranteeing it is the right size at model level
                attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
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
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                    "sliding_window": self.sliding_window,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

                # Here we need to slice as we use a static cache by default, but FA2 does not support it
                if (
                    attention_mask is not None
                    and self.config._attn_implementation == "flash_attention_2"
                ):
                    seq_len = attention_mask.shape[-1]
                    key_states, value_states = (
                        key_states[:, :, :seq_len, :],
                        value_states[:, :, :seq_len, :],
                    )

            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get(
                    "output_attentions", False
                ):
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                        "Falling back to eager attention. This warning can be removed using the argument "
                        '`attn_implementation="eager"` when loading the model.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[
                        self.config._attn_implementation
                    ]
            if attention_mask is not None:
                # backwards compatibility
                attention_mask = attention_mask.to(query_states)
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
