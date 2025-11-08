import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.attention.gemma3attention import eager_attention_forward
from muillm.modules.rope.ropeops import apply_rotary_pos_emb
from muillm.modules.kvcache.cache_utils import MuiCache, MuiHybridChunkedCache
from muillm.modules.module import MuiModule
from muillm.modules.norm.qkrmsnorm import MuiQKRMSNorm

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
)

from muillm.modules.parallellinear import MuiParallelLinear
from muillm.replacement.replacementcontext import MuiReplacementContext

import muillm_ext

logger = logging.get_logger(__name__)


class _MuiParallelGemma3AttentionFullForward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module,
        cache_module,
        q,
        k,
        v,
        m,
        cos,
        sin,
        cache_positions,
    ):
        output = muillm_ext.muillm_parallel_gemma3_attention_module_rope_forward(
            module,
            cache_module,
            q,
            k,
            v,
            m,
            cos,
            sin,
            cache_positions,
        )

        ctx.save_for_backward(q, k, v, m)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")


class _MuiParallelGemma3Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, q, k, v, m):
        output = muillm_ext.muillm_parallel_gemma3_attention_module_forward(
            module,
            q,
            k,
            v,
            m,
        )

        ctx.save_for_backward(q, k, v, m)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")


class MuiParallelGemma3Attention(MuiModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        prev_module: Gemma3Attention,
        qk_norm: MuiQKRMSNorm,
        o_proj: MuiParallelLinear,
    ):
        super().__init__(engine_config=engine_config)

        self.cpp_engine = engine_config.cpp_engine
        # the cpp module will be created at the end of all layer replacements
        # (set the field here before potential OOM errors so that it can still be manipulated in
        # the destructor)
        self.cpp_module = None
        self.comms = engine_config.comms
        self.tensor_parallelism = engine_config.tensor_parallelism

        config = prev_module.config
        layer_idx = prev_module.layer_idx

        self.is_sliding = bool((layer_idx + 1) % config.sliding_window_pattern)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self.num_attention_heads = config.num_attention_heads
        self.num_tp_attention_heads = (
            self.num_attention_heads // self.tensor_parallelism
        )

        self.num_key_value_heads = config.num_key_value_heads
        self.num_tp_key_value_heads = (
            self.num_key_value_heads // self.tensor_parallelism
        )

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

        self.qk_norm = qk_norm

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        self.dispatchable = self.o_proj.dispatchable

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_gemma3_attention_module_deinit(self.cpp_module)

        if (
            (self.cpp_engine is None)
            or (self.comms.comms is None)
            or (self.o_proj.cpp_module is None)
        ):
            # cannot initialize the cpp module
            self.cpp_module = None
            return

        self.cpp_module = muillm_ext.muillm_parallel_gemma3_attention_module_init(
            self.cpp_engine,
            self.comms.comms,
            self.o_proj.cpp_module,
            self.num_tp_attention_heads,
            self.num_tp_key_value_heads,
            self.head_dim,
            self.qk_norm.q_weights,
            self.qk_norm.k_weights,
            self.qk_norm.variance_epsilon,
            self.qk_norm.weight_offset,
            self.layer_idx,
        )

    def finalize_deinit(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_gemma3_attention_module_deinit(self.cpp_module)
            self.cpp_module = None

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext, prev_module: Gemma3Attention
    ) -> "MuiParallelGemma3Attention":
        engine_config = replacement_context.engine_config

        # replace q and k norms
        qk_norm = MuiQKRMSNorm.replace(
            replacement_context,
            prev_module.q_norm,
            prev_module.k_norm,
        )

        o_proj = MuiParallelLinear.replace(
            replacement_context,
            prev_module.o_proj,
        )

        return MuiParallelGemma3Attention(
            engine_config=engine_config,
            prev_module=prev_module,
            qk_norm=qk_norm,
            o_proj=o_proj,
        )

    def parallel_forward(
        self,
        query_states: Union[torch.Tensor, List[torch.Tensor]],
        key_states: Union[torch.Tensor, List[torch.Tensor]],
        value_states: Union[torch.Tensor, List[torch.Tensor]],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        List[torch.Tensor], List[Optional[torch.Tensor]], Optional[Tuple[torch.Tensor]]
    ]:
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

        bsz, q_len, _ = query_states.size()

        cos, sin = position_embeddings

        if (q_len == 1) and self.dispatchable:
            if isinstance(past_key_value, MuiHybridChunkedCache):
                attn_output = _MuiParallelGemma3AttentionFullForward.apply(
                    self.cpp_module,
                    past_key_value.cpp_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    cos,
                    sin,
                    cache_position,
                )
            else:
                # as q_len == 1, we can avoid the transposes
                query_states = query_states.view(
                    bsz, self.num_tp_attention_heads, q_len, self.head_dim
                )
                key_states = key_states.view(
                    bsz, self.num_tp_key_value_heads, q_len, self.head_dim
                )
                value_states = value_states.view(
                    bsz, self.num_tp_key_value_heads, q_len, self.head_dim
                )

                query_states, key_states = self.qk_norm(query_states, key_states)

                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                    "sliding_window": self.sliding_window,
                }
                if isinstance(past_key_value, MuiCache):
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    query_states, key_states, value_states = past_key_value.rope_update(
                        query_states,
                        key_states,
                        value_states,
                        position_embeddings,
                        self.layer_idx,
                        cache_kwargs,
                    )
                else:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin
                    )

                    if past_key_value is not None:
                        # sin and cos are specific to RoPE models; cache_position needed for the static cache
                        key_states, value_states = past_key_value.update(
                            key_states, value_states, self.layer_idx, cache_kwargs
                        )

                attn_output = _MuiParallelGemma3Attention.apply(
                    self.cpp_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                )

            attn_weights = None
        else:
            query_states = query_states.view(
                bsz, q_len, self.num_tp_attention_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, self.num_tp_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_tp_key_value_heads, self.head_dim
            ).transpose(1, 2)

            query_states, key_states = self.qk_norm(query_states, key_states)

            cos, sin = position_embeddings

            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            if isinstance(past_key_value, MuiCache):
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                query_states, key_states, value_states = past_key_value.rope_update(
                    query_states,
                    key_states,
                    value_states,
                    position_embeddings,
                    self.layer_idx,
                    cache_kwargs,
                )
            else:
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

                if past_key_value is not None:
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
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
            attn_output = self.o_proj.parallel_forward([attn_output])[0]
        return [attn_output], [attn_weights]

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
        **kwargs,
    ) -> torch.Tensor:
        if self.tensor_parallelism > 1:
            attn_outputs, attn_weights = self.parallel_forward(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                residual=residual,
                **kwargs,
            )

            return attn_outputs[0], attn_weights[0]

        raise ValueError("Only parallel inference is supported")
