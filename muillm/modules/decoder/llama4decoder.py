from typing import Optional, Tuple, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.attention.llama4attention import MuiLlama4TextAttention
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
from muillm.modules.module import MuiModule
from muillm.replacement.replacementcontext import MuiReplacementContext

import torch

from transformers.models.llama4.modeling_llama4 import (
    Llama4TextAttention,
    Llama4TextDecoderLayer,
    Llama4TextMoe,
    Llama4TextMLP,
)

from muillm.modules.moe.gateupdownmlpmoe import MuiGateUpDownMLPMoe
from muillm.modules.multilinear import MuiMultiLinear


class MuiLlama4TextDecoderLayer(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        prev_module: Llama4TextDecoderLayer,
        qkv_proj: MuiMultiLinear,
        self_attn: MuiLlama4TextAttention,
        feed_forward: Union[MuiGateUpDownMLP, MuiGateUpDownMLPMoe],
    ):
        super().__init__(engine_config=engine_config)

        self.hidden_size = prev_module.hidden_size
        self.self_attn = self_attn
        self.use_chunked_attention = prev_module.use_chunked_attention  # <=> use rope
        self.is_moe_layer = (
            prev_module.is_moe_layer
        )  # the 128E model interleaves dense / sparse

        self.feed_forward = feed_forward

        self.qkv_proj = qkv_proj

        self.layer_idx = prev_module.layer_idx

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_module: Union[Llama4TextDecoderLayer, "MuiLlama4TextDecoderLayer"],
    ) -> "MuiLlama4TextDecoderLayer":
        engine_config = replacement_context.engine_config
        device = replacement_context.device
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiLlama4TextDecoderLayer):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_module

        prev_attn = prev_module.self_attn

        qkv_proj = None
        new_attention = None
        if isinstance(prev_attn, Llama4TextAttention):
            prev_q, prev_k, prev_v = (
                prev_attn.q_proj,
                prev_attn.k_proj,
                prev_attn.v_proj,
            )

            input_layernorm = prev_module.input_layernorm
            qkv_proj = MuiMultiLinear.replace(
                replacement_context,
                prev_modules=[prev_q, prev_k, prev_v],
                prev_layernorm_module=input_layernorm,
            )

            new_attention = MuiLlama4TextAttention.replace(
                replacement_context,
                prev_attn,
            )

        else:
            raise ValueError(f"Not supported {type(prev_module.self_attn)}")

        post_attention_layernorm = prev_module.post_attention_layernorm
        feed_forward = None
        if isinstance(prev_module.feed_forward, Llama4TextMoe) or isinstance(
            prev_module.feed_forward, MuiGateUpDownMLPMoe
        ):
            # MoE layer
            feed_forward = MuiGateUpDownMLPMoe.replace(
                replacement_context,
                prev_module=prev_module.feed_forward,
                prev_layernorm_module=post_attention_layernorm,
            )
        elif isinstance(prev_module.feed_forward, Llama4TextMLP) or isinstance(
            prev_module.feed_forward, MuiGateUpDownMLP
        ):
            # dense layer
            feed_forward = MuiGateUpDownMLP.replace(
                replacement_context,
                prev_module=prev_module.feed_forward,
                prev_layernorm_module=post_attention_layernorm,
            )
        else:
            raise ValueError(
                f"Unsupported replacement {type(prev_module.feed_forward)}"
            )

        new_module = MuiLlama4TextDecoderLayer(
            engine_config=engine_config,
            prev_module=prev_module,
            qkv_proj=qkv_proj,
            self_attn=new_attention,
            feed_forward=feed_forward,
        )

        # delete the previous modules to free memory
        del prev_module.self_attn
        del prev_module.feed_forward
        del prev_module.input_layernorm
        del prev_module.post_attention_layernorm

        # trigger garbage collection to free memory
        trigger_gc()

        return new_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        # Transform q, k, v
        # input layer norm is fused
        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        # use local attention mask for ROPE layers
        if self.use_chunked_attention and chunk_causal_mask is not None:
            attention_mask = chunk_causal_mask

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            residual=residual,
            **kwargs,
        )

        # Fully Connected
        residual = hidden_states

        # the post layer norm & residual are fused in the feed forward
        hidden_states = self.feed_forward(hidden_states, residual=residual)
        if self.is_moe_layer:
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
