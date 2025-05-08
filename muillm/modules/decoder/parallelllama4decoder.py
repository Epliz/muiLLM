from typing import Optional, Tuple, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.attention.parallelllama4attention import (
    MuiParallelLlama4TextAttention,
)
from muillm.modules.module import MuiModule

import torch

from transformers.models.llama4.modeling_llama4 import (
    Llama4TextAttention,
    Llama4TextDecoderLayer,
)

from muillm.modules.parallelmultilinear import MuiParallelMultiLinear


class MuiParallelLlama4TextDecoderLayer(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        prev_module: Llama4TextDecoderLayer,
        qkv_proj: MuiParallelMultiLinear,
        self_attn: MuiParallelLlama4TextAttention,
    ):
        super().__init__(engine_config=engine_config)

        self.hidden_size = prev_module.hidden_size
        self.self_attn = self_attn
        self.use_chunked_attention = prev_module.use_chunked_attention  # <=> use rope
        self.is_moe_layer = (
            prev_module.is_moe_layer
        )  # the 128E model interleaves dense / sparse

        self.feed_forward = prev_module.feed_forward

        self.qkv_proj = qkv_proj
        # TODO: fuse this
        self.post_attention_layernorm = prev_module.post_attention_layernorm

        self.layer_idx = prev_module.layer_idx

    @staticmethod
    def replace(
        prev_module: Union[Llama4TextDecoderLayer, "MuiParallelLlama4TextDecoderLayer"],
        engine_config: MuiEngineConfig,
        device=None,
    ) -> "MuiParallelLlama4TextDecoderLayer":
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiParallelLlama4TextDecoderLayer):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_module

        prev_attn = prev_module.self_attn

        input_layernorm = prev_module.input_layernorm
        qkv_proj = None
        new_attention = None
        if isinstance(prev_attn, Llama4TextAttention):
            qkv_proj = MuiParallelMultiLinear.replace(
                prev_modules=[prev_attn.q_proj, prev_attn.k_proj, prev_attn.v_proj],
                engine_config=engine_config,
                prev_layernorm_module=input_layernorm,
                device=device,
                sharding_dim=1,  # should switch to row-wise sharding once we split the attention heads
            )
            new_attention = MuiParallelLlama4TextAttention.replace(
                prev_module.self_attn, engine_config=engine_config, device=device
            )
        else:
            raise ValueError(f"Not supported {type(prev_module.self_attn)}")

        new_module = MuiParallelLlama4TextDecoderLayer(
            engine_config=engine_config,
            prev_module=prev_module,
            qkv_proj=qkv_proj,
            self_attn=new_attention,
        )

        # delete the previous module to save memory
        del prev_module

        # trigger GC to save memory
        trigger_gc()

        return new_module

    # TODO: parallel forward

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
        attention_states, self_attn_weights = self.self_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + attention_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.view(residual.shape)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
