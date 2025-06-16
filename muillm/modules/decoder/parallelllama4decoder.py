from typing import List, Optional, Tuple, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.attention.parallelllama4attention import (
    MuiParallelLlama4TextAttention,
)
from muillm.modules.kvcache.cache_utils import MuiCache
from muillm.modules.module import MuiModule

import torch

from transformers.models.llama4.modeling_llama4 import (
    Llama4TextAttention,
    Llama4TextDecoderLayer,
    Llama4TextMoe,
    Llama4TextMLP,
)

from muillm.modules.moe.parallelgateupdownmlpmoe import MuiParallelGateUpDownMLPMoe
from muillm.modules.parallelgateupdownmlp import MuiParallelGateUpDownMLP
from muillm.modules.parallelmultilinear import MuiParallelMultiLinear


import muillm_ext


class _MuiParallelLlama4Decoder(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module,
        cache_module,
        h,
        attention_mask,
        chunk_causal_mask,
        position_embeddings,
        cache_positions,
    ):
        output = muillm_ext.muillm_parallel_llama4_decoder_module_forward(
            module,
            cache_module,
            h,
            attention_mask,
            chunk_causal_mask,
            position_embeddings,
            cache_positions,
        )

        ctx.save_for_backward(h, attention_mask, chunk_causal_mask, position_embeddings)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")


class MuiParallelLlama4TextDecoderLayer(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        prev_module: Llama4TextDecoderLayer,
        qkv_proj: MuiParallelMultiLinear,
        self_attn: MuiParallelLlama4TextAttention,
        feed_forward: Union[MuiParallelGateUpDownMLP, MuiParallelGateUpDownMLPMoe],
    ):
        super().__init__(engine_config=engine_config)

        self.cpp_engine = engine_config.cpp_engine
        # the cpp module will be created at the end of all layer replacements
        # (set the field here before potential OOM errors so that it can still be manipulated in
        # the destructor)
        self.cpp_module = None
        self.comms = engine_config.comms
        self.tensor_parallelism = engine_config.tensor_parallelism

        self.hidden_size = prev_module.hidden_size
        self.self_attn = self_attn
        self.use_chunked_attention = prev_module.use_chunked_attention  # <=> use rope
        self.is_moe_layer = (
            prev_module.is_moe_layer
        )  # the 128E model interleaves dense / sparse

        self.feed_forward = feed_forward

        self.qkv_proj = qkv_proj

        self.layer_idx = prev_module.layer_idx

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        self.dispatchable = (
            self.self_attn.dispatchable and self.feed_forward.dispatchable
        )

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # initialize the cpp module
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_llama4_decoder_module_deinit(self.cpp_module)

        self.cpp_module = muillm_ext.muillm_parallel_llama4_decoder_module_init(
            self.cpp_engine,
            self.comms.comms,
            self.qkv_proj.cpp_module,
            self.self_attn.cpp_module,
            self.feed_forward.cpp_module,
            self.use_chunked_attention,
        )

    def finalize_deinit(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_llama4_decoder_module_deinit(self.cpp_module)
            self.cpp_module = None

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

        # severe the reference to be able to delete the previous module
        prev_module.self_attn = None

        qkv_proj = None
        new_attention = None
        if isinstance(prev_attn, Llama4TextAttention):
            prev_q, prev_k, prev_v = (
                prev_attn.q_proj,
                prev_attn.k_proj,
                prev_attn.v_proj,
            )

            # severe the reference to be able to delete the previous module
            prev_attn.q_proj = None
            prev_attn.k_proj = None
            prev_attn.v_proj = None

            input_layernorm = prev_module.input_layernorm
            qkv_proj = MuiParallelMultiLinear.replace(
                prev_modules=[prev_q, prev_k, prev_v],
                prev_layernorm_module=input_layernorm,
                engine_config=engine_config,
                device=device,
                sharding_dim=0,  # row-wise sharding to split attention heads
            )

            del prev_q
            del prev_k
            del prev_v

            # trigger GC to save memory
            trigger_gc()

            new_attention = MuiParallelLlama4TextAttention.replace(
                prev_attn, engine_config=engine_config, device=device
            )

            del prev_attn

            # trigger GC to save memory
            trigger_gc()
        else:
            raise ValueError(f"Not supported {type(prev_module.self_attn)}")

        post_attention_layernorm = prev_module.post_attention_layernorm
        feed_forward = None
        if isinstance(prev_module.feed_forward, Llama4TextMoe) or isinstance(
            prev_module.feed_forward, MuiParallelGateUpDownMLPMoe
        ):
            # MoE layer
            feed_forward = MuiParallelGateUpDownMLPMoe.replace(
                prev_module=prev_module.feed_forward,
                engine_config=engine_config,
                prev_layernorm_module=post_attention_layernorm,
                device=device,
            )
        elif isinstance(prev_module.feed_forward, Llama4TextMLP) or isinstance(
            prev_module.feed_forward, MuiParallelGateUpDownMLP
        ):
            # dense layer
            feed_forward = MuiParallelGateUpDownMLP.replace(
                prev_module=prev_module.feed_forward,
                engine_config=engine_config,
                prev_layernorm_module=post_attention_layernorm,
                device=device,
            )
        else:
            raise ValueError(
                f"Unsupported replacement {type(prev_module.feed_forward)}"
            )

        new_module = MuiParallelLlama4TextDecoderLayer(
            engine_config=engine_config,
            prev_module=prev_module,
            qkv_proj=qkv_proj,
            self_attn=new_attention,
            feed_forward=feed_forward,
        )

        # delete the previous module to save memory
        del prev_module

        # trigger GC to save memory
        trigger_gc()

        return new_module

    def parallel_forward(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        chunk_causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[MuiCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[
        List[torch.FloatTensor], Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        if output_attentions:
            raise ValueError("output_attention is not supported")

        if output_router_logits:
            raise ValueError("output_router_logits is not supported")

        # unwrap inputs if needed
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[0]

        bsz, q_len, _ = hidden_states.size()
        if (
            self.dispatchable
            and (bsz == 1)
            and (q_len == 1)
            and isinstance(past_key_value, MuiCache)
        ):
            hidden_states = _MuiParallelLlama4Decoder.apply(
                self.cpp_module,
                past_key_value.cpp_module,
                hidden_states,
                attention_mask,
                chunk_causal_mask,
                position_embeddings,
                cache_position,
            )
        else:
            residual = hidden_states

            # Transform q, k, v
            # input layer norm is fused
            query_states, key_states, value_states = self.qkv_proj.parallel_forward(
                [hidden_states], collect_outputs=False
            )[0]

            # use local attention mask for ROPE layers
            if self.use_chunked_attention:
                attention_mask = chunk_causal_mask

            # Self Attention
            hidden_states, _ = self.self_attn.parallel_forward(
                query_states=[query_states],
                key_states=[key_states],
                value_states=[value_states],
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                residual=residual,
                **kwargs,
            )

            hidden_states = hidden_states[0]

            # Fully Connected
            residual = hidden_states

            # the post layer norm & residual are fused in the feed forward
            hidden_states = self.feed_forward.parallel_forward(
                [hidden_states],
                residual=residual,
            )
            if self.is_moe_layer:
                hidden_states, router_logits = hidden_states

                hidden_states = hidden_states[0]
            else:
                hidden_states = hidden_states[0]
                router_logits = None

        outputs = ([hidden_states],)

        return outputs

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
        if self.tensor_parallelism > 1:
            layer_outputs = self.parallel_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                chunk_causal_mask=chunk_causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0][0]

            final_outputs = (hidden_states,)

            if output_attentions:
                attn_weights = (layer_outputs[1][0],)
                final_outputs += (attn_weights,)

            if output_router_logits:
                router_logits = (layer_outputs[2][0],)
                final_outputs += (router_logits,)

            return final_outputs

        raise ValueError("Only parallel inference is supported")
