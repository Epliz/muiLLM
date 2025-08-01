import math
from typing import List, Optional, Tuple, Union
import warnings
from muillm.modules.kvcache.cache_utils import MuiCache
from muillm.modules.multilinear import MuiMultiLinear
from muillm.modules.parallellinear import MuiParallelLinear
from muillm.modules.parallelmultilinear import MuiParallelMultiLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers.utils.logging as logging
from transformers.cache_utils import Cache
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.configuration_llama import LlamaConfig

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.module import MuiModule
from muillm.modules.attention.rotaryembedding import MuiRotaryEmbedding
from muillm.modules.attention.causaltransformerdecoding import (
    mui_causally_decode,
    mui_causally_decode_masked,
)
from muillm.modules.attention.kvcache import repeat_kv
from muillm.modules.linear import MuiLinear
import muillm_ext
from muillm.replacement.replacementcontext import MuiReplacementContext

logger = logging.get_logger(__name__)


class _MuiParallelAttentionRope(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module,
        cache_module,
        q,
        k,
        v,
        m,
        residual,
        position_ids,
        position_embeddings,
        cache_positions,
    ):
        output = muillm_ext.muillm_parallel_attention_module_rope_forward(
            module,
            cache_module,
            q,
            k,
            v,
            m,
            residual,
            position_ids,
            position_embeddings,
            cache_positions,
        )

        ctx.save_for_backward(q, k, v, m)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")


class _MuiParallelAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, q, k, v, m, residual):
        output = muillm_ext.muillm_parallel_attention_module_forward(
            module, q, k, v, m, residual
        )

        ctx.save_for_backward(q, k, v, m)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")


class MuiParallelBaseAttention(MuiModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        config: Union[LlamaConfig, MistralConfig],
        rotary_emb: MuiRotaryEmbedding,
        o_proj: MuiParallelLinear,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(engine_config=engine_config)

        self.cpp_engine = engine_config.cpp_engine
        # the cpp module will be created at the end of all layer replacements
        # (set the field here before potential OOM errors so that it can still be manipulated in
        # the destructor)
        self.cpp_module = None
        self.comms = engine_config.comms
        self.tensor_parallelism = engine_config.tensor_parallelism

        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        if isinstance(config, LlamaConfig) and config.pretraining_tp > 1:
            raise ValueError("Not supported")

        self.attention_dropout = config.attention_dropout

        self.hidden_size = config.hidden_size
        self.tp_hidden_size = self.hidden_size // self.tensor_parallelism

        self.num_heads = config.num_attention_heads
        self.num_tp_heads = self.num_heads // self.tensor_parallelism

        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_tp_key_value_heads = (
            self.num_key_value_heads // self.tensor_parallelism
        )

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        if isinstance(config, LlamaConfig):
            attention_bias = config.attention_bias
        else:
            attention_bias = False

        self.o_proj = o_proj

        self.rotary_emb = rotary_emb

        self.sqrt_head_dim = math.sqrt(self.head_dim)

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        self.dispatchable = self.rotary_emb.dispatchable

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_attention_module_deinit(self.cpp_module)

        self.cpp_module = muillm_ext.muillm_parallel_attention_module_init(
            self.cpp_engine,
            self.comms.comms,
            self.rotary_emb.cpp_module,
            self.o_proj.cpp_module,
            self.num_tp_heads,
            self.num_tp_key_value_heads,
            self.head_dim,
        )

    def finalize_deinit(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_attention_module_deinit(self.cpp_module)
            self.cpp_module = None

    @staticmethod
    def _create_rotary_embeddings(
        engine_config: MuiEngineConfig,
        config: Union[LlamaConfig, MistralConfig],
        layer_idx: int,
        device=None,
        dtype=None,
    ) -> MuiRotaryEmbedding:

        rotary_emb = MuiRotaryEmbedding(
            engine_config,
            config,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        return rotary_emb

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_module: Union[LlamaAttention, MistralAttention],
    ) -> "MuiParallelBaseAttention":
        engine_config = replacement_context.engine_config
        device = replacement_context.device
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module.o_proj, MuiParallelLinear):
            device = prev_module.o_proj.weights[0].device if device is None else device
            dtype = prev_module.o_proj.weights[0].dtype
        else:
            device = prev_module.o_proj.weight.device if device is None else device
            dtype = prev_module.o_proj.weight.dtype

        layer_idx = prev_module.layer_idx
        config = prev_module.config

        rotary_emb = MuiParallelBaseAttention._create_rotary_embeddings(
            engine_config, config, layer_idx, device, dtype
        )

        new_o_proj = MuiParallelLinear.replace(
            replacement_context,
            prev_module.o_proj,
        )

        new_module = MuiParallelBaseAttention(
            engine_config=engine_config,
            config=config,
            rotary_emb=rotary_emb,
            o_proj=new_o_proj,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )

        return new_module

    def parallel_forward(
        self,
        query_states: Union[torch.Tensor, List[torch.Tensor]],
        key_states: Union[torch.Tensor, List[torch.Tensor]],
        value_states: Union[torch.Tensor, List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        all_ones_mask: Optional[bool] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
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

        if all_ones_mask is None:
            # if not specified, assume it might not have just ones
            all_ones_mask = False
        if all_ones_mask:
            attention_mask = None

        bsz, q_len, _ = query_states.size()

        # at this point, we have the following shapes:
        #  q: [B, num_q_heads, T, embed_dim]
        #  k: [B, num_k_heads, S, embed_dim]
        #  v: [B, num_v_heads, S, embed_dim]

        if (q_len == 1) and (
            (query_states.dtype == torch.float16)
            or (query_states.dtype == torch.bfloat16)
        ):
            #
            # if all_ones_mask or (attention_mask is None):
            #     attn_output = mui_causally_decode(query_states, key_states, value_states)
            # else:
            #     # The mask has shape:
            #     # M: [B, 1, S, T]
            #     # It contains 0 where OK, min_dtype where padded
            #     # min_dtype obtained with torch.finfo(dtype).min
            #     attn_output = mui_causally_decode_masked(query_states, key_states, value_states, attention_mask)

            # attn_output = self.o_proj.parallel_forward([attn_output], residual=residual)[0]

            # The mask has shape:
            # M: [B, 1, S, T]

            if self.dispatchable and isinstance(past_key_value, MuiCache):
                # can use the C++ module for doing rope + cache write + attention
                attn_output = _MuiParallelAttentionRope.apply(
                    self.cpp_module,
                    past_key_value.cpp_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    residual,
                    position_ids,
                    position_embeddings,
                    cache_position,
                )
            else:
                # as q_len is 1, we can avoid the transpose
                query_states = query_states.view(
                    bsz, self.num_tp_heads, q_len, self.head_dim
                )
                key_states = key_states.view(
                    bsz, self.num_tp_key_value_heads, q_len, self.head_dim
                )
                value_states = value_states.view(
                    bsz, self.num_tp_key_value_heads, q_len, self.head_dim
                )

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
                attn_output = _MuiParallelAttention.apply(
                    self.cpp_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    residual,
                )
        else:
            if q_len == 1:
                # as q_len is 1, we can avoid the transpose
                query_states = query_states.view(
                    bsz, self.num_tp_heads, q_len, self.head_dim
                )
                key_states = key_states.view(
                    bsz, self.num_tp_key_value_heads, q_len, self.head_dim
                )
                value_states = value_states.view(
                    bsz, self.num_tp_key_value_heads, q_len, self.head_dim
                )
            else:
                query_states = query_states.view(
                    bsz, q_len, self.num_tp_heads, self.head_dim
                ).transpose(1, 2)
                key_states = key_states.view(
                    bsz, q_len, self.num_tp_key_value_heads, self.head_dim
                ).transpose(1, 2)
                value_states = value_states.view(
                    bsz, q_len, self.num_tp_key_value_heads, self.head_dim
                ).transpose(1, 2)

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

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3))
                / self.sqrt_head_dim
            )

            if attention_mask is not None:  # no matter the length, we just slice it
                # The mask has shape:
                # M: [B, 1, S, T]
                # It contains 0 where OK, min_dtype where padded
                # min_dtype obtained with torch.finfo(dtype).min
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_tp_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_tp_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            # from shape [B, num_q_heads, T, embed_dim], go to [B, T, num_q_heads, embed_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            # from shape [B, T, num_q_heads, embed_dim] go to [B, T, hidden_size]
            attn_output = attn_output.view(bsz, q_len, self.tp_hidden_size)

            attn_output = self.o_proj.parallel_forward(
                [attn_output], residual=residual
            )[0]

        return [attn_output]

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
        ] = None,  # will become mandatory in v4.46
        all_ones_mask: Optional[bool] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.tensor_parallelism > 1:
            attn_outputs = self.parallel_forward(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                all_ones_mask=all_ones_mask,
                residual=residual,
                **kwargs,
            )

            return attn_outputs[0]

        raise ValueError("Only parallel inference is supported")
