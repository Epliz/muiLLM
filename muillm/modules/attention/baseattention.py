import math
from typing import Optional, Tuple, Union
import warnings
import torch
import torch.nn as nn

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

logger = logging.get_logger(__name__)


class MuiBaseAttention(MuiModule):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        config: Union[LlamaConfig, MistralConfig],
        rotary_emb: MuiRotaryEmbedding,
        o_proj: MuiLinear,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(engine_config=engine_config)
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

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

    staticmethod

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
        prev_module: Union[LlamaAttention, MistralAttention],
        engine_config: MuiEngineConfig,
        device=None,
    ) -> "MuiBaseAttention":
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

        new_module = MuiBaseAttention(
            engine_config=engine_config,
            config=config,
            rotary_emb=rotary_emb,
            o_proj=new_o_proj,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )

        return new_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
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
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                # The mask has shape:
                # M: [B, 1, NEW_T, T]
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

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            # from shape [B, num_q_heads, T, embed_dim], go to [B, T, num_q_heads, embed_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            # from shape [B, T, num_q_heads, embed_dim] go to [B, T, hidden_size]
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output, residual=residual)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
