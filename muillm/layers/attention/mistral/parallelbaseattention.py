import math
from typing import Optional, Tuple, Union
import warnings
from muillm.layers.attention.mistral.baseattention import MuiMistralAttention
from muillm.layers.parallellinear import MuiParallelLinear
import torch
import torch.nn as nn

import transformers.utils.logging as logging
from transformers.cache_utils import Cache
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralAttention

from muillm.engineconfig import MuiEngineConfig
from muillm.layers.module import MuiModule
from muillm.layers.attention.mistral.rotaryembedding import MuiMistralRotaryEmbedding
from muillm.layers.attention.mistral.causaltransformerdecoding import mui_causally_decode
from muillm.layers.attention.mistral.kvcache import repeat_kv
from muillm.layers.linear import MuiLinear

logger = logging.get_logger(__name__)

class MuiParallelMistralAttention(MuiModule):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, engine_config: MuiEngineConfig, config: MistralConfig, layer_idx: Optional[int] = None, device=None, dtype=None):
        super().__init__(engine_config=engine_config)
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # TODO: determine what to change here
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

        # QKV will be sharded by rows so that it corresponds to having sharded heads
        self.o_proj = MuiParallelLinear(engine_config, self.num_heads * self.head_dim, self.hidden_size, bias=False, device=device, dtype=dtype, sharding_dim=1)

        self.rotary_emb = MuiMistralRotaryEmbedding(
            engine_config,
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def replace(prev_module: Union[MistralAttention, MuiMistralAttention], engine_config: MuiEngineConfig) -> "MuiParallelMistralAttention":
        if isinstance(prev_module, MistralAttention):
            device = prev_module.o_proj.weight.device
            dtype = prev_module.o_proj.weight.dtype
        elif isinstance(prev_module, MuiMistralAttention):
            device = prev_module.o_proj.device
            dtype = prev_module.o_proj.dtype
        else:
            raise ValueError(f"unsupported module type: {type(prev_module)}")

        new_module = MuiParallelMistralAttention(engine_config=engine_config, config=prev_module.config, layer_idx=prev_module.layer_idx, device=device, dtype=dtype)

        new_module.o_proj.copy_module(prev_module=prev_module.o_proj)

        return new_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        query_states, key_states, value_states = self.rotary_emb.apply_rotary_pos_emb_write_kv_cache(query_states, key_states, position_ids, kv_seq_len, value_states, past_key_value)

        # at this point, we have the following shapes:
        #  q: [B, num_q_heads, T, embed_dim]
        #  k: [B, num_k_heads, NEW_T, embed_dim]
        #  v: [B, num_v_heads, NEW_T, embed_dim]

        if (bsz == 1) and (q_len == 1) and all_ones_mask and (query_states.dtype == torch.float16):
            #
            attn_output = mui_causally_decode(query_states, key_states, value_states)
        else:
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )

                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
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