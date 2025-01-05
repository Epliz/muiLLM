import math
from typing import List, Optional, Tuple, Union
import warnings
from muillm.layers.attention.mistral.baseattention import MuiMistralAttention
from muillm.layers.attention.mistral.parallelcausaltransformerdecoding import mui_parallel_causally_decode
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
        self.tensor_parallelism = engine_config.tensor_parallelism

        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.tp_hidden_size = self.hidden_size // self.tensor_parallelism

        self.num_heads = config.num_attention_heads
        self.num_tp_heads = self.num_heads // self.tensor_parallelism

        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_tp_key_value_heads = self.num_key_value_heads // self.tensor_parallelism

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_tp_heads * self.tensor_parallelism) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # QKV will be sharded by rows so that it corresponds to having sharded heads
        self.o_proj = MuiParallelLinear(engine_config, self.num_heads * self.head_dim, self.hidden_size, bias=False, device=device, dtype=dtype, sharding_dim=1)

        # one rotary cache per device
        self.rotary_embs = [MuiMistralRotaryEmbedding(
            engine_config,
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            layer_idx=layer_idx,
            device=d,
            dtype=dtype,
        ) for d in self.engine_config.devices]

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

    def __shard_inputs(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        # if we are sharding along the k-dim, we need to shard the input accordingly
        tensors = torch.tensor_split(tensor, self.tensor_parallelism, -1)
        return MuiParallelLinear._transfer_across(self.engine_config, tensors)

    def parallel_forward(
        self,
        query_states: Union[torch.Tensor, List[torch.Tensor]],
        key_states: Union[torch.Tensor, List[torch.Tensor]],
        value_states: Union[torch.Tensor, List[torch.Tensor]],
        attention_masks: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[List[torch.LongTensor]] = None,
        past_key_values: Optional[List[Cache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        all_ones_mask: Optional[bool] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[Cache]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        if (attention_masks is not None) and (not isinstance(attention_masks, list)):
            raise ValueError("attention mask needs to be a list")
        if (position_ids is not None) and (not isinstance(position_ids, list)):
            raise ValueError("position_ids needs to be a list")
        if (past_key_values is not None) and (not isinstance(past_key_values, list)):
            raise ValueError("must pass list of caches")

        if output_attentions:
            raise ValueError("outputting attention weights is not supported")

        sharded_inputs = isinstance(query_states, list)
        if not sharded_inputs:
            query_states = self.__shard_inputs(query_states)
            key_states = self.__shard_inputs(key_states)
            value_states = self.__shard_inputs(value_states)
        else:
            # already sharded
            query_states = query_states
            key_states = key_states
            value_states = value_states

        num_head_groups = len(query_states)

        if all_ones_mask is None:
            # if not specified, assume it might not have just ones
            all_ones_mask = False

        bsz, q_len, _ = query_states[0].size()

        if q_len == 1:
            # the transposition doesn nothing, so we can just avoid it to avoid the CPU cost of the operation
            query_states = [query_state.view(bsz, self.num_tp_heads, q_len, self.head_dim) for query_state in query_states]
            key_states = [key_state.view(bsz, self.num_tp_key_value_heads, q_len, self.head_dim) for key_state in key_states]
            value_states = [value_state.view(bsz, self.num_tp_key_value_heads, q_len, self.head_dim) for value_state in value_states]
        else:
            query_states = [query_state.view(bsz, q_len, self.num_tp_heads, self.head_dim).transpose(1, 2) for query_state in query_states]
            key_states = [key_state.view(bsz, q_len, self.num_tp_key_value_heads, self.head_dim).transpose(1, 2) for key_state in key_states]
            value_states = [value_state.view(bsz, q_len, self.num_tp_key_value_heads, self.head_dim).transpose(1, 2) for value_state in value_states]
    
        kv_seq_len = key_states[0].shape[-2]
        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_values[0].get_usable_length(kv_seq_len, self.layer_idx)


        for d in range(num_head_groups):
            # do the rotary embeddings on each head group
            pos_ids = position_ids[d] if position_ids is not None else None
            past_key_value = past_key_values[d] if past_key_values is not None else None
            query_states[d], key_states[d], value_states[d] = self.rotary_embs[d].apply_rotary_pos_emb_write_kv_cache(query_states[d], key_states[d], pos_ids, kv_seq_len, value_states[d], past_key_value)

        # at this point, we have the following shapes:
        #  q: [B, num_q_heads, T, embed_dim]
        #  k: [B, num_k_heads, NEW_T, embed_dim]
        #  v: [B, num_v_heads, NEW_T, embed_dim]

        if (bsz == 1) and (q_len == 1) and all_ones_mask and (query_states[0].dtype == torch.float16):
            #
            attn_outputs = mui_parallel_causally_decode(query_states, key_states, value_states)

            # from shape [B, num_q_heads, T, embed_dim], go to [B, T, num_q_heads, embed_dim]
            # then from shape [B, T, num_q_heads, embed_dim] go to [B, T, hidden_size]

            # q_len is 1 so we can remove the transposition
            attn_outputs = [attn_output.reshape(bsz, q_len, self.tp_hidden_size) for attn_output in attn_outputs]
        else:
            attn_outputs = []
            for d in range(num_head_groups):
                # repeat k/v heads if n_kv_heads < n_heads
                key_state = repeat_kv(key_states[d], self.num_key_value_groups)
                value_state = repeat_kv(value_states[d], self.num_key_value_groups)
                query_state = query_states[d]

                attn_weights = torch.matmul(query_state, key_state.transpose(2, 3)) / math.sqrt(self.head_dim)

                if attn_weights.size() != (bsz, self.num_tp_heads, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, self.num_tp_heads, q_len, kv_seq_len)}, but is"
                        f" {attn_weights.size()}"
                    )

                if attention_masks is not None:
                    attention_mask = attention_masks[d]
                    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                        )

                    attn_weights = attn_weights + attention_mask

                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_state.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, value_state)

                if attn_output.size() != (bsz, self.num_tp_heads, q_len, self.head_dim):
                    raise ValueError(
                        f"`attn_output` should be of size {(bsz, self.num_tp_heads, q_len, self.head_dim)}, but is"
                        f" {attn_output.size()}"
                    )
                
                # from shape [B, num_q_heads, T, embed_dim], go to [B, T, num_q_heads, embed_dim]
                # then from shape [B, T, num_q_heads, embed_dim] go to [B, T, hidden_size]
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.tp_hidden_size)

                attn_outputs.append(attn_output)

        attn_outputs = self.o_proj.parallel_forward(attn_outputs, residual=residual)

        return attn_outputs, None, past_key_values

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[List[Cache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        all_ones_mask: Optional[bool] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Cache]]]:
        if self.tensor_parallelism > 1:
            attn_outputs, attn_weights, past_key_values = self.parallel_forward(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=MuiParallelLinear._broadcast(self.engine_config, attention_mask),
                position_ids=MuiParallelLinear._broadcast(self.engine_config, position_ids),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                all_ones_mask=all_ones_mask,
                residual=residual,
            )

            # Get output from GPU0
            return attn_outputs[0], attn_weights, past_key_values

        raise ValueError("Only parallel inference is supported")