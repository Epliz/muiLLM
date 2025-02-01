
import math
from typing import List, Optional, Tuple, Union
import warnings
from muillm.modules.attention.baseattention import MuiBaseAttention
from muillm.modules.attention.parallelcausaltransformerdecoding import mui_parallel_causally_decode, mui_parallel_causally_decode_masked
from muillm.modules.parallellinear import MuiParallelLinear
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
from muillm.modules.attention.rotaryembedding import MuiMistralRotaryEmbedding
from muillm.modules.attention.causaltransformerdecoding import mui_causally_decode
from muillm.modules.attention.kvcache import repeat_kv

logger = logging.get_logger(__name__)

class MuiParallelBaseAttention(MuiModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, engine_config: MuiEngineConfig, config: Union[LlamaConfig, MistralConfig], rotary_embs: List[MuiMistralRotaryEmbedding], layer_idx: Optional[int] = None, device=None, dtype=None):
        super().__init__(engine_config=engine_config)
        self.config = config
        self.tensor_parallelism = engine_config.tensor_parallelism

        if isinstance(config, LlamaConfig) and config.pretraining_tp > 1:
            raise ValueError("Not supported")

        self.layer_idx = layer_idx

        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout

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

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        has_attention_bias = False
        if isinstance(config, LlamaConfig):
            has_attention_bias = config.attention_bias

        # QKV will be sharded by rows so that it corresponds to having sharded heads
        self.o_proj = MuiParallelLinear(engine_config, self.hidden_size, self.hidden_size, bias=has_attention_bias, sharding_dim=1)

        # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_embs = rotary_embs

    staticmethod
    def _create_rotary_embeddings(engine_config: MuiEngineConfig, config: Union[LlamaConfig, MistralConfig], layer_idx:int, device=None, dtype=None) -> List[MuiMistralRotaryEmbedding]:

        rotary_embs = [MuiMistralRotaryEmbedding(
            engine_config,
            config,
            layer_idx=layer_idx,
            device=d,
            dtype=dtype,
        ) for d in engine_config.devices]

        return rotary_embs

    @staticmethod
    def replace(prev_module: Union[MistralAttention, LlamaAttention, MuiBaseAttention], engine_config: MuiEngineConfig) -> "MuiParallelBaseAttention":
        if isinstance(prev_module, MistralAttention) or isinstance(prev_module, LlamaAttention):
            device = prev_module.o_proj.weight.device
            dtype = prev_module.o_proj.weight.dtype
        else:
            raise ValueError(f"unsupported module type: {type(prev_module)}")

        layer_idx=prev_module.layer_idx
        config=prev_module.config

        rotary_embs = MuiParallelBaseAttention._create_rotary_embeddings(engine_config, config, layer_idx, device, dtype)

        new_module = MuiParallelBaseAttention(engine_config=engine_config, config=config, rotary_embs=rotary_embs, layer_idx=prev_module.layer_idx, device=device, dtype=dtype)
        new_module.o_proj.copy_module(prev_module=prev_module.o_proj)

        return new_module

    def __shard_inputs(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        # if we are sharding along the k-dim, we need to shard the input accordingly
        tensors = torch.tensor_split(tensor, self.tensor_parallelism, -1)
        return MuiParallelLinear._transfer_across(self.engine_config, tensors)

    def parallel_forward(
        self,
        query_statess: Union[torch.Tensor, List[torch.Tensor]],
        key_statess: Union[torch.Tensor, List[torch.Tensor]],
        value_statess: Union[torch.Tensor, List[torch.Tensor]],
        attention_masks: Optional[List[torch.Tensor]] = None,
        position_idss: Optional[List[torch.LongTensor]] = None,
        past_key_values: Optional[List[Cache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_positions: Optional[List[torch.LongTensor]] = None,
        position_embeddingss: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,  # will become mandatory in v4.45
        all_ones_mask: Optional[bool] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[Cache]]]:
        if (attention_masks is not None) and (not isinstance(attention_masks, list)):
            raise ValueError("attention mask needs to be a list")
        if (position_idss is not None) and (not isinstance(position_idss, list)):
            raise ValueError("position_ids needs to be a list")
        if (past_key_values is not None) and (not isinstance(past_key_values, list)):
            raise ValueError("must pass list of caches")

        if output_attentions:
            raise ValueError("Not supported")

        sharded_inputs = isinstance(query_statess, list)
        if not sharded_inputs:
            query_statess = self.__shard_inputs(query_statess)
            key_statess = self.__shard_inputs(key_statess)
            value_statess = self.__shard_inputs(value_statess)
        else:
            # already sharded
            pass

        # q has the shape [B, NEW_T, num_q_heads * embed_dim]
        bsz, q_len, _ = query_statess[0].size()

        num_head_groups = len(query_statess)

        attn_outputs = []
        for d in range(num_head_groups):
            if (q_len == 1):
                # the transposition doesn nothing, so we can just avoid it to avoid the CPU cost of the operation
                query_states = query_statess[d].view(bsz, self.num_tp_heads, q_len, self.head_dim)
                key_states = key_statess[d].view(bsz, self.num_tp_key_value_heads, q_len, self.head_dim)
                value_states = value_statess[d].view(bsz, self.num_tp_key_value_heads, q_len, self.head_dim)
            else:
                query_states = query_statess[d].view(bsz, q_len, self.num_tp_heads, self.head_dim).transpose(1, 2)
                key_states = key_statess[d].view(bsz, q_len, self.num_tp_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_statess[d].view(bsz, q_len, self.num_tp_key_value_heads, self.head_dim).transpose(1, 2)

            pos_embeds = None
            if position_embeddingss is not None:
                coses, sines = position_embeddingss
                cos, sin = coses[d], sines[d]
                pos_embeds = cos, sin

            # Now the shapes are:
            # q: [B, num_q_heads, NEW_T, embed_dim]
            # k: [B, num_kv_heads, NEW_T, embed_dim]
            # v: [B, num_kv_heads, NEW_T, embed_dim]

            past_key_value = past_key_values[d] if past_key_values is not None else None
            query_states, key_states, value_states = self.rotary_embs[d].apply_rotary_pos_emb_write_kv_cache(
                query_states,
                key_states,
                position_idss[d],
                pos_embeds,
                value_states,
                past_key_value,
                cache_positions[d]
            )

            # Now the shapes are:
            # q: [B, num_q_heads, NEW_T, embed_dim]
            # k: [B, num_kv_heads, T, embed_dim]
            # v: [B, num_kv_heads, T, embed_dim]

            query_statess[d] = query_states
            key_statess[d] = key_states
            value_statess[d] = value_states

        if (q_len == 1) and (query_statess[0].dtype == torch.float16):
            if all_ones_mask:
                attn_outputs = mui_parallel_causally_decode(query_statess, key_statess, value_statess)
            else:
                # The mask has shape:
                # M: [B, 1, NEW_T, T]
                # It contains 0 where OK, min_dtype where padded
                # min_dtype obtained with torch.finfo(dtype).min
                attn_outputs = mui_parallel_causally_decode_masked(query_statess, key_statess, value_statess, attention_masks)

            # from shape [B, num_q_heads, NEW_T, embed_dim], go to [B, NEW_T, num_q_heads, embed_dim]
            # then from shape [B, NEW_T, num_q_heads, embed_dim] go to [B, NEW_T, hidden_size]

            # q_len is 1 so we can remove the transposition
            attn_outputs = [attn_output.reshape(bsz, q_len, self.tp_hidden_size) for attn_output in attn_outputs]
        else:
            for d in range(num_head_groups):
                query_states = query_statess[d]
                key_states = key_statess[d]
                value_states = value_statess[d]

                # Before, key_states has the shape [B, num_k_heads, T, embed_dim]
                # after key_states has the shape [B, num_q_heads, T, embed_dim]
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                # W: [B, num_q_heads, NEW_T, T]
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

                if attention_masks is not None:  # no matter the length, we just slice it

                    # The mask has shape:
                    # M: [B, 1, NEW_T, T]
                    # It contains 0 where OK, min_dtype where padded
                    # min_dtype obtained with torch.finfo(dtype).min
                    causal_mask = attention_masks[d]
                    attn_weights = attn_weights + causal_mask

                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, value_states)

                if attn_output.size() != (bsz, self.num_tp_heads, q_len, self.head_dim):
                    raise ValueError(
                        f"`attn_output` should be of size {(bsz, self.num_tp_heads, q_len, self.head_dim)}, but is"
                        f" {attn_output.size()}"
                    )


                # Before, the shapes are:
                # A: [B, num_q_heads, NEW_T, embed_dim]
                attn_output = attn_output.transpose(1, 2).contiguous()

                attn_output = attn_output.reshape(bsz, q_len, -1)

                # Now the shapes are:
                # A: [B, NEW_T, num_q_heads * embed_dim]
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
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        all_ones_mask: Optional[bool] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        position_idss = MuiParallelLinear._broadcast(self.engine_config, position_ids)

        attention_masks = None
        if attention_mask is not None:
            attention_masks = MuiParallelLinear._broadcast(self.engine_config, attention_mask)

        cache_positions = None
        if cache_position is not None:
            cache_positions = MuiParallelLinear._broadcast(self.engine_config, cache_position)

        position_embeddingss = None
        if position_embeddings is not None:
            cos, sin = position_embeddings
            coses = MuiParallelLinear._broadcast(self.engine_config, cos)
            sines = MuiParallelLinear._broadcast(self.engine_config, sin)
            position_embeddingss = coses, sines

        attn_outputs, attn_weights, past_key_values = self.parallel_forward(
            query_states,
            key_states,
            value_states,
            attention_masks,
            position_idss,
            None, # past_key_values
            output_attentions,
            use_cache,
            cache_positions,
            position_embeddingss,
            all_ones_mask,
            residual
        )

        # Get results from GPU0
        return attn_outputs[0], None, None