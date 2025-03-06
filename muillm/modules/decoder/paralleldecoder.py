from typing import List, Optional, Tuple, Union
import warnings
from muillm.modules.attention.parallelbaseattention import MuiParallelBaseAttention
from muillm.modules.attention.parallelsdpaattention import MuiParallelSdpaAttention
from muillm.modules.kvcache.cache_utils import MuiCache
from muillm.modules.module import MuiModule
from muillm.modules.parallelgateupdownmlp import MuiParallelGateUpDownMLP
from muillm.modules.parallelmultilinear import MuiParallelMultiLinear
import torch
import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.attention.baseattention import MuiBaseAttention
from muillm.modules.attention.sdpaattention import MuiSdpaAttention
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
from muillm.modules.rmsnorm import MuiRMSNorm
from muillm.modules.multilinear import MuiMultiLinear

import muillm_ext

from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaMLP
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralMLP

class _MuiParallelDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, cache_module, h, m, position_ids, position_embeddings, cache_positions):
        output = muillm_ext.muillm_parallel_decoder_module_forward(
            module,
            cache_module,
            h,
            m,
            position_ids,
            position_embeddings,
            cache_positions
        )

        ctx.save_for_backward(h, m)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")

class MuiParallelDecoderLayer(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, qkv_proj: MuiParallelMultiLinear, self_attn: MuiParallelBaseAttention, mlp: MuiParallelGateUpDownMLP):
        super().__init__(engine_config=engine_config)

        self.cpp_engine = engine_config.cpp_engine
        self.comms = engine_config.comms
        self.tensor_parallelism = engine_config.tensor_parallelism

        self.qkv_proj = qkv_proj
        self.self_attn = self_attn
        self.mlp = mlp

        # the cpp module will be created at the end of all layer replacements
        self.cpp_module = None

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        self.dispatchable = self.self_attn.dispatchable and self.mlp.dispatchable

    def finalize_init(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_decoder_module_deinit(self.cpp_module)

        self.cpp_module = muillm_ext.muillm_parallel_decoder_module_init(
            self.cpp_engine,
            self.comms.comms,
            self.qkv_proj.cpp_module,
            self.self_attn.cpp_module,
            self.mlp.cpp_module
        )

    @staticmethod
    def replace(prev_module: Union[LlamaDecoderLayer, MistralDecoderLayer], engine_config: MuiEngineConfig) -> "MuiParallelGateUpDownMLP":
        prev_attn = prev_module.self_attn

        qkv_proj = None
        self_attn = None

        if isinstance(prev_module, MuiParallelDecoderLayer):
            if isinstance(prev_module.qkv_proj, MuiParallelMultiLinear):
                qkv_proj = prev_module.qkv_proj
            elif isinstance(prev_module.qkv_proj, MuiMultiLinear):
            # When using tensor parallelism, we shard the attention by head, so we need to shard qkv by rows
                qkv_proj = MuiParallelMultiLinear.replace(prev_modules=qkv_proj, engine_config=engine_config, sharding_dim=0)
            else:
                raise ValueError(f"Not supported {type(prev_module.qkv_proj)}")

            self_attn = prev_module.self_attn
            if not isinstance(self_attn, MuiParallelBaseAttention):
                # replace the attention module itself if necessary
                self_attn = MuiParallelSdpaAttention.replace(self_attn, engine_config=engine_config)

        elif isinstance(prev_module, MistralDecoderLayer) or isinstance(prev_module, LlamaDecoderLayer):
            input_layernorm = prev_module.input_layernorm
            # When using tensor parallelism, we shard the attention by head, so we need to shard qkv by rows
            qkv_proj = MuiParallelMultiLinear.replace(prev_modules=[prev_attn.q_proj, prev_attn.k_proj, prev_attn.v_proj], engine_config=engine_config, prev_layernorm_module=input_layernorm, sharding_dim=0)
            self_attn = MuiParallelSdpaAttention.replace(prev_attn, engine_config=engine_config)
        else:
            raise ValueError(f"Not supported {type(prev_module.self_attn)}")


        mlp = None
        if isinstance(prev_module.mlp, MuiGateUpDownMLP):
            mlp = MuiParallelGateUpDownMLP.replace(prev_module=prev_module.mlp, engine_config=engine_config)
        elif isinstance(prev_module.mlp, MistralMLP) or isinstance(prev_module.mlp, LlamaMLP):
            post_attention_layernorm = prev_module.post_attention_layernorm
            mlp = MuiParallelGateUpDownMLP.replace(prev_module=prev_module.mlp, engine_config=engine_config, prev_layernorm_module=post_attention_layernorm)
        else:
            raise ValueError(f"Not supported {type(prev_module.mlp)}")

        return MuiParallelDecoderLayer(engine_config=engine_config, qkv_proj=qkv_proj, self_attn=self_attn, mlp=mlp)

    
    def parallel_forward(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        all_ones_mask: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[List[torch.FloatTensor], Optional[Tuple[List[torch.FloatTensor], Optional[List[torch.FloatTensor]]]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        if output_attentions:
            raise ValueError("output_attention is not supported")

        # unwrap inputs if needed
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[0]

        if all_ones_mask is None:
            # if not specified, assume it might not have just ones
            all_ones_mask = False
        if all_ones_mask:
            attention_mask = None

        bsz, q_len, _ = hidden_states.size()
        if self.dispatchable and (bsz == 1) and (q_len == 1) and isinstance(past_key_value, MuiCache):
            hidden_states = _MuiParallelDecoder.apply(
                self.cpp_module,
                past_key_value.cpp_module,
                hidden_states,
                attention_mask,
                position_ids,
                position_embeddings,
                cache_position,
            )
        else:
            residual = hidden_states

            # mark the inputs as already sharding by wrapping them in list
            query_states, key_states, value_states = self.qkv_proj.parallel_forward([hidden_states], collect_outputs=False)[0]

            # Self Attention
            hidden_states = self.self_attn.parallel_forward(
                query_states=[query_states],
                key_states=[key_states],
                value_states=[value_states],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                all_ones_mask=all_ones_mask,
                residual=residual,
            )

            hidden_states = hidden_states[0]
            # Fully Connected
            residual = hidden_states
            # post attention layer norm is fused in the MLP
            # wrap in list to mark as already sharded
            hidden_states = self.mlp.parallel_forward([hidden_states], residual=residual)[0]

        outputs = (hidden_states,)

        if use_cache:
            outputs += ([past_key_value],)

        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        all_ones_mask: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.tensor_parallelism > 1:
            layer_outputs = self.parallel_forward(
                hidden_states=[hidden_states],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                all_ones_mask=all_ones_mask,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            final_outputs = (hidden_states,)
            
            if output_attentions:
                attn_weights = (layer_outputs[1][0],)
                final_outputs += (attn_weights,)

            if use_cache:
                present_key_value = layer_outputs[2 if output_attentions else 1][0]
                final_outputs += (present_key_value,)

            return final_outputs

        raise ValueError("Only parallel inference is supported")