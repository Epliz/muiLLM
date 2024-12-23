from typing import Optional, Tuple, Union
import warnings
from muillm.layers.attention.mistral.parallelsdpaattention import MuiParallelMistralSdpaAttention
from muillm.layers.module import MuiModule
from muillm.layers.parallelgateupdownmlp import MuiParallelGateUpDownMLP
from muillm.layers.parallelmultilinear import MuiParallelMultiLinear
from muillm.layers.transformer.decoder import MuiDecoderLayer
import torch

from muillm.engineconfig import MuiEngineConfig
from muillm.layers.attention.mistral.baseattention import MuiMistralAttention
from muillm.layers.attention.mistral.sdpaattention import MuiMistralSdpaAttention
from muillm.layers.gateupdownmlp import MuiGateUpDownMLP
from muillm.layers.multilinear import MuiMultiLinear

from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralSdpaAttention, MistralMLP


class MuiParallelDecoderLayer(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, qkv_proj: MuiParallelMultiLinear, self_attn: MuiMistralAttention, mlp: MuiParallelGateUpDownMLP):
        super().__init__(engine_config=engine_config)

        self.qkv_proj = qkv_proj
        self.self_attn = self_attn

        self.mlp = mlp

    @staticmethod
    def replace(prev_module: Union[MistralDecoderLayer, MuiDecoderLayer], engine_config: MuiEngineConfig) -> "MuiParallelDecoderLayer":
        prev_attn = prev_module.self_attn

        qkv_proj = None
        self_attn = None

        if isinstance(prev_module, MuiDecoderLayer):
            if isinstance(prev_module.qkv_proj, MuiParallelMultiLinear):
                qkv_proj = prev_module.qkv_proj
            elif isinstance(prev_module.qkv_proj, MuiMultiLinear):
            # When using tensor parallelism, we shard the attention by head, so we need to shard qkv by rows
                qkv_proj = MuiParallelMultiLinear.replace(prev_modules=qkv_proj, engine_config=engine_config, sharding_dim=0)
            else:
                raise ValueError(f"Not supported {type(prev_module.qkv_proj)}")

            self_attn = prev_module.self_attn
            if not isinstance(self_attn, MuiParallelMistralSdpaAttention):
                # replace the attention module itself if necessary
                self_attn = MuiParallelMistralSdpaAttention.replace(self_attn, engine_config=engine_config)

        elif isinstance(prev_module, MistralDecoderLayer):
            input_layernorm = prev_module.input_layernorm
            # When using tensor parallelism, we shard the attention by head, so we need to shard qkv by rows
            qkv_proj = MuiParallelMultiLinear.replace(prev_modules=[prev_attn.q_proj, prev_attn.k_proj, prev_attn.v_proj], engine_config=engine_config, prev_layernorm_module=input_layernorm, sharding_dim=0)
            self_attn = MuiParallelMistralSdpaAttention.replace(prev_module.self_attn, engine_config=engine_config)
        else:
            raise ValueError(f"Not supported {type(prev_module.self_attn)}")

        mlp = None
        if isinstance(prev_module.mlp, MuiGateUpDownMLP):
            mlp = MuiParallelGateUpDownMLP.replace(prev_module=prev_module.mlp, engine_config=engine_config)
        elif isinstance(prev_module.mlp, MistralMLP):
            post_attention_layernorm = prev_module.post_attention_layernorm
            mlp = MuiParallelGateUpDownMLP.replace(prev_module=prev_module.mlp, engine_config=engine_config, prev_layernorm_module=post_attention_layernorm)
        else:
            raise ValueError(f"Not supported {type(prev_module.mlp)}")

        return MuiParallelDecoderLayer(engine_config=engine_config, qkv_proj=qkv_proj, self_attn=self_attn, mlp=mlp)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        all_ones_mask: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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

        residual = hidden_states

        # Transform q, k, v
        # input layer norm is fused
        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        # Self Attention
        # TODO: parallel attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            all_ones_mask=all_ones_mask,
            residual=residual,
        )

        # Fully Connected
        residual = hidden_states
        # post attention layer norm is fused in the MLP
        hidden_states = self.mlp(hidden_states, residual=residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs