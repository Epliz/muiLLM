from typing import Optional, Tuple
import warnings
from muillm.engineconfig import MuiEngineConfig
from muillm.muimodule import MuiModule
import torch
import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig
from muillm.layers.attention.mistral.baseattention import MuiMistralAttention
from muillm.layers.attention.mistral.sdpaattention import MuiMistralSdpaAttention
from muillm.layers.gateupdownmlp import MuiGateUpDownMLP
from muillm.layers.rmsnorm import MuiRMSNorm
from muillm.layers.multilinear import MuiMultiLinear

from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralAttention, MistralSdpaAttention, MISTRAL_ATTENTION_CLASSES


class MuiDecoderLayer(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, qkv_proj: MuiMultiLinear, self_attn: MuiMistralAttention, mlp: MuiGateUpDownMLP):
        super().__init__(engine_config=engine_config)

        self.qkv_proj = qkv_proj
        self.self_attn = self_attn

        self.mlp = mlp

    @staticmethod
    def replace(prev_module: MistralDecoderLayer, engine_config: MuiEngineConfig) -> "MuiDecoderLayer":
        prev_attn = prev_module.self_attn

        input_layernorm = prev_module.input_layernorm
        qkv_proj = None
        self_attn = None
        if isinstance(prev_attn, MistralSdpaAttention):
            # When using tensor parallelism, we shard the attention by head, so we need to shard qkv by rows
            qkv_proj = MuiMultiLinear.replace(prev_modules=[prev_attn.q_proj, prev_attn.k_proj, prev_attn.v_proj], engine_config=engine_config, prev_layernorm_module=input_layernorm, sharding_dim=-2)
            self_attn = MuiMistralSdpaAttention.replace(prev_module.self_attn, engine_config=engine_config)
        else:
            raise ValueError(f"Not supported {type(prev_module.self_attn)}")

        post_attention_layernorm = prev_module.post_attention_layernorm
        mlp = prev_module.mlp
        if not isinstance(prev_module.mlp, MuiGateUpDownMLP):
            mlp = MuiGateUpDownMLP.replace(prev_module=prev_module.mlp, engine_config=engine_config, prev_layernorm_module=post_attention_layernorm)

        return MuiDecoderLayer(qkv_proj=qkv_proj, self_attn=self_attn, mlp=mlp, engine_config=engine_config)

    
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

        # when using tensor parallelism, we shard by head so that we don't need to sync after the qkv_proj

        # Transform q, k, v
        # input layer norm is fused
        query_states, key_states, value_states = self.qkv_proj(hidden_states, collect_output=False)

        # Self Attention
        # (when using tensor parallelism, query, key, states are split by head already,
        # so we don't need to shard the inputs)
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
            shard_inputs=False # for tensor parallelism
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