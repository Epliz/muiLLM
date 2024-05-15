from typing import Optional, Tuple
import warnings
import torch
import torch.nn as nn

from muillm.layers.attention.mistral.baseattention import MuiMistralAttention
from muillm.layers.attention.mistral.sdpaattention import MuiMistralSdpaAttention
from muillm.layers.gateupdownmlp import MuiGateUpDownMLP
from muillm.layers.rmsnorm import MuiRMSNorm

from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralSdpaAttention, MISTRAL_ATTENTION_CLASSES

class MuiDecoderLayer(nn.Module):
    def __init__(self, self_attn: MuiMistralAttention, mlp: MuiGateUpDownMLP, input_layernorm: MuiRMSNorm, post_attention_layernorm: MuiRMSNorm):
        super().__init__()
        self.self_attn = self_attn

        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    @staticmethod
    def replace(prev_module: MistralDecoderLayer) -> "MuiDecoderLayer":

        self_attn = None
        if isinstance(prev_module.self_attn, MuiMistralAttention):
            # already replaced
            self_attn = prev_module.self_attn
        elif isinstance(prev_module.self_attn, MistralSdpaAttention):
            self_attn = MuiMistralSdpaAttention.replace(prev_module.self_attn)
        else:
            raise ValueError(f"Not supported {type(prev_module.self_attn)}")

        mlp = prev_module.mlp
        if not isinstance(prev_module.mlp, MuiGateUpDownMLP):
            mlp = MuiGateUpDownMLP.replace(prev_module.mlp)

        input_layernorm = prev_module.input_layernorm
        if not isinstance(prev_module.input_layernorm, MuiRMSNorm):
            input_layernorm = MuiRMSNorm.replace(prev_module.input_layernorm)

        post_attention_layernorm = prev_module.post_attention_layernorm
        if not isinstance(prev_module.post_attention_layernorm, MuiRMSNorm):
            post_attention_layernorm = MuiRMSNorm.replace(prev_module.post_attention_layernorm)

        return MuiDecoderLayer(self_attn=self_attn, mlp=mlp, input_layernorm=input_layernorm, post_attention_layernorm=post_attention_layernorm)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
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

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            residual=residual,
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual=residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs