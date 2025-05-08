from typing import Optional, Tuple, Union
import warnings
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.module import MuiModule
import torch
import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.attention.baseattention import MuiBaseAttention
from muillm.modules.attention.sdpaattention import MuiSdpaAttention
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
from muillm.modules.multilinear import MuiMultiLinear

from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralAttention,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention


class MuiDecoderLayer(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        qkv_proj: MuiMultiLinear,
        self_attn: MuiBaseAttention,
        mlp: MuiGateUpDownMLP,
    ):
        super().__init__(engine_config=engine_config)

        self.qkv_proj = qkv_proj
        self.self_attn = self_attn

        self.mlp = mlp

    @staticmethod
    def replace(
        prev_module: Union["MuiDecoderLayer", LlamaDecoderLayer, MistralDecoderLayer],
        engine_config: MuiEngineConfig,
        device=None,
    ) -> "MuiDecoderLayer":
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiDecoderLayer):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_module

        prev_attn = prev_module.self_attn

        input_layernorm = prev_module.input_layernorm
        qkv_proj = None
        self_attn = None
        if isinstance(prev_attn, MistralAttention) or isinstance(
            prev_attn, LlamaAttention
        ):
            # When using tensor parallelism, we shard the attention by head, so we need to shard qkv by rows
            qkv_proj = MuiMultiLinear.replace(
                prev_modules=[prev_attn.q_proj, prev_attn.k_proj, prev_attn.v_proj],
                engine_config=engine_config,
                prev_layernorm_module=input_layernorm,
                device=device,
            )
            self_attn = MuiSdpaAttention.replace(
                prev_module.self_attn, engine_config=engine_config, device=device
            )
        else:
            raise ValueError(f"Not supported {type(prev_module.self_attn)}")

        post_attention_layernorm = prev_module.post_attention_layernorm
        # even if we had a MuiGateUpDownMLP, we need to replace it with a new one to get the layernorm module
        mlp = MuiGateUpDownMLP.replace(
            prev_module=prev_module.mlp,
            engine_config=engine_config,
            prev_layernorm_module=post_attention_layernorm,
            device=device,
        )

        new_decoder = MuiDecoderLayer(
            engine_config=engine_config, qkv_proj=qkv_proj, self_attn=self_attn, mlp=mlp
        )

        # delete the previous module to save memory
        del prev_module

        # trigger GC to save memory
        trigger_gc()

        return new_decoder

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.45
        all_ones_mask: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
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
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
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
