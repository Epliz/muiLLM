import math
from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from muillm.engineconfig import MuiEngineConfig
from transformers.models.mistral.modeling_mistral import MistralMLP
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from muillm.layers.linear import MuiLinear

from muillm.layers.rmsnorm import _MuiRMSNorm
import muillm_ext

class _MuiGateUpSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, norm_weights, variance_epsilon, gate_weights, up_weights):
        output = muillm_ext.muillm_gateupsilu_forward(norm_weights, variance_epsilon, gate_weights, up_weights, inputs)

        ctx.save_for_backward(inputs, norm_weights, variance_epsilon, gate_weights, up_weights)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GateUpSiLU backward is not implemented")

class MuiGateUpDownMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation_function: nn.Module, variance_epsilon:float = 0.0, normalize:bool = False, device=None, dtype=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device)) if normalize else None

        self.gate_proj = MuiLinear(self.hidden_size, self.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = MuiLinear(self.hidden_size, self.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = MuiLinear(self.intermediate_size, self.hidden_size, bias=False, device=device, dtype=dtype)
        self.activation_function = activation_function

        wdtype = self.gate_proj.weight.dtype
        dispatchable_activation = (isinstance(self.activation_function, nn.SiLU))
        dispatchable_type = (wdtype == torch.float16)
        dispatchable_device = self.gate_proj.weight.is_cuda
        self.dispatchable = dispatchable_activation and dispatchable_device and dispatchable_type

    @staticmethod
    def replace(prev_module: MistralMLP, engine_config: MuiEngineConfig, prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None) -> "MuiGateUpDownMLP":
        dtype=prev_module.gate_proj.weight.dtype
        device=prev_module.gate_proj.weight.device

        hidden_size = prev_module.hidden_size
        intermediate_size = prev_module.intermediate_size
        activation_function = prev_module.act_fn

        normalize = prev_layernorm_module is not None
        variance_epsilon = prev_layernorm_module.variance_epsilon if normalize else 0.0
        norm_weights = prev_layernorm_module.weight if normalize else None

        new_module = MuiGateUpDownMLP(hidden_size=hidden_size, intermediate_size=intermediate_size, activation_function=activation_function, variance_epsilon=variance_epsilon, normalize=normalize, dtype=dtype, device=device)
        new_module.copy_module(prev_module=prev_module, norm_weights=norm_weights)

        return new_module

    def copy_module(self, prev_module: MistralMLP, norm_weights: torch.Tensor = None, variance_epsilon: float = 0.0):
        self.gate_proj.copy_module(prev_module.gate_proj)
        self.up_proj.copy_module(prev_module.up_proj)

        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities

            norm_weights_requires_grad = norm_weights.requires_grad
            self.norm_weights = nn.Parameter(norm_weights.detach())
            self.norm_weights.requires_grad = norm_weights_requires_grad

            self.norm_weights = norm_weights

        self.down_proj.copy_module(prev_module.down_proj)

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type

            # Also check that we don't have quantized linear
            if isinstance(self.gate_proj, MuiLinear) and isinstance(self.up_proj, MuiLinear):
                gateup = _MuiGateUpSiLU.apply(input, self.norm_weights, self.variance_epsilon, self.gate_proj.weight, self.up_proj.weight)
                return self.down_proj(gateup, residual=residual)

        # else: # not dispatchable or not MuiLinear
        if self.normalize:
            input = _MuiRMSNorm.apply(input, self.norm_weights, self.variance_epsilon)

        g = self.gate_proj(input)
        u = self.up_proj(input)
        output = self.down_proj(self.activation_function(g) * u)

        if residual is not None:
            output = output + residual

        return output