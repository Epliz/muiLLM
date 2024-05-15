from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.mistral.modeling_mistral import MistralMLP

from muillm.layers.linear import MuiLinear

import muillm_ext

class _MuiGateUpSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, gate_weights, up_weights):
        output = muillm_ext.muillm_gateupsilu_forward(gate_weights, up_weights, inputs)

        ctx.save_for_backward(inputs, gate_weights, up_weights)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GateUpSiLU backward is not implemented")

class MuiGateUpDownMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation_function: nn.Module, device=None, dtype=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

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
    def replace(prev_module: MistralMLP) -> "MuiGateUpDownMLP":
        hidden_size = prev_module.hidden_size
        intermediate_size = prev_module.intermediate_size
        activation_function = prev_module.act_fn

        new_module = MuiGateUpDownMLP(hidden_size=hidden_size, intermediate_size=intermediate_size, activation_function=activation_function, dtype=prev_module.gate_proj.weight.dtype, device=prev_module.gate_proj.weight.device)
        new_module.copy_module(prev_module=prev_module)

        return new_module

    def copy_module(self, prev_module: MistralMLP):
        self.gate_proj.copy_module(prev_module.gate_proj)
        self.up_proj.copy_module(prev_module.up_proj)
        self.down_proj.copy_module(prev_module.down_proj)

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type
            gateup = _MuiGateUpSiLU.apply(input, self.gate_proj.weight, self.up_proj.weight)
            return self.down_proj(gateup, residual=residual)
        else:
            output = self.down_proj(self.activation_function(self.gate_proj(input)) * self.up_proj(input))
            if residual is not None:
                output = output + residual
            return output