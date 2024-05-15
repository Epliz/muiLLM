from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import muillm_ext

class _MuiLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, residual, weights, bias):
        if (bias is not None) and (residual is not None):
            raise ValueError("bias and residual at the same time is not supported")

        if residual is not None:
            bias = residual

        if bias is not None:
            output = muillm_ext.muillm_linear_forward(weights, bias, inputs)
        else:
            output = muillm_ext.muillm_linear_forward_no_bias(weights, inputs)

        ctx.save_for_backward(inputs, weights, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights, bias = ctx.saved_tensors

        g_x = torch.matmul(grad_output, weights)
        g_w = torch.matmul(inputs, grad_output)
        g_b = None
        if bias is not None:
            g_b = grad_output.sum(axis=-1)
        return g_x, g_w, g_b

class MuiLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)

        wdtype = self.weight.dtype
        dispatchable_type = (wdtype == torch.float16)
        dispatchable_device = self.weight.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    @staticmethod
    def replace(prev_module: nn.Linear) -> "MuiLinear":
        has_bias = prev_module.bias is not None
        in_features = prev_module.in_features
        out_features = prev_module.out_features

        new_module = MuiLinear(in_features=in_features, out_features=out_features, bias=has_bias, dtype=prev_module.weight.dtype, device=prev_module.weight.device)
        new_module.copy_module(prev_module=prev_module)

        return new_module

    def copy_module(self, prev_module: nn.Linear):
        has_bias = prev_module.bias is not None

        self.weight = nn.Parameter(prev_module.weight.detach())
        self.weight.requires_grad = prev_module.weight.requires_grad

        if has_bias:
            self.bias = nn.Parameter(prev_module.bias.detach())
            self.bias.requires_grad = prev_module.bias.requires_grad

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type
            return _MuiLinear.apply(input, residual, self.weight, self.bias)
        else:
            output = F.linear(input, self.weight, self.bias)
            if residual is not None:
                output = output + residual
            return output