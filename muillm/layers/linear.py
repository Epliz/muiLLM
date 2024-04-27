import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import muillm_ext

class _MuiLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias):
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

        new_module.weight = nn.Parameter(prev_module.weight.detach())
        new_module.weight.requires_grad = prev_module.weight.requires_grad

        if has_bias:
            new_module.bias = nn.Parameter(prev_module.bias.detach())
            new_module.bias.requires_grad = prev_module.bias.requires_grad

        return new_module

    def forward(self, input: Tensor) -> Tensor:
        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type
            return _MuiLinear.apply(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)