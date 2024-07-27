from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from muillm.layers.linear import MuiLinear
from muillm.layers.rmsnorm import _MuiRMSNorm
from muillm.quantization.quantizationmethod import Int8WeightOnlyQuantizationMethod
from muillm.quantization.rtnquantizer import RTNQuantizer
import muillm_ext

class _MuiInt8Dequantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weights,
        scales_min_vals,
        group_size_shift
    ):
        dequant_weights = muillm_ext.muillm_int8_dequantize_forward(
            weights,
            scales_min_vals,
            group_size_shift,
        )

        ctx.save_for_backward(weights)

        return dequant_weights

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Int8Dequantize backward is not implemented")


class _MuiInt8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, scales_min_vals, group_size_shift, norm_weights, variance_epsilon, add_bias, residual):
        if (add_bias is not None) and (residual is not None):
            raise ValueError("bias and residual at the same time is not supported")

        if residual is not None:
            add_bias = residual

        output = muillm_ext.muillm_int8_linear_forward(x, weights, scales_min_vals, group_size_shift, norm_weights, variance_epsilon, mul_bias=None, add_bias=add_bias)

        ctx.save_for_backward(x, weights, norm_weights, variance_epsilon, add_bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")

class MuiInt8Linear(nn.Module):
    def __init__(self, quantization_method: Int8WeightOnlyQuantizationMethod, in_features: int, out_features: int, bias: bool = True,
                 variance_epsilon:float = 0.0, normalize:bool = False, device=None, dtype=None,
                 prev_weights_uint8: torch.Tensor = None, prev_scales_min_vals: torch.Tensor = None, prev_bias: torch.Tensor = None) -> None:
        super().__init__()
        self.quantizer = RTNQuantizer(n_bit=8, groupsize=quantization_method.group_size, f=quantization_method.f)

        self.in_features = in_features
        self.out_features = out_features
        self.weight_dtype = dtype

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = nn.Parameter(torch.ones(in_features, dtype=dtype, device=device)) if normalize else None

        num_groups = int(in_features / quantization_method.group_size)
        # cannot set requires grad on uint8 tensors
        # TODO: lift limitation on contiguous?
        self.weights_uint8 = nn.Parameter(prev_weights_uint8.contiguous() if prev_weights_uint8 is not None else torch.zeros(size=(out_features, in_features), dtype=torch.uint8, device=device), requires_grad=False)
        self.scales_min_vals = nn.Parameter(prev_scales_min_vals.contiguous() if prev_scales_min_vals is not None else torch.zeros(size=(out_features * num_groups, 2), dtype=dtype, device=device))
        self.bias = nn.Parameter(prev_bias.contiguous() if prev_bias is not None else torch.zeros(size=(out_features,), dtype=dtype, device=device)) if bias else None

        dispatchable_type = (self.weight_dtype == torch.float16)
        dispatchable_device = self.weights_uint8.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    @staticmethod
    def replace(prev_module: Union[nn.Linear, MuiLinear], quantization_method: Int8WeightOnlyQuantizationMethod) -> "MuiInt8Linear":
        has_bias = prev_module.bias is not None
        in_features = prev_module.in_features
        out_features = prev_module.out_features

        normalize = False
        variance_epsilon = 0
        norm_weights = None

        if isinstance(prev_module, MuiLinear):
            normalize = prev_module.normalize
            variance_epsilon = prev_module.variance_epsilon if normalize else 0.0
            norm_weights = prev_module.norm_weights if normalize else None

        new_module = MuiInt8Linear(quantization_method=quantization_method, in_features=in_features, out_features=out_features, bias=has_bias, variance_epsilon=variance_epsilon, normalize=normalize, dtype=prev_module.weight.dtype, device=prev_module.weight.device)
        new_module.copy_module(prev_module=prev_module, norm_weights=norm_weights)

        return new_module

    def copy_module(self, prev_module: nn.Linear, norm_weights: torch.Tensor = None):
        has_bias = prev_module.bias is not None

        weights_require_grads = prev_module.weight.requires_grad

        # quantize
        weights_uint8, scales_min_vals = self.quantizer.group_quantize_tensor(prev_module.weight.detach())

        # Cannot set requires_grad on int8 tensors
        self.weights_uint8 = nn.Parameter(weights_uint8.detach(), requires_grad=False)

        self.scales_min_vals = nn.Parameter(scales_min_vals.detach())
        self.scales_min_vals.requires_grad = weights_require_grads

        if has_bias:
            self.bias = nn.Parameter(prev_module.bias.detach())
            self.bias.requires_grad = prev_module.bias.requires_grad

        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities

            norm_weights_requires_grad = norm_weights.requires_grad
            self.norm_weights = nn.Parameter(norm_weights.detach())
            self.norm_weights.requires_grad = norm_weights_requires_grad

            self.norm_weights = norm_weights

    def _dequantize_weights(self) -> torch.Tensor:
        if self.dispatchable:
            return _MuiInt8Dequantize.apply(
                self.weights_uint8,
                self.scales_min_vals,
                self.quantizer.group_size_shift
            )
        
        # not dispatchable

        return self.quantizer.group_dequantize_tensor(self.weights_uint8, self.scales_min_vals, dtype=self.weight_dtype)

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type
            return _MuiInt8Linear.apply(input, self.weights_uint8, self.scales_min_vals, self.quantizer.group_size_shift, self.norm_weights, self.variance_epsilon, self.bias, residual)
        else:
            if self.normalize:
                input = _MuiRMSNorm.apply(input, self.norm_weights, self.variance_epsilon)

            # dequantize
            dequantized_weights = self._dequantize_weights()

            # compute
            output = F.linear(input, dequantized_weights, self.bias)

            if residual is not None:
                output = output + residual
            return output