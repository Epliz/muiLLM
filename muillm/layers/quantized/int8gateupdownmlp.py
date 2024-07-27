import math
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from muillm.layers.quantized.int8linear import MuiInt8Linear
from muillm.layers.gateupdownmlp import MuiGateUpDownMLP
from muillm.layers.rmsnorm import _MuiRMSNorm
from muillm.quantization.rtnquantizer import RTNQuantizer
from muillm.quantization.quantizationmethod import Int8WeightOnlyQuantizationMethod

import muillm_ext

class _MuiInt8GateDequantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gate_up_weights,
        gate_up_scales_min_vals,
        group_size_shift
    ):
        gate_weights, up_weights = muillm_ext.muillm_int8_gateupsilu_dequantize_forward(
            gate_up_weights,
            gate_up_scales_min_vals,
            group_size_shift,
        )

        ctx.save_for_backward(gate_up_weights)

        return gate_weights, up_weights

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Int8GateUpDequantize backward is not implemented")


class _MuiInt8GateUpSiLU(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        norm_weights, variance_epsilon,
        gate_up_weights,
        gate_up_scales_min_vals,
        group_size_shift
    ):
        output = muillm_ext.muillm_int8_gateupsilu_forward(
            norm_weights, variance_epsilon,
            gate_up_weights,
            gate_up_scales_min_vals,
            group_size_shift,
            inputs
        )

        ctx.save_for_backward(inputs, norm_weights, variance_epsilon, gate_up_weights)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Int8GateUpSiLU backward is not implemented")

class MuiInt8GateUpDownMLP(nn.Module):
    def __init__(self, quantization_method: Int8WeightOnlyQuantizationMethod, hidden_size: int, intermediate_size: int, activation_function: nn.Module, variance_epsilon:float = 0.0, normalize:bool = False, device=None, dtype=None) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.quantization_method = quantization_method
        self.quantizer = RTNQuantizer(n_bit=8, groupsize=quantization_method.group_size, f=quantization_method.f)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device)) if normalize else None

        gate_proj = MuiInt8Linear(quantization_method, self.hidden_size, self.intermediate_size, bias=False, device=device, dtype=dtype)
        up_proj = MuiInt8Linear(quantization_method, self.hidden_size, self.intermediate_size, bias=False, device=device, dtype=dtype)

        self.group_size_shift = gate_proj.quantizer.group_size_shift

        # we pack the sets of weights from gate & up projs to have better memory bandwidth utilisation in the kernel
        self.gate_up_weights, self.gate_up_scales_min_vals = self._pack_gateup_linears(gate_proj, up_proj)

        self.down_proj = MuiInt8Linear(quantization_method, self.intermediate_size, self.hidden_size, bias=False, device=device, dtype=dtype)
        self.activation_function = activation_function

        wdtype = dtype
        dispatchable_activation = (isinstance(self.activation_function, nn.SiLU))
        dispatchable_type = (wdtype == torch.float16)
        dispatchable_device = self.gate_up_weights.is_cuda
        self.dispatchable = dispatchable_activation and dispatchable_device and dispatchable_type

    @staticmethod
    def replace(prev_module: MuiGateUpDownMLP, quantization_method: Int8WeightOnlyQuantizationMethod) -> "MuiInt8GateUpDownMLP":
        dtype=prev_module.gate_proj.weight.dtype
        device=prev_module.gate_proj.weight.device

        hidden_size = prev_module.hidden_size
        intermediate_size = prev_module.intermediate_size
        activation_function = prev_module.activation_function

        normalize = prev_module.normalize is not None
        variance_epsilon = prev_module.variance_epsilon

        new_module = MuiInt8GateUpDownMLP(quantization_method=quantization_method, hidden_size=hidden_size, intermediate_size=intermediate_size, activation_function=activation_function, variance_epsilon=variance_epsilon, normalize=normalize, dtype=dtype, device=device)
        new_module.copy_module(prev_module=prev_module)

        return new_module
    
    def _qint8_linear(self, new_qlinear: MuiInt8Linear, prev_linear: Union[nn.Linear, MuiInt8Linear]) -> MuiInt8Linear:
        if isinstance(prev_linear, MuiInt8Linear):
            return prev_linear
        else:
            new_qlinear.copy_module(prev_linear)
            return new_qlinear
        
    def _pack_gateup_weights(self, gate_weights: torch.Tensor, up_weights: torch.Tensor) -> torch.Tensor:
        packed =  torch.concat([gate_weights[:,:,None], up_weights[:,:,None]], dim=2)
        return packed

    def _pack_gateup_linears(self, gate_proj: MuiInt8Linear, up_proj: MuiInt8Linear) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_up_weights = self._pack_gateup_weights(gate_proj.weights_uint8, up_proj.weights_uint8)
        gate_up_scales_min_vals = torch.concat([gate_proj.scales_min_vals[:,None,:], up_proj.scales_min_vals[:,None,:]], dim=1)
        return gate_up_weights, gate_up_scales_min_vals

    def _unpack_gateup_weights(self, gate_up_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = gate_up_weights[:,:,0]
        up = gate_up_weights[:,:,1]
        return gate, up
    
    def _unpack_gateup_scales_min_vals(self, gate_up_scales_min_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_scales_min_vals = gate_up_scales_min_vals[:,0,:]
        up_scales_min_vals = gate_up_scales_min_vals[:,1,:]
        return gate_scales_min_vals, up_scales_min_vals

    def _dequantize_gateup_weights(self, gate_up_weights: torch.Tensor, gate_up_scales_min_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dispatchable:
            return _MuiInt8GateDequantize.apply(
                self.gate_up_weights,
                self.gate_up_scales_min_vals,
                self.group_size_shift
            )
        
        # not dispatchable
        gate_weights, up_weights = self._unpack_gateup_weights(gate_up_weights)
        gate_scales_min_vals, up_scales_min_vals = self._unpack_gateup_scales_min_vals(gate_up_scales_min_vals)

        dequant_gate_weights = self.quantizer.group_dequantize_tensor(gate_weights, gate_scales_min_vals, dtype=self.dtype)
        dequant_up_weights = self.quantizer.group_dequantize_tensor(up_weights, up_scales_min_vals, dtype=self.dtype)

        return dequant_gate_weights, dequant_up_weights

    def copy_module(self, prev_module: MuiGateUpDownMLP):

        if prev_module.norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities

            norm_weights_requires_grad = prev_module.norm_weights.requires_grad
            self.norm_weights = nn.Parameter(prev_module.norm_weights.detach())
            self.norm_weights.requires_grad = norm_weights_requires_grad

            self.norm_weights = prev_module.norm_weights

        gate_proj = MuiInt8Linear(self.quantization_method, self.hidden_size, self.intermediate_size, bias=False, device=self.device, dtype=self.dtype)
        up_proj = MuiInt8Linear(self.quantization_method, self.hidden_size, self.intermediate_size, bias=False, device=self.device, dtype=self.dtype)
        gate_proj = self._qint8_linear(gate_proj, prev_module.gate_proj)
        up_proj = self._qint8_linear(up_proj, prev_module.up_proj)

        self.gate_up_weights, self.gate_up_scales_min_vals = self._pack_gateup_linears(gate_proj, up_proj)

        self.down_proj = self._qint8_linear(self.down_proj, prev_module.down_proj)

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type
            gateup = _MuiInt8GateUpSiLU.apply(
                input,
                self.norm_weights, self.variance_epsilon,
                self.gate_up_weights,
                self.gate_up_scales_min_vals,
                self.group_size_shift,
            )
            return self.down_proj(gateup, residual=residual)

        #else: # not dispatchable or not MuiInt8Linear
        if self.normalize:
            input = _MuiRMSNorm.apply(input, self.norm_weights, self.variance_epsilon)

        gate_weights, up_weights = self._dequantize_gateup_weights(self.gate_up_weights, self.gate_up_scales_min_vals)

        g = F.linear(input, gate_weights)
        u = F.linear(input, up_weights)
        output = self.down_proj(self.activation_function(g) * u)

        if residual is not None:
            output = output + residual

        return output