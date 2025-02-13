from enum import Enum
import math
from typing import Optional, Union
from muillm.modules.module import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from muillm.engineconfig import MuiEngineConfig
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralMLP, MistralRMSNorm

from muillm.modules.linear import MuiLinear

from muillm.modules.rmsnorm import _MuiRMSNorm
import muillm_ext

class _MuiGateUpSiLUMethod(Enum):
    # Basic method where Gate/Up projections + mul are done distinctly
    GATEUPSILU_UNFUSED = 0
    # Method where the Gate/Up projections + mul are all fused
    GATEUPSILU_FUSED = 1
    # Method where the Gate/Up projections are done in the same kernel
    # but split between blocks to have more blocks.
    # A final reduction is done in an epilogue kernel
    GATEUPSILU_SPLIT = 2

class _MuiGateUpSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, engine, inputs, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual):
        output = muillm_ext.muillm_gateupsilu_forward(engine, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual, inputs)

        ctx.save_for_backward(inputs, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GateUpSiLU backward is not implemented")

class _MuiGateUpSiLUSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, engine, inputs, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual):
        output = muillm_ext.muillm_gateupsilu_split_forward(engine, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual, inputs)

        ctx.save_for_backward(inputs, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GateUpSiLU split K backward is not implemented")

class MuiGateUpDownMLP(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, hidden_size: int, intermediate_size: int, activation_function: nn.Module, variance_epsilon:float = 0.0, normalize:bool = False, device=None, dtype=None) -> None:
        super().__init__(engine_config=engine_config)
        self.cpp_engine = engine_config.cpp_engine

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device)) if normalize else None

        self.gate_proj = MuiLinear(engine_config, self.hidden_size, self.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = MuiLinear(engine_config, self.hidden_size, self.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = MuiLinear(engine_config, self.intermediate_size, self.hidden_size, bias=False, device=device, dtype=dtype)
        self.activation_function = activation_function

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # TODO: improve method selection
        self.method = _MuiGateUpSiLUMethod.GATEUPSILU_FUSED

    def _check_dispatchable(self):
        wdtype = self.gate_proj.weight.dtype
        dispatchable_activation = (isinstance(self.activation_function, nn.SiLU))
        dispatchable_type = (wdtype == torch.float16)
        dispatchable_device = self.gate_proj.weight.is_cuda
        self.dispatchable = dispatchable_activation and dispatchable_device and dispatchable_type

    @staticmethod
    def replace(prev_module: Union[LlamaMLP, MistralMLP], engine_config: MuiEngineConfig, prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None) -> "MuiGateUpDownMLP":
        dtype=prev_module.gate_proj.weight.dtype
        device=prev_module.gate_proj.weight.device

        hidden_size = prev_module.hidden_size
        intermediate_size = prev_module.intermediate_size
        activation_function = prev_module.act_fn

        normalize = prev_layernorm_module is not None
        variance_epsilon = prev_layernorm_module.variance_epsilon if normalize else 0.0
        norm_weights = prev_layernorm_module.weight if normalize else None

        new_module = MuiGateUpDownMLP(engine_config=engine_config, hidden_size=hidden_size, intermediate_size=intermediate_size, activation_function=activation_function, variance_epsilon=variance_epsilon, normalize=normalize, dtype=dtype, device=device)
        new_module.copy_module(prev_module=prev_module, norm_weights=norm_weights)

        return new_module

    def copy_module(self, prev_module: Union[LlamaMLP, MistralMLP], norm_weights: torch.Tensor = None, variance_epsilon: float = 0.0):
        self.gate_proj.copy_module(prev_module.gate_proj)
        self.up_proj.copy_module(prev_module.up_proj)

        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities

            norm_weights_requires_grad = norm_weights.requires_grad
            self.norm_weights = nn.Parameter(norm_weights.detach())
            self.norm_weights.requires_grad = norm_weights_requires_grad

            self.norm_weights = norm_weights

        self.down_proj.copy_module(prev_module.down_proj)

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _forward_unfused(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        # else: # not dispatchable or not MuiLinear
        if self.normalize:
            input = _MuiRMSNorm.apply(input, self.norm_weights, self.variance_epsilon)

        # we shard gate/up by rows so that the all_reduce from the gate/up linears can be avoided
        g = self.gate_proj(input)
        u = self.up_proj(input)

        output = self.down_proj(self.activation_function(g) * u)

        if residual is not None:
            output = output + residual

        return output

    def _forward_fused(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type

            # Also check that we don't have quantized linear
            if isinstance(self.gate_proj, MuiLinear) and isinstance(self.up_proj, MuiLinear):
                return _MuiGateUpSiLU.apply(self.cpp_engine, input, self.norm_weights, self.variance_epsilon, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight, residual)

        # else: # not dispatchable or not MuiLinear
        return self._forward_unfused(input=input, residual=residual)

    def _forward_split(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type

            # Also check that we don't have quantized linear
            if isinstance(self.gate_proj, MuiLinear) and isinstance(self.up_proj, MuiLinear):
                # we shard gate/up by rows so that we can still use the fused kernel and
                # the all_reduce from the gate/up linears can be avoided

                # as we shard gate/up by rows, we don't need to shard the input and we
                # still can use the fused RMSNorm
                return _MuiGateUpSiLUSplit.apply(self.cpp_engine, input, self.norm_weights, self.variance_epsilon, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight, residual)

        # else: # not dispatchable or not MuiLinear
        return self._forward_unfused(input=input, residual=residual)

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.method == _MuiGateUpSiLUMethod.GATEUPSILU_FUSED:
            return self._forward_fused(input=input, residual=residual)
        elif self.method == _MuiGateUpSiLUMethod.GATEUPSILU_UNFUSED:
            return self._forward_unfused(input=input, residual=residual)
        elif self.method == _MuiGateUpSiLUMethod.GATEUPSILU_SPLIT:
            return self._forward_split(input=input, residual=residual)
        else:
            raise ValueError("Unsupported Gate/Up Silu method")