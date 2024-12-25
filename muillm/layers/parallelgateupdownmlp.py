from enum import Enum
import math
from typing import List, Optional, Union
from muillm.layers.gateupdownmlp import MuiGateUpDownMLP
from muillm.layers.module import MuiModule
from muillm.layers.parallellinear import MuiParallelLinear
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

class _MuiParallelGateUpSiLUMethod(Enum):
    # Basic method where Gate/Up projections + mul are done distinctly
    GATEUPSILU_UNFUSED = 0
    # Method where the Gate/Up projections + mul are all fused
    GATEUPSILU_FUSED = 1
    # Method where the Gate/Up projections are done in the same kernel
    # but split between blocks to have more blocks.
    # A final reduction is done in an epilogue kernel
    GATEUPSILU_SPLIT = 2

class _MuiParallelGateUpSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, norm_weights, variance_epsilon, gate_weights, up_weights):
        output = muillm_ext.muillm_gateupsilu_forward(norm_weights, variance_epsilon, gate_weights, up_weights, inputs)

        ctx.save_for_backward(inputs, norm_weights, variance_epsilon, gate_weights, up_weights)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GateUpSiLU backward is not implemented")

class _MuiParallelGateUpSiLUSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, norm_weights, variance_epsilon, gate_weights, up_weights):
        output = muillm_ext.muillm_gateupsilu_split_forward(norm_weights, variance_epsilon, gate_weights, up_weights, inputs)

        ctx.save_for_backward(inputs, norm_weights, variance_epsilon, gate_weights, up_weights)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GateUpSiLU split K backward is not implemented")

class MuiParallelGateUpDownMLP(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, hidden_size: int, intermediate_size: int, activation_function: nn.Module, variance_epsilon:float = 0.0, normalize:bool = False, device=None, dtype=None) -> None:
        super().__init__(engine_config=engine_config)

        self.tensor_parallelism = engine_config.tensor_parallelism

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = nn.ParameterList([torch.ones(hidden_size, dtype=dtype, device=d) for d in self.engine_config.devices]) if normalize else None

        self.gate_proj = MuiParallelLinear(engine_config, self.hidden_size, self.intermediate_size, bias=False, sharding_dim=0, device=device, dtype=dtype)
        self.up_proj = MuiParallelLinear(engine_config, self.hidden_size, self.intermediate_size, bias=False, sharding_dim=0, device=device, dtype=dtype)
        self.down_proj = MuiParallelLinear(engine_config, self.intermediate_size, self.hidden_size, bias=False, sharding_dim=1, device=device, dtype=dtype)
        self.activation_function = activation_function

        wdtype = self.gate_proj.dtype
        dispatchable_activation = (isinstance(self.activation_function, nn.SiLU))
        dispatchable_type = (wdtype == torch.float16)
        dispatchable_device = self.gate_proj.is_cuda
        self.dispatchable = dispatchable_activation and dispatchable_device and dispatchable_type

        # TODO: improve method selection
        self.method = _MuiParallelGateUpSiLUMethod.GATEUPSILU_FUSED

    @staticmethod
    def replace(prev_module: Union[MistralMLP, MuiGateUpDownMLP], engine_config: MuiEngineConfig, prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None) -> "MuiGateUpDownMLP":
        dtype=prev_module.gate_proj.weight.dtype
        device=prev_module.gate_proj.weight.device

        if isinstance(prev_module, MuiGateUpDownMLP):
            hidden_size = prev_module.hidden_size
            intermediate_size = prev_module.intermediate_size
            activation_function = prev_module.activation_function

            normalize = prev_module.normalize
            variance_epsilon = prev_module.variance_epsilon
            norm_weights = None # will be taken from previous module
        elif isinstance(prev_module, MistralMLP):
            hidden_size = prev_module.hidden_size
            intermediate_size = prev_module.intermediate_size
            activation_function = prev_module.act_fn

            normalize = prev_layernorm_module is not None
            variance_epsilon = prev_layernorm_module.variance_epsilon if normalize else 0.0
            norm_weights = prev_layernorm_module.weight if normalize else None
        else:
            raise ValueError(f"Unsupported replacement: {prev_module.__class__.__name__}")


        new_module = MuiParallelGateUpDownMLP(engine_config=engine_config, hidden_size=hidden_size, intermediate_size=intermediate_size, activation_function=activation_function, variance_epsilon=variance_epsilon, normalize=normalize, dtype=dtype, device=device)
        new_module.copy_module(prev_module=prev_module, norm_weights=norm_weights)

        return new_module

    def copy_module(self, prev_module: Union[MistralMLP, MuiGateUpDownMLP], norm_weights: torch.Tensor = None, variance_epsilon: float = 0.0):
        self.gate_proj.copy_module(prev_module.gate_proj)
        self.up_proj.copy_module(prev_module.up_proj)

        if isinstance(prev_module, MuiGateUpDownMLP):
            if norm_weights is not None:
                raise ValueError("norm_weights should be None")
            norm_weights = prev_module.norm_weights
        elif isinstance(prev_module, MistralMLP):
            # norm_weights need to be set in calling args if needed
            pass
        else:
            raise ValueError(f"Unsupported replacement: {prev_module.__class__.__name__}")

        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities

            norm_weights_requires_grad = norm_weights.requires_grad
            self.norm_weights = nn.ParameterList([norm_weights.detach().to(device=d) for d in self.engine_config.devices])
            MuiParallelLinear._set_requires_grads(self.norm_weights, norm_weights_requires_grad)

        self.down_proj.copy_module(prev_module.down_proj)

    def _parallel_forward_unfused(self, inputs: Union[Tensor, List[Tensor]], residual: Optional[Tensor] = None) -> List[Tensor]:
        # else: # not dispatchable or not MuiLinear
        if self.normalize:
            if isinstance(inputs, list):
                # normalize on each GPU independently instead of doing on GPU0 then copy (saves a copy)
                inputs = [_MuiRMSNorm.apply(input, self.norm_weights[d], self.variance_epsilon) for d, input in enumerate(inputs)]
            else:
                inputs = _MuiRMSNorm.apply(inputs, self.norm_weights[0], self.variance_epsilon)

        # we shard gate/up by rows so that the all_reduce from the gate/up linears can be avoided
        gs = self.gate_proj.parallel_forward(inputs, collect_outputs=False)
        us = self.up_proj.parallel_forward(inputs, collect_outputs=False)

        gus = [self.activation_function(g) * u for g, u in zip(gs, us)]
        outputs = self.down_proj.parallel_forward(gus, residual=residual)

        return outputs

    def _parallel_forward_fused(self, inputs: Union[Tensor, List[Tensor]], residual: Optional[Tensor] = None) -> List[Tensor]:
        if False: #self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type

            # Also check that we don't have quantized linear
            if isinstance(self.gate_proj, MuiLinear) and isinstance(self.up_proj, MuiLinear):
                gateup = _MuiParallelGateUpSiLU.apply(input, self.norm_weights[d], self.variance_epsilon, self.gate_proj.weight, self.up_proj.weight)
                return self.down_proj(gateup, residual=residual)

        # else: # not dispatchable or not MuiLinear
        return self._parallel_forward_unfused(inputs=inputs, residual=residual)

    def _parallel_forward_split(self, inputs: Union[Tensor, List[Tensor]], residual: Optional[Tensor] = None) -> List[Tensor]:
        if False: #self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type

            # Also check that we don't have quantized linear
            if isinstance(self.gate_proj, MuiLinear) and isinstance(self.up_proj, MuiLinear):
                # we shard gate/up by rows so that we can still use the fused kernel and
                # the all_reduce from the gate/up linears can be avoided

                # as we shard gate/up by rows, we don't need to shard the input and we
                # still can use the fused RMSNorm
                gateup = _MuiParallelGateUpSiLUSplit.apply(input, self.norm_weights[d], self.variance_epsilon, self.gate_proj.weight, self.up_proj.weight)

                # we need to do an all_reduce here if we are using sharding (tensor parallelism)
                return self.down_proj(gateup, residual=residual)

        # else: # not dispatchable or not MuiLinear
        return self._parallel_forward_unfused(inputs=inputs, residual=residual)


    def parallel_forward(self, inputs: Union[Tensor, List[Tensor]], residual: Optional[Tensor] = None) -> List[Tensor]:
        if self.method == _MuiParallelGateUpSiLUMethod.GATEUPSILU_FUSED:
            return self._parallel_forward_fused(inputs=inputs, residual=residual)
        elif self.method == _MuiParallelGateUpSiLUMethod.GATEUPSILU_UNFUSED:
            return self._parallel_forward_unfused(inputs=inputs, residual=residual)
        elif self.method == _MuiParallelGateUpSiLUMethod.GATEUPSILU_SPLIT:
            return self._parallel_forward_split(inputs=inputs, residual=residual)
        else:
            raise ValueError("Unsupported Gate/Up Silu method")

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.tensor_parallelism > 1:
            return self.parallel_forward(input, residual)[0]

        raise ValueError("Only parallel inference is supported")