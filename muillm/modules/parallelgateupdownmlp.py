from enum import Enum
import math
from typing import List, Optional, Union
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
from muillm.modules.module import MuiModule
from muillm.modules.parallellinear import MuiParallelLinear
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from muillm.engineconfig import MuiEngineConfig
from transformers.models.mistral.modeling_mistral import MistralMLP, MistralRMSNorm
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

from muillm.modules.rmsnorm import _MuiRMSNorm
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
    def forward(ctx, engine, comm, inputs, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual):
        output = muillm_ext.muillm_parallel_gateupsilu_forward(engine, comm.comms, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual, inputs)

        ctx.save_for_backward(inputs, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GateUpSiLU backward is not implemented")

class _MuiParallelGateUpSiLUSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, engine, comm, inputs, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual):
        output = muillm_ext.muillm_parallel_gateupsilu_split_forward(engine, comm.comms, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual, inputs)

        ctx.save_for_backward(inputs, norm_weights, variance_epsilon, gate_weights, up_weights, down_weights, residual)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GateUpSiLU split K backward is not implemented")

class MuiParallelGateUpDownMLP(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, hidden_size: int, intermediate_size: int, activation_function: nn.Module, variance_epsilon:float = 0.0, normalize:bool = False, device=None, dtype=None) -> None:
        super().__init__(engine_config=engine_config)

        self.tensor_parallelism = engine_config.tensor_parallelism
        self.cpp_engine = engine_config.cpp_engine
        self.comms = engine_config.comms

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = nn.ParameterList([torch.ones(hidden_size, dtype=dtype, device=device)]) if normalize else None

        self.gate_proj = MuiParallelLinear(engine_config, self.hidden_size, self.intermediate_size, bias=False, sharding_dim=0, device=device, dtype=dtype)
        self.up_proj = MuiParallelLinear(engine_config, self.hidden_size, self.intermediate_size, bias=False, sharding_dim=0, device=device, dtype=dtype)
        self.down_proj = MuiParallelLinear(engine_config, self.intermediate_size, self.hidden_size, bias=False, sharding_dim=1, device=device, dtype=dtype)
        self.activation_function = activation_function

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # TODO: improve method selection
        self.method = _MuiParallelGateUpSiLUMethod.GATEUPSILU_FUSED

    def _check_dispatchable(self):
        wdtype = self.gate_proj.dtype
        dispatchable_activation = (isinstance(self.activation_function, nn.SiLU))
        dispatchable_type = (wdtype == torch.float16)
        dispatchable_device = self.gate_proj.is_cuda
        self.dispatchable = dispatchable_activation and dispatchable_device and dispatchable_type

    @staticmethod
    def replace(prev_module: Union[LlamaMLP, MistralMLP, MuiGateUpDownMLP], engine_config: MuiEngineConfig, prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None) -> "MuiGateUpDownMLP":
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
        elif isinstance(prev_module, LlamaMLP):
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

    def copy_module(self, prev_module: Union[LlamaMLP, MistralMLP, MuiGateUpDownMLP], norm_weights: torch.Tensor = None, variance_epsilon: float = 0.0):
        self.gate_proj.copy_module(prev_module.gate_proj)
        self.up_proj.copy_module(prev_module.up_proj)

        if isinstance(prev_module, MuiGateUpDownMLP):
            if norm_weights is not None:
                raise ValueError("norm_weights should be None")
            norm_weights = prev_module.norm_weights
        elif isinstance(prev_module, MistralMLP):
            # norm_weights need to be set in calling args if needed
            pass
        elif isinstance(prev_module, LlamaMLP):
            # norm_weights need to be set in calling args if needed
            pass
        else:
            raise ValueError(f"Unsupported replacement: {prev_module.__class__.__name__}")

        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities

            norm_weights_requires_grad = norm_weights.requires_grad
            self.norm_weights = nn.ParameterList([norm_weights.detach()])
            MuiParallelLinear._set_requires_grads(self.norm_weights, norm_weights_requires_grad)

        self.down_proj.copy_module(prev_module.down_proj)

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def __collect_outputs(self, tensor: Tensor) -> Tensor:
        return MuiParallelLinear._collect_outputs(self.engine_config, tensor, sharding_dim=1)

    def _parallel_forward_unfused(self, inputs: Union[Tensor, List[Tensor]], residual: Optional[Tensor] = None) -> List[Tensor]:
        sharded_inputs = isinstance(inputs, list)
        if sharded_inputs:
            inputs = inputs[0]
        else:
            raise ValueError("not implemented")

        # else: # not dispatchable or not MuiLinear
        if self.normalize:
            norm_weights = self.norm_weights[0] if self.norm_weights is not None else None
            inputs = _MuiRMSNorm.apply(inputs, norm_weights, self.variance_epsilon)

        # we shard gate/up by rows so that the all_reduce from the gate/up linears can be avoided
        g = self.gate_proj.parallel_forward([inputs], collect_outputs=False)[0]
        u = self.up_proj.parallel_forward([inputs], collect_outputs=False)[0]

        gus = [self.activation_function(g) * u]
        outputs = self.down_proj.parallel_forward(gus, residual=residual)

        return outputs

    def _parallel_forward_fused(self, inputs: Union[Tensor, List[Tensor]], residual: Optional[Tensor] = None) -> List[Tensor]:
        sharded_inputs = isinstance(inputs, list)
        if sharded_inputs:
            inputs = inputs[0]
        else:
            raise ValueError("not implemented")

        if self.dispatchable and (inputs.numel() == inputs.shape[-1]):
            # input is effectively 1D, and we support the type

            # Also check that we don't have quantized linear
            if isinstance(self.gate_proj, MuiParallelLinear) and isinstance(self.up_proj, MuiParallelLinear):
                norm_weights = self.norm_weights[0] if self.norm_weights is not None else None
                # the kernel handles residual or not
                output = _MuiParallelGateUpSiLU.apply(self.cpp_engine, self.comms, inputs, norm_weights, self.variance_epsilon, self.gate_proj.weights[0], self.up_proj.weights[0], self.down_proj.weights[0], residual)

                return [output]

        # else: # not dispatchable or not MuiLinear
        return self._parallel_forward_unfused(inputs=[inputs], residual=residual)

    def _parallel_forward_split(self, inputs: Union[Tensor, List[Tensor]], residual: Optional[Tensor] = None) -> List[Tensor]:
        sharded_inputs = isinstance(inputs, list)
        if sharded_inputs:
            inputs = inputs[0]
        else:
            raise ValueError("not implemented")

        if self.dispatchable and (inputs.numel() == inputs.shape[-1]):
            # input is effectively 1D, and we support the type

            # Also check that we don't have quantized linear
            if isinstance(self.gate_proj, MuiParallelLinear) and isinstance(self.up_proj, MuiParallelLinear):
                # we shard gate/up by rows so that we can still use the fused kernel and
                # the all_reduce from the gate/up linears can be avoided

                # as we shard gate/up by rows, we don't need to shard the input and we
                # still can use the fused RMSNorm
                norm_weights = self.norm_weights[0] if self.norm_weights is not None else None
                # the kernel handles residual or not
                output = _MuiParallelGateUpSiLUSplit.apply(self.cpp_engine, self.comms, inputs, norm_weights, self.variance_epsilon, self.gate_proj.weights[0], self.up_proj.weights[0], self.down_proj.weights[0], residual)

                return [output]

        # else: # not dispatchable or not MuiLinear
        return self._parallel_forward_unfused(inputs=[inputs], residual=residual)


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