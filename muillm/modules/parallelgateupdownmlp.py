from enum import IntEnum
import math
from typing import List, Optional, Union
from muillm.memorymanagement.gc import trigger_gc
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
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextMLP,
    Llama4TextRMSNorm,
)

from muillm.modules.rmsnorm import _MuiRMSNorm, MuiRMSNorm
import muillm_ext


class _MuiParallelGateUpSiLUMethod(IntEnum):
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
    def forward(ctx, module, inputs, residual, collect_outputs):
        output = muillm_ext.muillm_parallel_gateupdownmlp_module_forward(
            module, inputs, residual, collect_outputs
        )

        ctx.save_for_backward(inputs, residual)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GateUpSiLU backward is not implemented")


class MuiParallelGateUpDownMLP(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        hidden_size: int,
        intermediate_size: int,
        activation_function: nn.Module,
        variance_epsilon: float = 0.0,
        normalize: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(engine_config=engine_config)

        self.tensor_parallelism = engine_config.tensor_parallelism
        self.cpp_engine = engine_config.cpp_engine
        self.comms = engine_config.comms

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = (
            nn.ParameterList([torch.ones(hidden_size, dtype=dtype, device=device)])
            if normalize
            else None
        )

        # We shard the gate and up projections by row so that we don't have to
        # do an all reduce before the down projection

        self.gate_proj = MuiParallelLinear(
            engine_config,
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            sharding_dim=0,
            device=device,
            dtype=dtype,
        )
        self.up_proj = MuiParallelLinear(
            engine_config,
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            sharding_dim=0,
            device=device,
            dtype=dtype,
        )
        self.down_proj = MuiParallelLinear(
            engine_config,
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            sharding_dim=1,
            device=device,
            dtype=dtype,
        )
        self.activation_function = activation_function

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # the cpp module will be created at the end of all layer replacements
        self.cpp_module = None

        # TODO: improve method selection
        self.method = _MuiParallelGateUpSiLUMethod.GATEUPSILU_FUSED

    def finalize_init(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_gateupdownmlp_module_deinit(self.cpp_module)

        norm_weights = self.norm_weights[0] if self.norm_weights is not None else None

        self.cpp_module = muillm_ext.muillm_parallel_gateupdownmlp_module_init(
            self.cpp_engine,
            self.comms.comms,
            int(self.method),
            norm_weights,
            self.gate_proj.weights[0],
            self.up_proj.weights[0],
            self.down_proj.weights[0],
            self.variance_epsilon,
        )

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        wdtype = self.gate_proj.dtype
        dispatchable_activation = isinstance(self.activation_function, nn.SiLU)
        dispatchable_type = wdtype == torch.float16
        dispatchable_device = self.gate_proj.is_cuda
        self.dispatchable = (
            dispatchable_activation and dispatchable_device and dispatchable_type
        )

    def finalize_deinit(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_gateupdownmlp_module_deinit(self.cpp_module)
            self.cpp_module = None

    @staticmethod
    def replace(
        prev_module: Union[
            "MuiParallelGateUpDownMLP",
            MuiGateUpDownMLP,
            LlamaMLP,
            MistralMLP,
            Llama4TextMLP,
        ],
        engine_config: MuiEngineConfig,
        prev_layernorm_module: Union[
            LlamaRMSNorm, MistralRMSNorm, Llama4TextRMSNorm
        ] = None,
        device=None,
    ) -> "MuiGateUpDownMLP":
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiParallelGateUpDownMLP) and (
            prev_layernorm_module is None
        ):
            # re-creating a module would replace nothing so we can avoid it
            return prev_module

        if isinstance(prev_module.gate_proj, MuiParallelLinear):
            dtype = prev_module.gate_proj.weights[0].dtype
            device = (
                prev_module.gate_proj.weights[0].device if device is None else device
            )
        else:
            dtype = prev_module.gate_proj.weight.dtype
            device = prev_module.gate_proj.weight.device if device is None else device

        # put on the end device to accelerate things
        # (ok as we are replacing the module entirely so we can change its device)
        if device is not None:
            prev_module = prev_module.to(device)
            prev_layernorm_module = (
                prev_layernorm_module.to(device)
                if prev_layernorm_module is not None
                else None
            )

        hidden_size = prev_module.gate_proj.in_features
        intermediate_size = prev_module.gate_proj.out_features

        if isinstance(prev_module, MuiParallelGateUpDownMLP) or isinstance(
            prev_module, MuiGateUpDownMLP
        ):
            # due to replacement order, we might get the normalization weights already in
            # or in prev_layernorm_module
            # but not both
            if (prev_module.normalize) and (prev_layernorm_module is not None):
                raise ValueError(
                    "both norm weights in MuiParallelGateUpDownMLP/MuiGateUpDownMLP and layernorm module provided"
                )

            activation_function = prev_module.activation_function

            if prev_module.normalize:
                normalize = True
                variance_epsilon = prev_module.variance_epsilon
                norm_weights = None  # needs to be None for copy_module
            elif prev_layernorm_module is not None:
                normalize = True
                variance_epsilon = MuiRMSNorm._extract_eps(prev_layernorm_module)
                norm_weights = prev_layernorm_module.weight
            else:
                normalize = False
                variance_epsilon = 0.0
                norm_weights = None

        elif (
            isinstance(prev_module, MistralMLP)
            or isinstance(prev_module, LlamaMLP)
            or isinstance(prev_module, Llama4TextMLP)
        ):
            if isinstance(prev_module, Llama4TextMLP):
                # Llama4TextMLP has a different activation function
                activation_function = prev_module.activation_fn
            else:
                activation_function = prev_module.act_fn

            normalize = prev_layernorm_module is not None
            variance_epsilon = (
                MuiRMSNorm._extract_eps(prev_layernorm_module) if normalize else 0.0
            )
            norm_weights = prev_layernorm_module.weight if normalize else None
        else:
            raise ValueError(
                f"Unsupported replacement: {prev_module.__class__.__name__}"
            )

        new_module = MuiParallelGateUpDownMLP(
            engine_config=engine_config,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            variance_epsilon=variance_epsilon,
            normalize=normalize,
            dtype=dtype,
            device=device,
        )
        new_module.copy_module(
            prev_module=prev_module, norm_weights=norm_weights, device=device
        )

        # delete the previous module to save memory
        del prev_module

        # trigger GC to save memory
        trigger_gc()

        return new_module

    def copy_module(
        self,
        prev_module: Union[
            "MuiParallelGateUpDownMLP",
            LlamaMLP,
            MistralMLP,
            Llama4TextRMSNorm,
            MuiGateUpDownMLP,
        ],
        norm_weights: torch.Tensor = None,
        variance_epsilon: float = 0.0,
        device=None,
    ):
        if device is None:
            raise ValueError("device was None")

        self.gate_proj.copy_module(prev_module.gate_proj, device=device)
        self.up_proj.copy_module(prev_module.up_proj, device=device)

        if isinstance(prev_module, MuiParallelGateUpDownMLP) or isinstance(
            prev_module, MuiGateUpDownMLP
        ):
            if (prev_module.norm_weights is not None) and (norm_weights is not None):
                raise ValueError(
                    "both norm weights in MuiParallelGateUpDownMLP/MuiGateUpDownMLP and norm_weight provided"
                )

            if prev_module.norm_weights is not None:
                norm_weights = prev_module.norm_weights
        elif isinstance(prev_module, (MistralMLP, LlamaMLP, Llama4TextMLP)):
            # norm_weights need to be set in calling args if needed
            pass
        else:
            raise ValueError(
                f"Unsupported replacement: {prev_module.__class__.__name__}"
            )

        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities

            norm_weights_requires_grad = norm_weights.requires_grad
            self.norm_weights = nn.ParameterList([norm_weights.detach()])
            MuiParallelLinear._set_requires_grads(
                self.norm_weights, norm_weights_requires_grad
            )

        self.down_proj.copy_module(prev_module.down_proj, device=device)

        # put ourselves on the right device
        self.to(device=device)

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        self.finalize_init()

    def __collect_outputs(self, tensor: Tensor) -> Tensor:
        return MuiParallelLinear._collect_outputs(
            self.engine_config, tensor, sharding_dim=1
        )

    def _parallel_forward_unfused(
        self,
        inputs: Tensor,
        residual: Optional[Tensor] = None,
        collect_outputs: bool = True,
    ) -> List[Tensor]:
        if self.normalize:
            norm_weights = (
                self.norm_weights[0] if self.norm_weights is not None else None
            )
            inputs = _MuiRMSNorm.apply(inputs, norm_weights, self.variance_epsilon)

        # we shard gate/up by rows so that the all_reduce from the gate/up linears can be avoided
        g = self.gate_proj.parallel_forward([inputs], collect_outputs=False)[0]
        u = self.up_proj.parallel_forward([inputs], collect_outputs=False)[0]

        gus = [self.activation_function(g) * u]
        outputs = self.down_proj.parallel_forward(
            gus, residual=residual, collect_outputs=collect_outputs
        )

        return outputs

    def parallel_forward(
        self,
        inputs: Union[Tensor, List[Tensor]],
        residual: Optional[Tensor] = None,
        collect_outputs: bool = True,
    ) -> List[Tensor]:
        sharded_inputs = isinstance(inputs, list)
        if sharded_inputs:
            inputs = inputs[0]
        else:
            raise ValueError("not implemented")

        if self.dispatchable and (inputs.numel() == inputs.shape[-1]):
            output = _MuiParallelGateUpSiLU.apply(
                self.cpp_module, inputs, residual, collect_outputs
            )
            return [output]
        else:
            return self._parallel_forward_unfused(
                inputs=inputs, residual=residual, collect_outputs=collect_outputs
            )

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.tensor_parallelism > 1:
            return self.parallel_forward([input], residual)[0]

        raise ValueError("Only parallel inference is supported")
