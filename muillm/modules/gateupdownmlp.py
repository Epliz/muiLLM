from enum import Enum
import math
from typing import Optional, Union
from muillm.hftensorparallelism.hftensorparallelism import _to_local_module
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.module import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from muillm.engineconfig import MuiEngineConfig
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm
from transformers.models.llama4.modeling_llama4 import Llama4TextMLP, Llama4TextRMSNorm
from transformers.models.mistral.modeling_mistral import MistralMLP, MistralRMSNorm

from muillm.modules.linear import MuiLinear

from muillm.modules.norm.rmsnorm import _MuiRMSNorm, MuiRMSNorm
import muillm_ext
from muillm.replacement.replacementcontext import MuiReplacementContext


class _MuiGateUpMLPMethod(Enum):
    # Basic method where Gate/Up projections + mul are done distinctly
    GATEUPMLP_UNFUSED = 0
    # Method where the Gate/Up projections + mul are all fused
    GATEUPMLP_FUSED = 1
    # Method where the Gate/Up projections are done in the same kernel
    # but split between blocks to have more blocks.
    # A final reduction is done in an epilogue kernel
    GATEUPMLP_SPLIT = 2


class _MuiGateUpMLP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        engine,
        inputs,
        norm_weights,
        variance_epsilon,
        gate_weights,
        up_weights,
        down_weights,
        residual,
    ):
        output = muillm_ext.muillm_gateupmlp_forward(
            engine,
            norm_weights,
            variance_epsilon,
            gate_weights,
            up_weights,
            down_weights,
            residual,
            inputs,
        )

        ctx.save_for_backward(
            inputs,
            norm_weights,
            gate_weights,
            up_weights,
            down_weights,
            residual,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GateUpMLP backward is not implemented")


class _MuiGateUpMLPSplit(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        engine,
        inputs,
        norm_weights,
        variance_epsilon,
        gate_weights,
        up_weights,
        down_weights,
        residual,
    ):
        output = muillm_ext.muillm_gateupmlp_split_forward(
            engine,
            norm_weights,
            variance_epsilon,
            gate_weights,
            up_weights,
            down_weights,
            residual,
            inputs,
        )

        ctx.save_for_backward(
            inputs,
            norm_weights,
            gate_weights,
            up_weights,
            down_weights,
            residual,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GateUpMLP split K backward is not implemented")


class MuiGateUpDownMLP(MuiModule):
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
        self.cpp_engine = engine_config.cpp_engine

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = (
            nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
            if normalize
            else None
        )

        self.gate_proj = MuiLinear(
            engine_config,
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up_proj = MuiLinear(
            engine_config,
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = MuiLinear(
            engine_config,
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.activation_function = activation_function

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # TODO: improve method selection
        self.method = _MuiGateUpMLPMethod.GATEUPMLP_FUSED

    def _check_dispatchable(self):
        wdtype = self.gate_proj.weight.dtype
        dispatchable_activation = isinstance(self.activation_function, nn.SiLU)
        dispatchable_type = (wdtype == torch.float16) or (wdtype == torch.bfloat16)
        dispatchable_device = self.gate_proj.weight.is_cuda
        self.dispatchable = (
            dispatchable_activation and dispatchable_device and dispatchable_type
        )

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        self.gate_proj.finalize_init()
        self.up_proj.finalize_init()
        self.down_proj.finalize_init()

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_module: Union["MuiGateUpDownMLP", LlamaMLP, MistralMLP, Llama4TextMLP],
        prev_layernorm_module: Union[
            MuiRMSNorm, LlamaRMSNorm, MistralRMSNorm, Llama4TextRMSNorm
        ] = None,
    ) -> "MuiGateUpDownMLP":
        engine_config = replacement_context.engine_config
        device = replacement_context.device

        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiGateUpDownMLP) and (
            prev_layernorm_module is None
        ):
            # re-creating a module would replace nothing so we can avoid it
            return prev_module

        if not isinstance(prev_module, MuiGateUpDownMLP):
            # Make sure we convert the previous module to a local module
            # so that we can safely copy its parameters
            # and avoid any DTensor issues
            prev_module = replacement_context.to_local_module(prev_module)

        if (prev_layernorm_module is not None) and (
            not isinstance(prev_layernorm_module, MuiRMSNorm)
        ):
            # Make sure we convert the previous layernorm module to a local module
            # so that we can safely copy its parameters
            # and avoid any DTensor issues
            prev_layernorm_module = replacement_context.to_local_module(
                prev_layernorm_module
            )

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
        intermediate_size = prev_module.up_proj.out_features

        if isinstance(prev_module, Llama4TextMLP):
            # Llama4TextMLP has a different activation function
            activation_function = prev_module.activation_fn
        elif isinstance(prev_module, (LlamaMLP, MistralMLP)):
            activation_function = prev_module.act_fn
        else:
            raise ValueError(
                f"Unsupported module type {type(prev_module)} for replacement"
            )

        if isinstance(prev_module, MuiGateUpDownMLP):
            # due to replacement order, we might get the normalization weights already in
            # or in prev_layernorm_module
            # but not both
            if (prev_module.normalize) and (prev_layernorm_module is not None):
                raise ValueError(
                    "both norm weights in MuiGateUpDownMLP and layernorm module provided"
                )

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
        else:
            normalize = prev_layernorm_module is not None
            variance_epsilon = (
                MuiRMSNorm._extract_eps(prev_layernorm_module) if normalize else 0.0
            )
            norm_weights = prev_layernorm_module.weight if normalize else None

        new_module = MuiGateUpDownMLP(
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

        # delete the previous modules to free memory
        if not isinstance(prev_module, MuiGateUpDownMLP):
            del prev_module.gate_proj
            del prev_module.up_proj
            del prev_module.down_proj

            # trigger garbage collection to free memory
            trigger_gc()

        return new_module

    def copy_module(
        self,
        prev_module: Union["MuiGateUpDownMLP", LlamaMLP, MistralMLP, Llama4TextMLP],
        norm_weights: torch.Tensor = None,
        variance_epsilon: float = 0.0,
        device=None,
    ):
        if device is None:
            raise ValueError("device was None")

        self.gate_proj.copy_module(prev_module.gate_proj, device=device)
        self.up_proj.copy_module(prev_module.up_proj, device=device)

        if isinstance(prev_module, MuiGateUpDownMLP):
            if (prev_module.norm_weights is not None) and (norm_weights is not None):
                raise ValueError(
                    "both norm weights in MuiGateUpDownMLP and norm_weight provided"
                )

            if prev_module.norm_weights is not None:
                norm_weights = prev_module.norm_weights

        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities
            norm_weights_requires_grad = norm_weights.requires_grad
            self.norm_weights = nn.Parameter(norm_weights.clone().detach())
            self.norm_weights.requires_grad = norm_weights_requires_grad

            self.norm_weights = norm_weights

        self.down_proj.copy_module(prev_module.down_proj, device=device)

        # put ourselves on the right device
        self.to(device=device)

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _forward_unfused(
        self, input: Tensor, residual: Optional[Tensor] = None
    ) -> Tensor:
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

    def _forward_fused(
        self, input: Tensor, residual: Optional[Tensor] = None
    ) -> Tensor:
        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type

            # Also check that we don't have quantized linear
            if isinstance(self.gate_proj, MuiLinear) and isinstance(
                self.up_proj, MuiLinear
            ):
                return _MuiGateUpMLP.apply(
                    self.cpp_engine,
                    input,
                    self.norm_weights,
                    self.variance_epsilon,
                    self.gate_proj.weight,
                    self.up_proj.weight,
                    self.down_proj.weight,
                    residual,
                )

        # else: # not dispatchable or not MuiLinear
        return self._forward_unfused(input=input, residual=residual)

    def _forward_split(
        self, input: Tensor, residual: Optional[Tensor] = None
    ) -> Tensor:
        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type

            # Also check that we don't have quantized linear
            if isinstance(self.gate_proj, MuiLinear) and isinstance(
                self.up_proj, MuiLinear
            ):
                # we shard gate/up by rows so that we can still use the fused kernel and
                # the all_reduce from the gate/up linears can be avoided

                # as we shard gate/up by rows, we don't need to shard the input and we
                # still can use the fused RMSNorm
                return _MuiGateUpMLPSplit.apply(
                    self.cpp_engine,
                    input,
                    self.norm_weights,
                    self.variance_epsilon,
                    self.gate_proj.weight,
                    self.up_proj.weight,
                    self.down_proj.weight,
                    residual,
                )

        # else: # not dispatchable or not MuiLinear
        return self._forward_unfused(input=input, residual=residual)

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.method == _MuiGateUpMLPMethod.GATEUPMLP_FUSED:
            return self._forward_fused(input=input, residual=residual)
        elif self.method == _MuiGateUpMLPMethod.GATEUPMLP_UNFUSED:
            return self._forward_unfused(input=input, residual=residual)
        elif self.method == _MuiGateUpMLPMethod.GATEUPMLP_SPLIT:
            return self._forward_split(input=input, residual=residual)
        else:
            raise ValueError("Unsupported Gate/Up Silu method")
