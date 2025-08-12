from typing import List, Optional, Union
from muillm.hftensorparallelism.hftensorparallelism import _to_local_module
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.module import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.norm.rmsnorm import _MuiRMSNorm, MuiRMSNorm
import muillm_ext
from muillm.replacement.replacementcontext import MuiReplacementContext


class _MuiLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, x, residual):
        output = muillm_ext.muillm_linear_module_forward(module, x, residual=residual)

        ctx.save_for_backward(x)

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


class MuiLinear(MuiModule, nn.Linear):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        in_features: int,
        out_features: int,
        bias: bool = True,
        norm: Optional[MuiRMSNorm] = None,
        device=None,
        dtype=None,
    ) -> None:
        MuiModule.__init__(self, engine_config=engine_config)
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.cpp_engine = engine_config.cpp_engine

        self.device = self.weight.device
        self.dtype = self.weight.dtype

        self.norm = norm

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # the cpp module will be created at the end of all layer replacements
        self.cpp_module = None

    def _check_dispatchable(self):
        wdtype = self.weight.dtype
        dispatchable_type = (wdtype == torch.float16) or (wdtype == torch.bfloat16)
        dispatchable_device = self.weight.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def finalize_init(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_linear_module_deinit(self.cpp_module)

        normalize = self.norm is not None
        bias = self.bias if self.bias is not None else None

        self.cpp_module = muillm_ext.muillm_linear_module_init(
            self.cpp_engine,
            self.weight,
            self.norm.weight if normalize else None,
            self.norm.variance_epsilon if normalize else 0.0,
            self.norm.weight_offset if normalize else 0.0,
            None,  # mul_bias
            bias,
        )

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _severe_ties(self):
        # severe ties to weights, biases and norm_weights
        weight = self.weight
        self.weight = None
        del weight

        if self.bias is not None:
            bias = self.bias
            self.bias = None
            del bias

        if self.norm is not None:
            del self.norm

        # destroy the C++ module as well to severe the ties to tensors
        if self.cpp_module is not None:
            muillm_ext.muillm_linear_module_deinit(self.cpp_module)
            self.cpp_module = None

    def finalize_deinit(self):
        self._severe_ties()

    @staticmethod
    def _set_requires_grads(param: nn.Parameter, requires_grads: bool) -> None:
        if param is not None:
            param.requires_grad = requires_grads

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_module: Union["MuiLinear", nn.Linear],
        prev_layernorm_module: Union[MuiRMSNorm, LlamaRMSNorm, MistralRMSNorm] = None,
    ) -> "MuiLinear":
        engine_config = replacement_context.engine_config
        device = replacement_context.device
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiLinear) and (prev_layernorm_module is None):
            # re-creating a module would change nothing, we can just avoid id
            return prev_module

        if not isinstance(prev_module, MuiLinear):
            # Make sure we convert the previous module to a local module
            # so that we can safely copy its parameters
            # and avoid any DTensor issues
            prev_module = replacement_context.to_local_module(prev_module)

        if (prev_layernorm_module is not None) and not isinstance(
            prev_layernorm_module, MuiRMSNorm
        ):
            # Make sure we convert the previous layernorm module to a local module
            # so that we can safely copy its parameters
            # and avoid any DTensor issues
            prev_layernorm_module = replacement_context.to_local_module(
                prev_layernorm_module
            )

        device = prev_module.weight.device if device is None else device
        dtype = prev_module.weight.dtype

        # put on the end device to accelerate things
        # (ok as we are replacing the module entirely so we can change its device)
        if device is not None:
            prev_module = prev_module.to(device)
            prev_layernorm_module = (
                prev_layernorm_module.to(device)
                if prev_layernorm_module is not None
                else None
            )

        has_bias = prev_module.bias is not None
        in_features = prev_module.in_features
        out_features = prev_module.out_features

        if isinstance(prev_module, MuiLinear):
            # due to replacement order, we might get the normalization weights already in
            # or in prev_layernorm_module
            # but not both
            if (prev_module.norm is not None) and (prev_layernorm_module is not None):
                raise ValueError(
                    "both norm weights in MuiLinear and layernorm module provided"
                )

            norm = (
                MuiRMSNorm.replace(
                    replacement_context,
                    prev_layernorm_module,
                )
                if prev_layernorm_module is not None
                else prev_module.norm
            )

        elif isinstance(prev_module, nn.Linear):
            norm = (
                MuiRMSNorm.replace(
                    replacement_context,
                    prev_layernorm_module,
                )
                if prev_layernorm_module is not None
                else None
            )
        else:
            raise ValueError(
                f"Unsupported replacement to MuiLinear: {prev_module.__class__.__name__}"
            )

        new_module = MuiLinear(
            engine_config=engine_config,
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            norm=norm,
            dtype=dtype,
            device=device,
        )
        new_module.copy_module(prev_module=prev_module, device=device)

        return new_module

    def _set_weights(
        self, weights: torch.Tensor, requires_grads: Optional[bool] = None
    ) -> None:
        weights_requires_grad = weights.requires_grad

        self.weight.data.copy_(weights.data)
        MuiLinear._set_requires_grads(
            self.weight,
            requires_grads if requires_grads is not None else weights_requires_grad,
        )

    def _set_bias(
        self, bias: torch.Tensor, requires_grads: Optional[bool] = None
    ) -> None:
        if bias is not None:
            bias_requires_grad = bias.requires_grad

            self.bias.data.copy_(bias.data)
            MuiLinear._set_requires_grads(
                self.bias,
                requires_grads if requires_grads is not None else bias_requires_grad,
            )
        else:
            self.bias = None

    def _set_norm_weights(
        self, norm_weights: torch.Tensor, requires_grads: Optional[bool] = None
    ) -> None:
        norm_weights_requires_grad = norm_weights.requires_grad

        self.norm.weight.data.copy_(norm_weights.data)
        MuiLinear._set_requires_grads(
            self.norm.weight,
            (
                requires_grads
                if requires_grads is not None
                else norm_weights_requires_grad
            ),
        )

    def copy_module(
        self,
        prev_module: Union["MuiLinear", nn.Linear],
        device=None,
    ):
        if device is None:
            raise ValueError("device was None")

        has_bias = prev_module.bias is not None

        self._set_weights(prev_module.weight)

        if has_bias:
            self._set_bias(prev_module.bias)

        # put ourselves on the right device
        self.to(device=device)

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.cpp_module is not None:
            return _MuiLinear.apply(
                self.cpp_module,
                input,
                residual,
            )
        else:
            if self.norm is not None:
                input = self.norm(input)

            output = F.linear(input, self.weight, self.bias)
            if residual is not None:
                output = output + residual
            return output
