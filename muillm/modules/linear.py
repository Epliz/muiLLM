from typing import List, Optional, Union
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.module import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.rmsnorm import _MuiRMSNorm, MuiRMSNorm
import muillm_ext


class _MuiLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, engine, x, weights, norm_weights, variance_epsilon, add_bias, residual
    ):
        if (add_bias is not None) and (residual is not None):
            raise ValueError("bias and residual at the same time is not supported")

        if residual is not None:
            add_bias = residual

        output = muillm_ext.muillm_linear_forward(
            engine,
            x,
            weights,
            norm_weights,
            variance_epsilon,
            mul_bias=None,
            add_bias=add_bias,
        )

        ctx.save_for_backward(x, weights, norm_weights, variance_epsilon, add_bias)

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
        variance_epsilon: float = 0.0,
        normalize: bool = False,
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

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = (
            nn.Parameter(torch.ones(in_features, dtype=dtype, device=device))
            if normalize
            else None
        )

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        wdtype = self.weight.dtype
        dispatchable_type = wdtype == torch.float16
        dispatchable_device = self.weight.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def finalize_init(self):
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

        if self.norm_weights is not None:
            norm_weights = self.norm_weights
            self.norm_weights = None
            del norm_weights

    def finalize_deinit(self):
        self._severe_ties()

    @staticmethod
    def _set_requires_grads(param: nn.Parameter, requires_grads: bool) -> None:
        if param is not None:
            param.requires_grad = requires_grads

    @staticmethod
    def replace(
        prev_module: Union["MuiLinear", nn.Linear],
        engine_config: MuiEngineConfig,
        prev_layernorm_module: Union[MuiRMSNorm, LlamaRMSNorm, MistralRMSNorm] = None,
        device=None,
    ) -> "MuiLinear":
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiLinear) and (prev_layernorm_module is None):
            # re-creating a module would change nothing, we can just avoid id
            return prev_module

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
            if (prev_module.normalize) and (prev_layernorm_module is not None):
                raise ValueError(
                    "both norm weights in MuiLinear and layernorm module provided"
                )

            if prev_module.normalize:
                normalize = True
                variance_epsilon = prev_module.variance_epsilon
                norm_weights = None  # needs to be None for copy_module
            elif prev_layernorm_module is not None:
                normalize = True
                variance_epsilon = prev_layernorm_module.variance_epsilon
                norm_weights = prev_layernorm_module.weight
            else:
                normalize = False
                variance_epsilon = 0.0
                norm_weights = None

        elif isinstance(prev_module, nn.Linear):
            normalize = prev_layernorm_module is not None
            variance_epsilon = (
                prev_layernorm_module.variance_epsilon if normalize else 0.0
            )
            norm_weights = prev_layernorm_module.weight if normalize else None
        else:
            raise ValueError(
                f"Unsupported replacement to MuiLinear: {prev_module.__class__.__name__}"
            )

        new_module = MuiLinear(
            engine_config=engine_config,
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            variance_epsilon=variance_epsilon,
            normalize=normalize,
            dtype=dtype,
            device=device,
        )
        new_module.copy_module(
            prev_module=prev_module, norm_weights=norm_weights, device=device
        )

        # delete the previous module to save memory
        if isinstance(prev_module, MuiLinear):
            prev_module._severe_ties()
        else:
            prev_weights = prev_module.weight
            prev_module.weight = None
            del prev_weights

            if prev_module.bias is not None:
                prev_bias = prev_module.bias
                prev_module.bias = None
                del prev_bias

        if prev_layernorm_module is not None:
            prev_layernorm_weights = prev_layernorm_module.weight
            prev_layernorm_module.weight = None
            del prev_layernorm_weights

        del prev_module

        # trigger GC to save memory
        trigger_gc()

        return new_module

    def _set_weights(self, weights: torch.Tensor) -> None:
        weights_requires_grad = weights.requires_grad
        self.weight = nn.Parameter(weights.clone().detach())
        MuiLinear._set_requires_grads(self.weight, weights_requires_grad)

    def _set_bias(self, bias: torch.Tensor) -> None:
        if bias is not None:
            bias_requires_grad = bias.requires_grad
            self.bias = nn.Parameter(bias.clone().detach())
            MuiLinear._set_requires_grads(self.bias, bias_requires_grad)
        else:
            self.bias = None

    def _set_norm_weights(self, norm_weights: torch.Tensor) -> None:
        norm_weights_requires_grad = norm_weights.requires_grad
        self.norm_weights = nn.Parameter(norm_weights.clone().detach())
        MuiLinear._set_requires_grads(self.norm_weights, norm_weights_requires_grad)

    def copy_module(
        self,
        prev_module: Union["MuiLinear", nn.Linear],
        norm_weights: torch.Tensor = None,
        variance_epsilon: float = 0.0,
        device=None,
    ):
        if device is None:
            raise ValueError("device was None")

        has_bias = prev_module.bias is not None

        self._set_weights(prev_module.weight)

        if has_bias:
            self._set_bias(prev_module.bias)

        if isinstance(prev_module, MuiLinear):
            # MuiLinear inherits nn.Linear, so need to check first
            if (prev_module.norm_weights is not None) and (norm_weights is not None):
                raise ValueError(
                    "both norm weights in MuiLinear and norm_weight provided"
                )

            if prev_module.normalize:
                norm_weights = prev_module.norm_weights
        elif isinstance(prev_module, nn.Linear):
            # norm_weights need to be set in calling args if needed
            pass
        else:
            raise ValueError(
                f"Unsupported replacement: {prev_module.__class__.__name__}"
            )

        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities
            self._set_norm_weights(norm_weights)

        # put ourselves on the right device
        self.to(device=device)

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:

        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type
            return _MuiLinear.apply(
                self.cpp_engine,
                input,
                self.weight,
                self.norm_weights,
                self.variance_epsilon,
                self.bias,
                residual,
            )
        else:
            if self.normalize:
                input = _MuiRMSNorm.apply(
                    input, self.norm_weights, self.variance_epsilon
                )

            output = F.linear(input, self.weight, self.bias)
            if residual is not None:
                output = output + residual
            return output
