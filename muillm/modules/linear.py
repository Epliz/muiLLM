from typing import List, Optional, Union
from muillm.modules.module import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.rmsnorm import _MuiRMSNorm
import muillm_ext

class _MuiLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, engine, x, weights, norm_weights, variance_epsilon, add_bias, residual):
        if (add_bias is not None) and (residual is not None):
            raise ValueError("bias and residual at the same time is not supported")

        if residual is not None:
            add_bias = residual

        output = muillm_ext.muillm_linear_forward(engine, x, weights, norm_weights, variance_epsilon, mul_bias=None, add_bias=add_bias)

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
    def __init__(self, engine_config: MuiEngineConfig, in_features: int, out_features: int, bias: bool = True,
                 variance_epsilon:float = 0.0, normalize:bool = False, device=None, dtype=None) -> None:
        MuiModule.__init__(self, engine_config=engine_config)
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)

        self.cpp_engine = engine_config.cpp_engine

        self.device = self.weight.device
        self.dtype = self.weight.dtype

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = nn.Parameter(torch.ones(in_features, dtype=dtype, device=device)) if normalize else None

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        wdtype = self.weight.dtype
        dispatchable_type = (wdtype == torch.float16)
        dispatchable_device = self.weight.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    @staticmethod
    def _set_requires_grads(param: nn.Parameter, requires_grads: bool) -> None:
        if param is not None:
            param.requires_grad = requires_grads

    @staticmethod
    def replace(prev_module: nn.Linear, engine_config: MuiEngineConfig, prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None) -> "MuiLinear":
        has_bias = prev_module.bias is not None
        in_features = prev_module.in_features
        out_features = prev_module.out_features

        normalize = prev_layernorm_module is not None
        variance_epsilon = prev_layernorm_module.variance_epsilon if normalize else 0.0
        norm_weights = prev_layernorm_module.weight if normalize else None

        new_module = MuiLinear(engine_config=engine_config, in_features=in_features, out_features=out_features, bias=has_bias, variance_epsilon=variance_epsilon, normalize=normalize, dtype=prev_module.weight.dtype, device=prev_module.weight.device)
        new_module.copy_module(prev_module=prev_module, norm_weights=norm_weights)

        return new_module
    
    def _set_norm_weights(self, norm_weights: torch.Tensor) -> None:
        norm_weights_requires_grad = norm_weights.requires_grad
        self.norm_weights = nn.Parameter(norm_weights.detach())
        MuiLinear._set_requires_grads(self.norm_weights, norm_weights_requires_grad)

    def copy_module(self, prev_module: nn.Linear, norm_weights: torch.Tensor = None, variance_epsilon: float = 0.0):
        has_bias = prev_module.bias is not None

        self.weight = nn.Parameter(prev_module.weight)
        MuiLinear._set_requires_grads(self.weight, prev_module.weight.requires_grad)

        if has_bias:
            self.bias = nn.Parameter(prev_module.bias) if prev_module.bias is not None else None
            MuiLinear._set_requires_grads(self.bias, prev_module.bias.requires_grad)

        if isinstance(prev_module, MuiLinear):
            # MuiLinear inherits nn.Linear, so need to check first
            if norm_weights is not None:
                raise ValueError("norm_weights should be None")
            norm_weights = prev_module.norm_weights
        elif isinstance(prev_module, nn.Linear):
            # norm_weights need to be set in calling args if needed
            pass
        else:
            raise ValueError(f"Unsupported replacement: {prev_module.__class__.__name__}")

        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities
            self._set_norm_weights(norm_weights)

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:

        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type
            return _MuiLinear.apply(self.cpp_engine, input, self.weight, self.norm_weights, self.variance_epsilon, self.bias, residual)
        else:
            if self.normalize:
                input = _MuiRMSNorm.apply(input, self.norm_weights, self.variance_epsilon)

            output = F.linear(input, self.weight, self.bias)
            if residual is not None:
                output = output + residual
            return output