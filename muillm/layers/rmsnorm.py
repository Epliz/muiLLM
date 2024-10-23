from typing import Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import muillm_ext

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from muillm.engineconfig import MuiEngineConfig

class _MuiRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, epsilon):
        output = muillm_ext.muillm_rmsnorm_forward(weights, inputs, epsilon)

        ctx.save_for_backward(inputs, weights)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("RMSNorm backward is not implemented")

class MuiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = eps

        wdtype = self.weight.dtype
        dispatchable_type = (wdtype == torch.float16)
        dispatchable_device = self.weight.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    @staticmethod
    def replace(prev_module: Union[LlamaRMSNorm, MistralRMSNorm], engine_config: MuiEngineConfig) -> "MuiRMSNorm":
        hidden_size = prev_module.weight.shape[0]
        eps = prev_module.variance_epsilon
        new_module = MuiRMSNorm(hidden_size=hidden_size, eps=eps, dtype=prev_module.weight.dtype, device=prev_module.weight.device)

        new_module.weight = nn.Parameter(prev_module.weight.detach())
        new_module.weight.requires_grad = prev_module.weight.requires_grad

        return new_module

    def forward(self, input: Tensor) -> Tensor:
        if self.dispatchable:
            # we support the type
            return _MuiRMSNorm.apply(input, self.weight, self.variance_epsilon)
        else:
            # non-fused implementation
            input_dtype = input.dtype
            hidden_states = input.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)