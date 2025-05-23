from typing import Union
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.module import MuiModule
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

class MuiRMSNorm(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, hidden_size, eps=1e-6,
                 device=None, dtype=None) -> None:
        super().__init__(engine_config=engine_config)
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = eps

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()
    
    def _check_dispatchable(self):
        wdtype = self.weight.dtype
        dispatchable_type = (wdtype == torch.float16)
        dispatchable_device = self.weight.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    @staticmethod
    def replace(prev_module: Union["MuiRMSNorm", LlamaRMSNorm, MistralRMSNorm], engine_config: MuiEngineConfig, device = None) -> "MuiRMSNorm":
        if device is None:
            raise ValueError("device was None")
        
        if isinstance(prev_module, MuiRMSNorm):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_module
        
        device = prev_module.weight.device if device is None else device

        # put on the end device to accelerate things
        # (ok as we are replacing the module entirely so we can change its device)
        if device is not None:
            prev_module = prev_module.to(device=device)

        hidden_size = prev_module.weight.shape[0]
        eps = prev_module.variance_epsilon

        new_module = MuiRMSNorm(engine_config=engine_config, hidden_size=hidden_size, eps=eps, dtype=prev_module.weight.dtype, device=device)
        new_module.copy_module(prev_module.weight, device=device)

        # delete the previous module to save memory
        del prev_module

        # trigger GC to save memory
        trigger_gc()

        return new_module

    def copy_module(self, prev_module: nn.Parameter, device = None):
        if device is None:
            raise ValueError("device was None")

        self.weight = nn.Parameter(prev_module.detach())
        self.weight.requires_grad = prev_module.requires_grad

        # put ourselves on the right device
        self.to(device=device)

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

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