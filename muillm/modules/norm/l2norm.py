from typing import Union
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.module import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import muillm_ext

from transformers.models.llama4.modeling_llama4 import Llama4TextL2Norm

from muillm.engineconfig import MuiEngineConfig
from muillm.replacement.replacementcontext import MuiReplacementContext


class _MuiL2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, epsilon):
        output = muillm_ext.muillm_l2norm_forward(inputs, epsilon)

        ctx.save_for_backward(inputs)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("L2Norm backward is not implemented")


class MuiL2Norm(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        eps=1e-6,
    ) -> None:
        super().__init__(engine_config=engine_config)
        self.variance_epsilon = eps

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        # always dispatchable given that the inputs are float 16
        self.dispatchable = True

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    @staticmethod
    def _extract_eps(
        prev_module: Union["MuiL2Norm", Llama4TextL2Norm],
    ) -> float:
        if isinstance(prev_module, Llama4TextL2Norm):
            # Llama4 RMSNorm has a different interface
            return prev_module.eps
        else:
            # Mistral and Llama RMSNorm have the same interface
            return prev_module.variance_epsilon

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_module: Union["MuiL2Norm", Llama4TextL2Norm],
    ) -> "MuiL2Norm":
        engine_config = replacement_context.engine_config
        device = replacement_context.device
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiL2Norm):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_module

        eps = MuiL2Norm._extract_eps(prev_module)

        new_module = MuiL2Norm(
            engine_config=engine_config,
            eps=eps,
        )

        return new_module

    def forward(self, input: Tensor) -> Tensor:
        if (
            self.dispatchable
            and (input.is_cuda)
            and ((input.dtype == torch.float16) or (input.dtype == torch.bfloat16))
        ):
            # we support the type
            return _MuiL2Norm.apply(input, self.variance_epsilon)
        else:
            # non-fused implementation
            input_dtype = input.dtype
            hidden_states = input.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return hidden_states.to(input_dtype)
