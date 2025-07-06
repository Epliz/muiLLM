from typing import Tuple, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.module import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import muillm_ext

from muillm.modules.norm.l2norm import MuiL2Norm


from transformers.models.llama4.modeling_llama4 import Llama4TextL2Norm
from muillm.replacement.replacementcontext import MuiReplacementContext


class _MuiQKL2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, epsilon):
        output = muillm_ext.muillm_qkl2norm_forward(q, k, epsilon)

        ctx.save_for_backward(q, k)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("QKL2Norm backward is not implemented")


class MuiQKL2Norm(MuiModule):
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
        prev_qmodule: Union["MuiL2Norm", Llama4TextL2Norm],
    ) -> "MuiQKL2Norm":
        engine_config = replacement_context.engine_config
        device = replacement_context.device
        if device is None:
            raise ValueError("device was None")

        qeps = MuiL2Norm._extract_eps(prev_qmodule)

        new_module = MuiQKL2Norm(
            engine_config=engine_config,
            eps=qeps,
        )

        return new_module

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        if (
            self.dispatchable
            and (q.is_cuda)
            and ((q.dtype == torch.float16) or (q.dtype == torch.bfloat16))
        ):
            # we support the type
            return _MuiQKL2Norm.apply(q, k, self.variance_epsilon)
        else:
            # non-fused implementation
            input_dtype = q.dtype
            q_states = q.to(torch.float32)
            k_states = k.to(torch.float32)
            q_variance = q_states.pow(2).mean(-1, keepdim=True)
            k_variance = k_states.pow(2).mean(-1, keepdim=True)
            q_states = q_states * torch.rsqrt(q_variance + self.variance_epsilon)
            k_states = k_states * torch.rsqrt(k_variance + self.variance_epsilon)
            return q_states.to(input_dtype), k_states.to(input_dtype)
