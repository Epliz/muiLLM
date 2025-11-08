from typing import Tuple, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.module import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from muillm.modules.norm.rmsnorm import MuiRMSNorm
import muillm_ext

from muillm.modules.norm.l2norm import MuiL2Norm


from transformers.models.llama4.modeling_llama4 import Llama4TextL2Norm
from muillm.replacement.replacementcontext import MuiReplacementContext


class _MuiQKRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_w, k_w, q, k, epsilon, weight_offset):
        q = q.contiguous()
        k = k.contiguous()
        output = muillm_ext.muillm_qkrmsnorm_forward(
            q_w, k_w, q, k, epsilon, weight_offset
        )

        ctx.save_for_backward(q, k)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("QKRMSNorm backward is not implemented")


class MuiQKRMSNorm(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        q_weights: nn.Parameter,
        k_weights: nn.Parameter,
        eps=1e-6,
        weight_offset: float = 0.0,
    ) -> None:
        super().__init__(engine_config=engine_config)

        self.q_weights = q_weights
        self.k_weights = k_weights

        self.variance_epsilon = eps
        self.weight_offset = weight_offset

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        # always dispatchable given that the inputs are float 16
        weight_dtype = self.q_weights.dtype
        dispatchable_type = (weight_dtype == torch.bfloat16) or (
            weight_dtype == torch.float16
        )
        self.dispatchable = dispatchable_type

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    @staticmethod
    def _extract_eps(
        prev_module: Union[MuiRMSNorm, Llama4TextL2Norm],
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
        prev_qmodule: Union[MuiRMSNorm, Llama4TextL2Norm],
        prev_kmodule: Union[MuiRMSNorm, Llama4TextL2Norm],
    ) -> "MuiQKRMSNorm":
        engine_config = replacement_context.engine_config
        device = replacement_context.device
        if device is None:
            raise ValueError("device was None")

        q_weights = MuiRMSNorm._extract_weights(prev_qmodule)
        k_weights = MuiRMSNorm._extract_weights(prev_kmodule)

        qeps = MuiRMSNorm._extract_eps(prev_qmodule)
        keps = MuiRMSNorm._extract_eps(prev_kmodule)

        q_weight_offset = MuiRMSNorm._extract_weight_offset(prev_qmodule)
        k_weight_offset = MuiRMSNorm._extract_weight_offset(prev_kmodule)

        if qeps != keps:
            raise ValueError("q and k epsilon must be the same")

        if q_weight_offset != k_weight_offset:
            raise ValueError("q and k weight offfsets must be the same")

        new_module = MuiQKRMSNorm(
            engine_config=engine_config,
            q_weights=q_weights,
            k_weights=k_weights,
            eps=qeps,
            weight_offset=k_weight_offset,
        )

        return new_module

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        if (
            self.dispatchable
            and (q.is_cuda)
            and ((q.dtype == torch.float16) or (q.dtype == torch.bfloat16))
        ):
            # we support the type
            return _MuiQKRMSNorm.apply(
                self.q_weights,
                self.k_weights,
                q,
                k,
                self.variance_epsilon,
                self.weight_offset,
            )
        else:
            # non-fused implementation
            input_dtype = q.dtype
            q_states = q.to(torch.float32)
            k_states = k.to(torch.float32)

            q_variance = q_states.pow(2).mean(-1, keepdim=True)
            k_variance = k_states.pow(2).mean(-1, keepdim=True)

            q_states = q_states * torch.rsqrt(q_variance + self.variance_epsilon)
            k_states = k_states * torch.rsqrt(k_variance + self.variance_epsilon)

            if self.weight_offset != 0:
                q_w = self.q_weights + self.weight_offset
                k_w = self.k_weights + self.weight_offset
            else:
                q_w = self.q_weights
                k_w = self.k_weights

            q_states = q_w * q_states.to(input_dtype)
            k_states = k_w * k_states.to(input_dtype)

            return q_states, k_states
