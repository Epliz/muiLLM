from typing import Iterable, List, Tuple, Union
from muillm.hftensorparallelism.hftensorparallelism import _to_local_module
from muillm.engineconfig import (
    MuiEngineConfig,
)
from muillm.modules.module import MuiModule
from muillm.modules.norm.rmsnorm import MuiRMSNorm
import torch
from torch import Tensor
import torch.nn as nn

from muillm.modules.linear import MuiLinear
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from muillm.replacement.replacementcontext import MuiReplacementContext


def _all_or_none(it: Iterable[bool], exception_message) -> bool:
    has_all = all([i for i in it])
    has_any = not all([not i for i in it])

    if has_any and not has_all:
        raise ValueError(exception_message)
    return has_all


class MuiMultiLinear(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        in_features: int,
        out_features: List[int],
        bias: bool = True,
        variance_epsilon: float = 0.0,
        normalize: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(engine_config=engine_config)

        self.linear = MuiLinear(
            engine_config=engine_config,
            in_features=in_features,
            out_features=sum(out_features),
            bias=bias,
            variance_epsilon=variance_epsilon,
            normalize=normalize,
            device=device,
            dtype=dtype,
        )

        self.in_features = in_features
        self.out_features = list(out_features)

        self.slice_starts = []
        self.slice_ends = []

        current_start = 0
        for out_feature in out_features:
            current_end = current_start + out_feature

            self.slice_starts.append(current_start)
            self.slice_ends.append(current_end)

            current_start = current_end

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self.linear.finalize_init()

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_modules: List[Union[MuiLinear, nn.Linear]],
        prev_layernorm_module: Union[MuiRMSNorm, LlamaRMSNorm, MistralRMSNorm] = None,
    ) -> "MuiMultiLinear":
        engine_config = replacement_context.engine_config
        device = replacement_context.device
        if device is None:
            raise ValueError("device was None")

        # Make sure we convert the previous modules to local modules
        # so that we can safely copy their parameters
        prev_modules = [
            (
                replacement_context.to_local_module(prev_module)
                if not isinstance(prev_module, MuiLinear)
                else prev_module
            )
            for prev_module in prev_modules
        ]

        if (prev_layernorm_module is not None) and (
            not isinstance(prev_layernorm_module, MuiRMSNorm)
        ):
            # Make sure we convert the previous layernorm module to a local module
            # so that we can safely copy its parameters
            prev_layernorm_module = replacement_context.to_local_module(
                prev_layernorm_module
            )

        if len(prev_modules) < 1:
            raise ValueError(
                "MuiMultiLinear needs some linear layers passed in but none were provided"
            )

        has_bias = _all_or_none(
            [prev_module.bias is not None for prev_module in prev_modules],
            "All linear modules or none must have a bias to merge into a MuiMultiLinear layer",
        )

        in_features = prev_modules[0].in_features
        out_features = [prev_module.out_features for prev_module in prev_modules]
        dtype = prev_modules[0].weight.dtype
        device = prev_modules[0].weight.device if device is None else device

        # put on the end device to accelerate things
        # (ok as we are replacing the module entirely so we can change its device)
        if device is not None:
            prev_modules = [prev_module.to(device) for prev_module in prev_modules]
            prev_layernorm_module = (
                prev_layernorm_module.to(device)
                if prev_layernorm_module is not None
                else None
            )

        normalize = prev_layernorm_module is not None
        variance_epsilon = (
            MuiRMSNorm._extract_eps(prev_layernorm_module) if normalize else 0.0
        )
        norm_weights = prev_layernorm_module.weight if normalize else None

        new_module = MuiMultiLinear(
            engine_config=engine_config,
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            variance_epsilon=variance_epsilon,
            normalize=normalize,
            dtype=dtype,
            device=device,
        )
        new_module.copy_modules(
            prev_modules=prev_modules, norm_weights=norm_weights, device=device
        )

        return new_module

    def _get_linear_back(self, slice_start: int, slice_end: int) -> nn.Linear:
        out_features = slice_end - slice_start
        has_bias = self.linear.bias is not None

        weight = self.linear.weight[slice_start:slice_end, ...]
        bias = self.linear.bias[slice_start:slice_end] if has_bias else None

        linear = nn.Linear(
            in_features=self.in_features,
            out_features=out_features,
            bias=has_bias,
            dtype=weight.dtype,
            device=weight.device,
        )

        weights_require_grads = self.linear.weight.requires_grad
        linear.weight = nn.Parameter(weight)
        linear.weight.requires_grad = weights_require_grads

        if has_bias:
            bias_requires_grads = self.linear.bias.requires_grad
            linear.bias = nn.Parameter(bias)
            linear.bias.requires_grad = bias_requires_grads

        return linear

    def replace_back(self) -> Tuple[List[nn.Linear], torch.Tensor]:
        # split back in different modules
        norm_weights = self.linear.norm_weights

        linears = [
            self._get_linear_back(slice_start, slice_end)
            for slice_start, slice_end in zip(self.slice_starts, self.slice_ends)
        ]

        return linears, norm_weights

    def copy_modules(
        self,
        prev_modules: List[Union[MuiLinear, nn.Linear]],
        norm_weights: torch.Tensor = None,
        variance_epsilon: float = 0.0,
        device=None,
    ):
        if device is None:
            raise ValueError("device was None")

        device = prev_modules[0].weight.device if device is None else device
        has_bias = self.linear.bias is not None

        concat_weights_require_grads = _all_or_none(
            [prev_module.weight.requires_grad for prev_module in prev_modules],
            "all or none weights must required grads but got a mix",
        )

        concat_weights = torch.cat(
            [prev_module.weight.detach() for prev_module in prev_modules],
            dim=0,
        )
        self.linear._set_weights(concat_weights, concat_weights_require_grads)

        if has_bias:
            concat_biases = torch.cat(
                [prev_module.bias.detach() for prev_module in prev_modules],
                dim=0,
            )
            concat_biases_requires_grad = _all_or_none(
                [prev_module.bias.requires_grad for prev_module in prev_modules],
                "all or none biases must required grads but got a mix",
            )
            self.linear._set_bias(concat_biases, concat_biases_requires_grad)

        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities
            self.linear._set_norm_weights(norm_weights)

        # put ourselves on the right device
        self.to(device=device)

    def forward(self, input: Tensor) -> Tuple[Tensor, ...]:
        all_outputs = self.linear(input)

        return tuple(
            [
                all_outputs[..., slice_start:slice_end]
                for slice_start, slice_end in zip(self.slice_starts, self.slice_ends)
            ]
        )
