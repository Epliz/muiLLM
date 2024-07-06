
from typing import Iterable, List, Tuple, Union
import torch
from  torch import Tensor
import torch.nn as nn

from muillm.layers.linear import MuiLinear
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

def _all_or_none(it: Iterable[bool], exception_message) -> bool:
    has_all = all([i for i in it])
    has_any = not all([not i for i in it])

    if has_any and not has_all:
        raise ValueError(exception_message)
    return has_all

class MuiMultiLinear(nn.Module):
    def __init__(self, in_features: int, out_features: List[int], bias: bool = True,
                 variance_epsilon:float = 0.0, normalize:bool = False, device=None, dtype=None) -> None:
        super().__init__()

        self.linear = MuiLinear(in_features=in_features, out_features=sum(out_features), bias=bias, variance_epsilon=variance_epsilon, normalize=normalize, device=device, dtype=dtype)

        self.slice_starts = []
        self.slice_ends = []

        current_start = 0
        for out_feature in out_features:
            current_end = current_start + out_feature

            self.slice_starts.append(current_start)
            self.slice_ends.append(current_end)

            current_start = current_end

    @staticmethod
    def replace(prev_modules: List[nn.Linear], prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None) -> "MuiMultiLinear":
        if len(prev_modules) < 1:
            raise ValueError("MuiMultiLinear needs some linear layers passed in but none were provided")
        
        has_bias =  _all_or_none([prev_module.bias is not None for prev_module in prev_modules], "All linear modules or none must have a bias to merge into a MuiMultiLinear layer")

        in_features = prev_modules[0].in_features
        out_features = [prev_module.out_features for prev_module in prev_modules]
        dtype = prev_modules[0].weight.dtype
        device = prev_modules[0].weight.device

        normalize = prev_layernorm_module is not None
        variance_epsilon = prev_layernorm_module.variance_epsilon if normalize else 0.0
        norm_weights = prev_layernorm_module.weight if normalize else None

        new_module = MuiMultiLinear(in_features=in_features, out_features=out_features, bias=has_bias, variance_epsilon=variance_epsilon, normalize=normalize, dtype=dtype, device=device)
        new_module.copy_modules(prev_modules=prev_modules, norm_weights=norm_weights)

        return new_module

    def copy_modules(self, prev_modules: List[nn.Linear], norm_weights: torch.Tensor = None, variance_epsilon: float = 0.0):
        has_bias = self.linear.bias is not None

        self.linear.weight = nn.Parameter(torch.cat([prev_module.weight.detach() for prev_module in prev_modules], dim=0))
        self.linear.weight.requires_grad = _all_or_none([prev_module.weight.requires_grad for prev_module in prev_modules], "all or none weights must required grads but got a mix")

        if has_bias:
            self.linear.bias = nn.Parameter(torch.cat([prev_module.bias.detach() for prev_module in prev_modules], dim=0))
            self.linear.bias.requires_grad = _all_or_none([prev_module.bias.requires_grad for prev_module in prev_modules], "all or none biases must required grads but got a mix")


        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities

            norm_weights_requires_grad = norm_weights.requires_grad
            self.linear.norm_weights = nn.Parameter(norm_weights.detach())
            self.linear.norm_weights.requires_grad = norm_weights_requires_grad

            self.linear.norm_weights = norm_weights

    def forward(self, input: Tensor) -> Tuple[Tensor, ...]:
        all_outputs = self.linear(input)

        return tuple([all_outputs[...,slice_start:slice_end] for slice_start, slice_end in zip(self.slice_starts, self.slice_ends)])