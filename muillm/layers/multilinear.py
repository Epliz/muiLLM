
from typing import Iterable, List, Tuple, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.muimodule import MuiModule
import torch
from  torch import Tensor
import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig
from muillm.layers.linear import MuiLinear
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from muillm.tensorparallelism.sharding import _shard

def _all_or_none(it: Iterable[bool], exception_message) -> bool:
    has_all = all([i for i in it])
    has_any = not all([not i for i in it])

    if has_any and not has_all:
        raise ValueError(exception_message)
    return has_all

class MuiMultiLinear(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, in_features: int, out_features: List[int], bias: bool = True,
                 variance_epsilon:float = 0.0, normalize:bool = False, sharding_dim:int = -1, device=None, dtype=None) -> None:
        super().__init__(engine_config=engine_config)

        tensor_parallelism = engine_config.tensor_parallelism

        self.linear = MuiLinear(engine_config=engine_config, in_features=in_features, out_features=sum(out_features), bias=bias, variance_epsilon=variance_epsilon, normalize=normalize, sharding_dim=sharding_dim, device=device, dtype=dtype)

        self.tensor_parallelism = tensor_parallelism
        # MuiLinear will bring back sharding_dim in [0, 1] if negative
        self.sharding_dim = self.linear.sharding_dim

        self.slice_starts = []
        self.slice_ends = []

        current_start = 0
        for out_feature in out_features:
            if self.linear.sharding_dim == 0:
                # if sharding by rows, the slice sizes get impacted by the tp level
                out_feature = out_feature // tensor_parallelism
            elif self.linear.sharding_dim == 1:
                pass
            else:
                raise ValueError("unsupported")

            current_end = current_start + out_feature

            self.slice_starts.append(current_start)
            self.slice_ends.append(current_end)

            current_start = current_end

    @staticmethod
    def replace(prev_modules: List[nn.Linear], engine_config: MuiEngineConfig, prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None, sharding_dim:int = -1) -> "MuiMultiLinear":
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

        new_module = MuiMultiLinear(engine_config=engine_config, in_features=in_features, out_features=out_features, bias=has_bias, variance_epsilon=variance_epsilon, normalize=normalize, sharding_dim=sharding_dim, dtype=dtype, device=device)
        new_module.copy_modules(prev_modules=prev_modules, norm_weights=norm_weights)

        return new_module

    def copy_modules(self, prev_modules: List[nn.Linear], norm_weights: torch.Tensor = None):
        has_bias = self.linear.bias is not None

        # shard for tensor parallelism if necessary
        # (we shard each inner tensor so that there is no distinction between the row or column case)
        self.linear.weight = nn.Parameter(torch.cat([_shard(prev_module.weight, tensor_parallelism=self.tensor_parallelism, dim=self.sharding_dim).detach() for prev_module in prev_modules], dim=0))
        self.linear.weight.requires_grad = _all_or_none([prev_module.weight.requires_grad for prev_module in prev_modules], "all or none weights must required grads but got a mix")

        if has_bias:
            # shard for tensor parallelism if necessary
            bias = torch.cat([MuiLinear._shard_bias(prev_module.bias, self.tensor_parallelism, self.sharding_dim).detach() for prev_module in prev_modules], dim=0)
            self.linear.bias = nn.Parameter(bias)
            self.linear.bias.requires_grad = _all_or_none([prev_module.bias.requires_grad for prev_module in prev_modules], "all or none biases must required grads but got a mix")


        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities

            norm_weights_requires_grad = norm_weights.requires_grad
            # shard for tensor parallelism if necessary
            self.linear.norm_weights = nn.Parameter(MuiLinear._shard_norm_weights(norm_weights.detach(), self.tensor_parallelism, self.sharding_dim))
            self.linear.norm_weights.requires_grad = norm_weights_requires_grad

            self.linear.norm_weights = norm_weights

    def forward(self, input: Tensor, shard_inputs:bool = True, collect_output:bool = True) -> Tuple[Tensor, ...]:
        all_outputs = self.linear(input, shard_inputs=shard_inputs, collect_output=collect_output)

        return tuple([all_outputs[...,slice_start:slice_end] for slice_start, slice_end in zip(self.slice_starts, self.slice_ends)])