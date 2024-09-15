from typing import Optional, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.muimodule import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from muillm.engineconfig import MuiEngineConfig
from muillm.layers.rmsnorm import _MuiRMSNorm
from muillm.tensorparallelism.sharding import _shard
import muillm_ext

class _MuiLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tensor_parallelism, sharding_dim, weights, norm_weights, variance_epsilon, add_bias, residual):
        if (add_bias is not None) and (residual is not None):
            raise ValueError("bias and residual at the same time is not supported")

        if residual is not None:
            add_bias = residual

        output = muillm_ext.muillm_linear_forward(x, tensor_parallelism, sharding_dim, weights, norm_weights, variance_epsilon, mul_bias=None, add_bias=add_bias)

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
                 variance_epsilon:float = 0.0, normalize:bool = False, sharding_dim: int = -1, device=None, dtype=None) -> None:
        MuiModule.__init__(self, engine_config=engine_config)
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)

        self.comm = engine_config.communicator

        tensor_parallelism = engine_config.tensor_parallelism

        self.tensor_parallelism = tensor_parallelism
        self.sharding_dim = sharding_dim if sharding_dim >= 0 else sharding_dim + len(self.weight.shape)

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = nn.Parameter(self._shard_norm_weights(torch.ones(in_features, dtype=dtype, device=device), self.tensor_parallelism, self.sharding_dim)) if normalize else None

        self.weight = nn.Parameter(_shard(self.weight, tensor_parallelism=self.tensor_parallelism, dim=self.sharding_dim))

        # we need to shard the bias as well
        if bias:
            requires_grads = self.bias.requires_grad
            self.bias = nn.Parameter(MuiLinear._shard_bias(self.bias, self.tensor_parallelism, self.sharding_dim))
            self.bias.requires_grad = requires_grads

        self.weight = nn.Parameter(_shard(self.weight, tensor_parallelism=self.tensor_parallelism, dim=-1))

        wdtype = self.weight.dtype
        dispatchable_type = (wdtype == torch.float16)
        dispatchable_device = self.weight.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    @staticmethod
    def _shard_inputs(inputs:torch.Tensor, tensor_parallelism:int, sharding_dim:int, shard_inputs:bool = True):
        if not shard_inputs:
            # in some cases (e.g. sharded MLP), we can skip the sharding as we do things manually
            return inputs

        if sharding_dim == 1:
            # if we shard along the K dim (column-wise), we need to shard as well the inputs
            # on the last dim
            return _shard(inputs, tensor_parallelism=tensor_parallelism, dim=-1)
        else:
            # but if we shard along the M dim (row-wise), we should not shard at all
            return inputs

    @staticmethod
    def _shard_norm_weights(norm_weights:torch.Tensor, tensor_parallelism:int, sharding_dim:int):
        # we need to shard the norm weights like we shard the inputs
        return MuiLinear._shard_inputs(norm_weights, tensor_parallelism, sharding_dim)

    @staticmethod
    def _shard_bias(bias:torch.Tensor, tensor_parallelism:int, sharding_dim:int):
        if bias is None:
            return None

        if (sharding_dim == 0):
            # for row-wise sharding, we need to shard the bias
            sharded_bias = _shard(bias, tensor_parallelism=tensor_parallelism).detach()
            return sharded_bias
        elif (sharding_dim == 1):
            # for column-wise sharding, we will just scale the bias during computations
            return bias
        else:
            raise ValueError(f"Unsupported sharding dim {sharding_dim}")

    @staticmethod
    def replace(prev_module: nn.Linear, engine_config: MuiEngineConfig, prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None, sharding_dim:int = -1) -> "MuiLinear":
        has_bias = prev_module.bias is not None
        in_features = prev_module.in_features
        out_features = prev_module.out_features

        normalize = prev_layernorm_module is not None
        variance_epsilon = prev_layernorm_module.variance_epsilon if normalize else 0.0
        norm_weights = prev_layernorm_module.weight if normalize else None

        new_module = MuiLinear(engine_config=engine_config, in_features=in_features, out_features=out_features, bias=has_bias, variance_epsilon=variance_epsilon, normalize=normalize, sharding_dim=sharding_dim, dtype=prev_module.weight.dtype, device=prev_module.weight.device)
        new_module.copy_module(prev_module=prev_module, norm_weights=norm_weights)

        return new_module

    def copy_module(self, prev_module: nn.Linear, norm_weights: torch.Tensor = None):
        has_bias = prev_module.bias is not None

        # shard for tensor parallelism if necessary
        self.weight = nn.Parameter(_shard(prev_module.weight, tensor_parallelism=self.tensor_parallelism, dim=self.sharding_dim).detach())
        self.weight.requires_grad = prev_module.weight.requires_grad

        if has_bias:
            # we don't shard the bias for column-wise sharding, but scale it down to get the correct result in the kernel
            # for row-wise, we need to shard it as well
            requires_grads = prev_module.bias.requires_grad
            self.bias = nn.Parameter(MuiLinear._shard_bias(prev_module.bias, self.tensor_parallelism, self.sharding_dim))
            self.bias.requires_grad = requires_grads


        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities

            norm_weights_requires_grad = norm_weights.requires_grad
            # shard for tensor parallelism if necessary
            self.norm_weights = nn.Parameter(MuiLinear._shard_norm_weights(norm_weights.detach(), self.tensor_parallelism, self.sharding_dim))
            self.norm_weights.requires_grad = norm_weights_requires_grad

            self.norm_weights = norm_weights

    def _collect_output(self, output:torch.Tensor, collect_output:bool = True):
        if not collect_output:
            # in some cases (e.g. sharded MLP), we can skip the collection as we do things manually
            return output

        if self.tensor_parallelism > 1:
            if self.sharding_dim == 1:
                # if we sharded along the K-dim (columnwise), collecting the pieces
                # is summing them up (all reduce)
                self.comm.all_reduce_sum(output)
            else:
                # if we sharded long the M-dim (row-wise), collecting the pieces
                # is concatenating them up
                # (but we don't really need to implement it as we should not be using it
                # anywhere at the moment)
                raise ValueError("not implemented")


    def forward(self, input: Tensor, residual: Optional[Tensor] = None, shard_inputs:bool = True, collect_output:bool = True) -> Tensor:

        if self.dispatchable and (input.numel() == input.shape[-1]):
            # input is effectively 1D, and we support the type

            if self.normalize and (self.tensor_parallelism > 1) and (self.sharding_dim == 1):
                # we need to actually do the normalization before sharding if sharding column-wise
                # as otherwise the variance is not correct (not getting the full sum and scaled improperly)
                input = _MuiRMSNorm.apply(input, self.norm_weights, self.variance_epsilon)
                input = MuiLinear._shard_inputs(input, self.tensor_parallelism, self.sharding_dim, shard_inputs=shard_inputs)
                output = _MuiLinear.apply(input, self.tensor_parallelism, self.sharding_dim, self.weight, None, self.variance_epsilon, self.bias,residual)
            else:
                input = MuiLinear._shard_inputs(input, self.tensor_parallelism, self.sharding_dim, shard_inputs=shard_inputs)
                output = _MuiLinear.apply(input, self.tensor_parallelism, self.sharding_dim, self.weight, self.norm_weights, self.variance_epsilon, self.bias,residual)

            self._collect_output(output, collect_output=collect_output)

            return output

        # else: not dispatchable, or batch size different from 1
        if self.normalize:
            input = _MuiRMSNorm.apply(input, self.norm_weights, self.variance_epsilon)

        input = MuiLinear._shard_inputs(input, self.tensor_parallelism, self.sharding_dim, shard_inputs=shard_inputs)

        bias = self.bias
        if bias is not None:
            # when doing column-wise sharding, we need to scale the bias as we are adding it several time due to the
            # all-reduce
            bias = bias / self.tensor_parallelism if (self.tensor_parallelism != 1 and self.sharding_dim == 1) else bias

        output = F.linear(input, self.weight, bias)

        self._collect_output(output, collect_output=collect_output)

        if residual is not None:
            output = output + residual

        return output