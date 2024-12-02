from typing import List, Optional, Union
from muillm.layers.linear import MuiLinear
from muillm.layers.module import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from muillm.engineconfig import MuiEngineConfig
from muillm.layers.rmsnorm import _MuiRMSNorm
import muillm_ext

class _MuiParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, norm_weights, variance_epsilon, add_bias, residual):
        if (add_bias is not None) and (residual is not None):
            raise ValueError("bias and residual at the same time is not supported")

        if residual is not None:
            add_bias = residual

        output = muillm_ext.muillm_linear_forward(x, weights, norm_weights, variance_epsilon, mul_bias=None, add_bias=add_bias)

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

class MuiParallelLinear(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, in_features: int, out_features: int, bias: bool = True,
                 variance_epsilon:float = 0.0, normalize:bool = False, sharding_dim: int = 1, device=None, dtype=None) -> None:
        MuiModule.__init__(self, engine_config=engine_config)

        linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)

        self.tensor_parallelism = engine_config.tensor_parallelism
        self.sharding_dim = sharding_dim + len(linear.weight.shape) if sharding_dim < 0 else sharding_dim

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = nn.Parameter(torch.ones(in_features, dtype=dtype, device=device)) if normalize else None

        self.weights = nn.ParameterList(self._shard_weigths(linear.weight))
        self._set_requires_grads(self.weights, linear.weight.requires_grad)

        if linear.bias is not None:
            self.biases = nn.ParameterList(self._shard_bias(linear.bias))
            self._set_requires_grads(self.biases, linear.bias.requires_grad)
        else:
            self.biases = None

        wdtype = linear.weight.dtype
        dispatchable_type = (wdtype == torch.float16)
        dispatchable_device = linear.weight.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def _set_requires_grads(self, params: Union[List[nn.parameter.Parameter], nn.ParameterList], requires_grads: bool) -> None:
        for p in params:
            p.requires_grad = requires_grads

    @staticmethod
    def replace(prev_module: Union[nn.Linear, MuiLinear], engine_config: MuiEngineConfig, prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None) -> "MuiParallelLinear":
        has_bias = prev_module.bias is not None
        in_features = prev_module.in_features
        out_features = prev_module.out_features

        if isinstance(prev_module, MuiLinear):
            # MuiLinear inherits nn.Linear, so need to check first
            if prev_layernorm_module is not None:
                raise ValueError("prev_layernorm_module should be None")

            normalize = prev_module.normalize
            variance_epsilon = prev_module.variance_epsilon
            norm_weights = None # will be taken from the previous module
        elif isinstance(prev_module, nn.Linear):
            normalize = prev_layernorm_module is not None
            variance_epsilon = prev_layernorm_module.variance_epsilon if normalize else 0.0
            norm_weights = prev_layernorm_module.weight if normalize else None
        else:
            raise ValueError(f"Unsupported replacement: {prev_module.__class__.__name__}")

        new_module = MuiParallelLinear(engine_config=engine_config, in_features=in_features, out_features=out_features, bias=has_bias, variance_epsilon=variance_epsilon, normalize=normalize, dtype=prev_module.weight.dtype, device=prev_module.weight.device)
        new_module.copy_module(prev_module=prev_module, norm_weights=norm_weights)

        return new_module

    def copy_module(self, prev_module: Union[nn.Linear, MuiLinear], norm_weights: torch.Tensor = None, variance_epsilon: float = 0.0):
        has_bias = prev_module.bias is not None

        self.weights = nn.ParameterList(self._shard_weigths(prev_module.weight))
        self._set_requires_grads(self.weights, prev_module.weight.requires_grad)

        if has_bias:
            self.biases = nn.ParameterList(self._shard_bias(prev_module.bias)) if prev_module.bias is not None else None
            self._set_requires_grads(self.biases, prev_module.bias.requires_grad)

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

            norm_weights_requires_grad = norm_weights.requires_grad
            self.norm_weights = nn.Parameter(norm_weights.detach())
            self.norm_weights.requires_grad = norm_weights_requires_grad

    def _shard_weigths(self, tensor: Tensor) -> List[Tensor]:
        return torch.tensor_split(tensor, self.tensor_parallelism, self.sharding_dim)

    def _shard_inputs(self, tensor: Tensor) -> List[Tensor]:
        if self.sharding_dim == 1:
            # if we are sharding along the k-dim, we need to shard the input accordingly
            return torch.tensor_split(tensor, self.tensor_parallelism, -1)
        elif (self.sharding_dim == 0):
            # but if we shard by row, we just need the inputs on all devices
            return [tensor] * self.tensor_parallelism
        else:
            raise ValueError("Unsupported sharding dimension")

    def _shard_bias(self, bias: Optional[Tensor]) -> Optional[List[Tensor]]:
        if bias is None:
            return None
        
        if self.sharding_dim == 0:
            # if we shard by rows, we need to shard the bias
            return torch.tensor_split(bias, self.tensor_parallelism, 0)
        elif self.sharding_dim == 1:
            # if we shard by columns (k-dim), we should not shard
            # we can instead apply it only on the first GPU
            return [bias if i == 0 else None for i in range(self.tensor_parallelism)]
        else:
            raise ValueError("Unsupported sharding dimension")

    def _collect_outputs(self, tensors: List[Tensor]) -> List[Tensor]:
        if self.sharding_dim == 1:
            # reduce on GPU0
            output = tensors[0]
            for i in range(1, self.tensor_parallelism):
                output = output + tensors[i]

            return [output] * self.tensor_parallelism
        else:
            raise ValueError("Not supported")

    def parallel_forward(self, input: Tensor, residual: Optional[Tensor] = None) -> List[Tensor]:
        # TODO: adapt
        # if self.dispatchable and (input.numel() == input.shape[-1]):
        #     # input is effectively 1D, and we support the type
        #     if (self.sharding_dim == 1) and self.normalize:
        #         # when splitting along k-dim, we need to normalize first
        #         input = _MuiRMSNorm.apply(input, self.norm_weights, self.variance_epsilon)
        #         return _MuiParallelLinear.apply(input, self.weight, None, 0, self.bias, residual)
        #     else:
        #         return _MuiParallelLinear.apply(input, self.weight, self.norm_weights, self.variance_epsilon, self.bias, residual)

        # Not dispatchable
        if self.normalize:
            input = _MuiRMSNorm.apply(input, self.norm_weights, self.variance_epsilon)

        inputs = self._shard_inputs(input)

        # Do the sharded computation
        outputs = [F.linear(inputs[i], self.weights[i], self.biases[i] if self.biases is not None else None) for i in range(self.tensor_parallelism)]

        # Apply the residual on GPU0
        if residual is not None:
            outputs[0] = outputs[0] + residual

        # reduce
        outputs = self._collect_outputs(outputs)

        #output = F.linear(input, self.weight, self.bias)

        return outputs

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.tensor_parallelism > 1:
            return self.parallel_forward(input, residual)[0]

        raise ValueError("Only parallel inference is supported")