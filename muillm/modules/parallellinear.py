from typing import List, Optional, Union
from muillm.modules.linear import MuiLinear
from muillm.modules.module import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.rmsnorm import _MuiRMSNorm
import muillm_ext

class _MuiParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, comm, x, weights, norm_weights, variance_epsilon, add_biases, residual, sharding_dim, reduce):
        output = muillm_ext.muillm_parallel_linear_forward(comm.comm, x, weights, norm_weights, variance_epsilon, mul_biases=None, add_biases=add_biases, residual=residual, sharding_dim=sharding_dim, reduce=reduce)

        ctx.save_for_backward(x, weights, norm_weights, variance_epsilon, add_biases, residual)

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

        self.comms = engine_config.comms
        self.tensor_parallelism = engine_config.tensor_parallelism
        self.sharding_dim = sharding_dim + len(linear.weight.shape) if sharding_dim < 0 else sharding_dim

        self.in_features = in_features
        self.out_features = out_features
        self.device = linear.weight.device
        self.dtype = linear.weight.dtype

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = nn.ParameterList([torch.ones(in_features, dtype=dtype, device=d) for d in self.engine_config.devices]) if normalize else None

        self.weights = nn.ParameterList(self.__shard_weigths(linear.weight))
        MuiParallelLinear._set_requires_grads(self.weights, linear.weight.requires_grad)

        if linear.bias is not None:
            self.biases = nn.ParameterList(self.__shard_bias(linear.bias))
            MuiParallelLinear._set_requires_grads(self.biases, linear.bias.requires_grad)
        else:
            self.biases = None

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # Need to synchronize after copying the tensors to make sure the transfers
        # completed
        self.__sync_all()

    def _check_dispatchable(self):
        self.wdtype = self.weights[0].dtype
        dispatchable_type = (self.wdtype == torch.float16)
        self.is_cuda = self.weights[0].is_cuda
        dispatchable_device = self.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    @staticmethod
    def _set_requires_grads(params: Union[List[nn.parameter.Parameter], nn.ParameterList], requires_grads: bool) -> None:
        for p in params:
            if p is not None:
                p.requires_grad = requires_grads

    @staticmethod
    def replace(prev_module: Union[nn.Linear, MuiLinear], engine_config: MuiEngineConfig, prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None, sharding_dim: int = 1) -> "MuiParallelLinear":
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

        new_module = MuiParallelLinear(engine_config=engine_config, in_features=in_features, out_features=out_features, bias=has_bias, variance_epsilon=variance_epsilon, normalize=normalize, sharding_dim=sharding_dim, dtype=prev_module.weight.dtype, device=prev_module.weight.device)
        new_module.copy_module(prev_module=prev_module, norm_weights=norm_weights)

        return new_module
    
    def _set_norm_weights(self, norm_weights: torch.Tensor) -> None:
        norm_weights_requires_grad = norm_weights.requires_grad
        self.norm_weights = nn.ParameterList([norm_weights.detach().to(device=d) for d in self.engine_config.devices])
        MuiParallelLinear._set_requires_grads(self.norm_weights, norm_weights_requires_grad)

    def copy_module(self, prev_module: Union[nn.Linear, MuiLinear], norm_weights: torch.Tensor = None, variance_epsilon: float = 0.0):
        has_bias = prev_module.bias is not None

        self.weights = nn.ParameterList(self.__shard_weigths(prev_module.weight))
        MuiParallelLinear._set_requires_grads(self.weights, prev_module.weight.requires_grad)

        if has_bias:
            self.biases = nn.ParameterList(self.__shard_bias(prev_module.bias)) if prev_module.bias is not None else None
            MuiParallelLinear._set_requires_grads(self.biases, prev_module.bias.requires_grad)

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
            self._set_norm_weights(norm_weights)

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # Need to synchronize after copying the tensors to make sure the transfers
        # completed
        self.__sync_all()

    def __sync_all(self):
        MuiParallelLinear._sync_all(engine_config=self.engine_config)

    @staticmethod
    def _sync_all(engine_config: MuiEngineConfig):
        devices = engine_config.devices
        for d in devices:
            torch.cuda.synchronize(d)

    @staticmethod
    def _broadcast(engine_config: MuiEngineConfig, tensor: Tensor) -> Optional[List[Tensor]]:
        return engine_config.comms.broadcast(tensor)

    def __shard_weigths(self, tensor: Tensor) -> List[Tensor]:
        return MuiParallelLinear._shard_weigths(self.engine_config, tensor, self.tensor_parallelism, self.sharding_dim)

    @staticmethod
    def _shard_weigths(engine_config: MuiEngineConfig, tensor: Tensor, tensor_parallelism: int, sharding_dim: int) -> List[Tensor]:
        tensors = torch.tensor_split(tensor, tensor_parallelism, sharding_dim)
        tensors = [t.contiguous() for t in tensors]
        return engine_config.comms.transfer_across(tensors)

    def _shard_inputs(self, tensor: Tensor) -> List[Tensor]:
        if self.sharding_dim == 1:
            # if we are sharding along the k-dim, we need to shard the input accordingly
            tensors = torch.tensor_split(tensor, self.tensor_parallelism, -1)
        elif (self.sharding_dim == 0):
            # but if we shard by row, we just need the inputs on all devices
            tensors = [tensor] * self.tensor_parallelism
        else:
            raise ValueError("Unsupported sharding dimension")

        return self.comms.transfer_across(tensors)

    def _shard_inputs_if_needed(self, tensors: Union[Tensor, List[Tensor]]) -> List[Tensor]:
        sharded_inputs = isinstance(tensors, list)

        if not sharded_inputs:
            return self._shard_inputs(tensors)
        else:
            # already sharded
            return tensors
    
    def __shard_bias(self, bias: Optional[Tensor]) -> Optional[List[Tensor]]:
        return MuiParallelLinear._shard_bias(self.engine_config, bias, tensor_parallelism=self.tensor_parallelism, sharding_dim=self.sharding_dim)

    @staticmethod
    def _shard_bias(engine_config: MuiEngineConfig, bias: Optional[Tensor], tensor_parallelism: int, sharding_dim: int) -> Optional[List[Tensor]]:
        if bias is None:
            return None
        
        if sharding_dim == 0:
            # if we shard by rows, we need to shard the bias
            tensors = torch.tensor_split(bias, tensor_parallelism, 0)
        elif sharding_dim == 1:
            # if we shard by columns (k-dim), we should not shard
            # we can instead apply it only on the first GPU
            tensors = [bias if i == 0 else None for i in range(tensor_parallelism)]
        else:
            raise ValueError("Unsupported sharding dimension")
        
        return engine_config.comms.transfer_across(tensors)

    def __collect_outputs(self, tensors: List[Tensor]) -> List[Tensor]:
        return MuiParallelLinear._collect_outputs(self.engine_config, tensors, self.sharding_dim)

    @staticmethod
    def _collect_outputs(engine_config: MuiEngineConfig, tensors: List[Tensor], sharding_dim: int) -> List[Tensor]:
        if sharding_dim == 1:
            # reduce
            return engine_config.comms.all_reduce(tensors)
        elif sharding_dim == 0:
            # concat all
            return engine_config.comms.all_gather(tensors)
        else:
            # TODO: implement for sharding_dim == 0 (needed by parallel multi linear)
            raise ValueError("Not supported")

    def parallel_forward(self, input: Union[Tensor, List[Tensor]], residual: Optional[Tensor] = None, collect_outputs: bool = True) -> List[Tensor]:
        inputs = self._shard_inputs_if_needed(input)

        if self.dispatchable and (inputs[0].numel() == inputs[0].shape[-1]):
            # input is effectively 1D, and we support the type
            outputs = _MuiParallelLinear.apply(self.comms, inputs, self.weights, self.norm_weights, self.variance_epsilon, self.biases, residual, self.sharding_dim, collect_outputs)
        else:
            # TODO: do this case in the C++ part as well
            if self.normalize:
                if self.sharding_dim == 1:
                    raise ValueError("normalizing sharded inputs is unsupported for sharding dim 1")
                
                inputs = [_MuiRMSNorm.apply(inputs[d], self.norm_weights[d], self.variance_epsilon) for d in range(self.tensor_parallelism)]

            # Do the sharded computation
            outputs = [F.linear(inputs[i], self.weights[i], self.biases[i] if self.biases is not None else None) for i in range(self.tensor_parallelism)]

            # Apply the residual on GPU0
            if residual is not None:
                outputs[0] = outputs[0] + residual

            # reduce
            if collect_outputs:
                # TODO: push reduction in C++
                outputs = self.__collect_outputs(outputs)

        return outputs

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.tensor_parallelism > 1:
            return self.parallel_forward(input, residual)[0]

        raise ValueError("Only parallel inference is supported")