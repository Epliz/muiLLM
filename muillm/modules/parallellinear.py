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
    def forward(ctx, module, x, residual, reduce):
        output = muillm_ext.muillm_parallel_linear_module_forward(module, x, residual=residual, reduce=reduce)

        ctx.save_for_backward(x)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")

class MuiParallelLinear(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, in_features: int, out_features: int, bias: bool = True,
                 variance_epsilon:float = 0.0, normalize:bool = False, sharding_dim: int = 1, device=None, dtype=None) -> None:
        super().__init__(engine_config=engine_config)

        linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)

        self.cpp_engine = engine_config.cpp_engine
        self.comms = engine_config.comms
        self.tensor_parallelism = engine_config.tensor_parallelism
        self.sharding_dim = sharding_dim + len(linear.weight.shape) if sharding_dim < 0 else sharding_dim

        self.in_features = in_features
        self.out_features = out_features
        self.device = linear.weight.device
        self.dtype = linear.weight.dtype

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = nn.ParameterList([torch.ones(in_features, dtype=dtype, device=device)]) if normalize else None

        self.weights = nn.ParameterList([self.__shard_weigths(linear.weight)])
        MuiParallelLinear._set_requires_grads(self.weights, linear.weight.requires_grad)

        if linear.bias is not None:
            self.biases = nn.ParameterList([self.__shard_bias(linear.bias)])
            MuiParallelLinear._set_requires_grads(self.biases, linear.bias.requires_grad)
        else:
            self.biases = None

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        self.cpp_module = None
        self._create_cpp_module()

        # Need to synchronize after copying the tensors to make sure the transfers
        # completed
        self.__sync_all()

    def _check_dispatchable(self):
        self.dtype = self.weights[0].dtype
        dispatchable_type = (self.dtype == torch.float16)
        self.is_cuda = self.weights[0].is_cuda
        dispatchable_device = self.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def _create_cpp_module(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_linear_module_deinit(self.cpp_module)

        norm_weights = self.norm_weights[0] if self.norm_weights is not None else None
        bias = self.biases[0] if self.biases is not None else None

        self.cpp_module = muillm_ext.muillm_parallel_linear_module_init(
            self.cpp_engine,
            self.comms.comms,
            self.weights[0],
            norm_weights,
            self.variance_epsilon,
            None, # mul_bias
            bias,
            self.sharding_dim
        )

    @staticmethod
    def _set_requires_grads(params: nn.ParameterList, requires_grads: bool) -> None:
        for param in params:
            if param is not None:
                param.requires_grad = requires_grads

    @staticmethod
    def replace(prev_module: nn.Linear, engine_config: MuiEngineConfig, prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None) -> "MuiParallelLinear":
        has_bias = prev_module.bias is not None
        in_features = prev_module.in_features
        out_features = prev_module.out_features

        normalize = prev_layernorm_module is not None
        variance_epsilon = prev_layernorm_module.variance_epsilon if normalize else 0.0
        norm_weights = prev_layernorm_module.weight if normalize else None

        new_module = MuiParallelLinear(engine_config=engine_config, in_features=in_features, out_features=out_features, bias=has_bias, variance_epsilon=variance_epsilon, normalize=normalize, dtype=prev_module.weight.dtype, device=prev_module.weight.device)
        new_module.copy_module(prev_module=prev_module, norm_weights=norm_weights)

        return new_module
    
    def _set_norm_weights(self, norm_weights: torch.Tensor) -> None:
        norm_weights_requires_grad = norm_weights.requires_grad
        self.norm_weights = nn.ParameterList([norm_weights.detach()])
        MuiParallelLinear._set_requires_grads(self.norm_weights, norm_weights_requires_grad)

        # re-create the cpp module
        self._create_cpp_module()

    def copy_module(self, prev_module: Union[nn.Linear, MuiLinear], norm_weights: torch.Tensor = None, variance_epsilon: float = 0.0):
        has_bias = prev_module.bias is not None

        self.weights = nn.ParameterList([self.__shard_weigths(prev_module.weight)])
        MuiParallelLinear._set_requires_grads(self.weights, prev_module.weight.requires_grad)

        if has_bias:
            self.biases = nn.ParameterList([self.__shard_bias(prev_module.bias)]) if prev_module.bias is not None else None
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

        self._create_cpp_module()

        # Need to synchronize after copying the tensors to make sure the transfers
        # completed
        self.__sync_all()

    def __sync_all(self):
        MuiParallelLinear._sync_all(engine_config=self.engine_config)

    @staticmethod
    def _sync_all(engine_config: MuiEngineConfig):
        torch.cuda.synchronize()

    def __shard_weigths(self, tensor: Tensor) -> Tensor:
        return MuiParallelLinear._shard_weigths(self.engine_config, tensor, self.tensor_parallelism, self.sharding_dim)

    @staticmethod
    def _shard_weigths(engine_config: MuiEngineConfig, tensor: Tensor, tensor_parallelism: int, sharding_dim: int) -> Tensor:
        rank = engine_config.comms.rank
        tensor = torch.tensor_split(tensor, tensor_parallelism, sharding_dim)[rank]
        return tensor.contiguous()

    def _shard_inputs(self, tensor: Tensor) -> Tensor:
        if self.sharding_dim == 1:
            # if we are sharding along the k-dim, we need to shard the input accordingly
            rank = self.engine_config.comms.rank
            tensor = torch.tensor_split(tensor, self.tensor_parallelism, -1)[rank]
            return tensor
        elif (self.sharding_dim == 0):
            # but if we shard by row, we just need the inputs on all devices
            return tensor
        else:
            raise ValueError("Unsupported sharding dimension")

    def _shard_inputs_if_needed(self, tensors: Union[Tensor, List[Tensor]]) -> List[Tensor]:
        # if it is a list already, it indicates it is shareded
        sharded_inputs = isinstance(tensors, list)

        if not sharded_inputs:
            return self._shard_inputs(tensors)
        else:
            # already sharded
            # unwrap
            return tensors[0]
    
    def __shard_bias(self, bias: Optional[Tensor]) -> Optional[Tensor]:
        return MuiParallelLinear._shard_bias(self.engine_config, bias, tensor_parallelism=self.tensor_parallelism, sharding_dim=self.sharding_dim)

    @staticmethod
    def _shard_bias(engine_config: MuiEngineConfig, bias: Optional[Tensor], tensor_parallelism: int, sharding_dim: int) -> Optional[Tensor]:
        if bias is None:
            return None
        
        rank = engine_config.comms.rank
        if sharding_dim == 0:
            # if we shard by rows, we need to shard the bias
            tensors = torch.tensor_split(bias, tensor_parallelism, 0)[rank]
            return tensors
        elif sharding_dim == 1:
            # if we shard by columns (k-dim), we should not shard
            # we can instead apply it only on the first GPU
            return bias if rank == 0 else None
        else:
            raise ValueError("Unsupported sharding dimension")

    def __collect_outputs(self, tensor: Tensor) -> Tensor:
        return MuiParallelLinear._collect_outputs(self.engine_config, tensor, self.sharding_dim)

    @staticmethod
    def _collect_outputs(engine_config: MuiEngineConfig, tensor: Tensor, sharding_dim: int) -> Tensor:
        if sharding_dim == 1:
            # reduce
            return engine_config.comms.all_reduce_sum(tensor)
        elif sharding_dim == 0:
            # concat all
            all_tensors = engine_config.comms.all_gather(tensor)
            return torch.cat(all_tensors, dim = -1)
        else:
            # TODO: implement for sharding_dim == 0 (needed by parallel multi linear)
            raise ValueError("Not supported")

    def parallel_forward(self, input: Union[Tensor, List[Tensor]], residual: Optional[Tensor] = None, collect_outputs: bool = True) -> Tensor:
        input = self._shard_inputs_if_needed(input)

        output = _MuiParallelLinear.apply(self.cpp_module, input, residual, collect_outputs)

        # wrap in a list to indicate that it is the output of parallel_forward
        return [output]

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.tensor_parallelism > 1:
            return self.parallel_forward(input, residual)[0]

        raise ValueError("Only parallel inference is supported")