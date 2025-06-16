from typing import List, Optional, Union
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.linear import MuiLinear
from muillm.modules.module import MuiModule
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.norm.rmsnorm import _MuiRMSNorm
import muillm_ext


class _MuiParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, x, residual, reduce):
        output = muillm_ext.muillm_parallel_linear_module_forward(
            module, x, residual=residual, reduce=reduce
        )

        ctx.save_for_backward(x)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")


class MuiParallelLinear(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        in_features: int,
        out_features: int,
        bias: bool = True,
        variance_epsilon: float = 0.0,
        normalize: bool = False,
        sharding_dim: int = 1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(engine_config=engine_config)

        self.cpp_engine = engine_config.cpp_engine
        # the cpp module will be created at the end of all layer replacements
        # (set the field here before potential OOM errors so that it can still be manipulated in
        # the destructor)
        self.cpp_module = None
        self.comms = engine_config.comms
        self.tensor_parallelism = engine_config.tensor_parallelism

        linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.sharding_dim = (
            sharding_dim + len(linear.weight.shape)
            if sharding_dim < 0
            else sharding_dim
        )

        self.in_features = in_features
        self.out_features = out_features
        self.device = linear.weight.device
        self.dtype = linear.weight.dtype

        self.normalize = normalize
        self.variance_epsilon = variance_epsilon
        self.norm_weights = (
            nn.ParameterList([torch.ones(in_features, dtype=dtype, device=device)])
            if normalize
            else None
        )

        self._set_weights(self.__shard_weigths(linear.weight))

        if linear.bias is not None:
            self._set_bias(self.__shard_bias(linear.bias))
        else:
            self.biases = None

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # Need to synchronize after copying the tensors to make sure the transfers
        # completed
        self.__sync_all()

    def finalize_init(self):
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
            None,  # mul_bias
            bias,
            self.sharding_dim,
        )

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _severe_ties(self):
        # severe ties to weights, biases and norm_weights
        weights = self.weights[0]
        self.weights = None
        del weights

        if self.biases is not None:
            biases = self.biases[0]
            self.biases = None
            del biases

        if self.norm_weights is not None:
            norm_weights = self.norm_weights[0]
            self.norm_weights = None
            del norm_weights

        # destroy the C++ module as well to severe the ties to tensors
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_linear_module_deinit(self.cpp_module)
            self.cpp_module = None

    def finalize_deinit(self):
        self._severe_ties()

    def _check_dispatchable(self):
        self.dtype = self.weights[0].dtype
        dispatchable_type = self.dtype == torch.float16
        self.is_cuda = self.weights[0].is_cuda
        dispatchable_device = self.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    @staticmethod
    def _set_requires_grads(params: nn.ParameterList, requires_grads: bool) -> None:
        for param in params:
            if param is not None:
                param.requires_grad = requires_grads

    @staticmethod
    def replace(
        prev_module: Union["MuiParallelLinear", MuiLinear, nn.Linear],
        engine_config: MuiEngineConfig,
        prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None,
        device=None,
    ) -> "MuiParallelLinear":
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiParallelLinear) and (
            prev_layernorm_module is None
        ):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_module

        if isinstance(prev_module, MuiParallelLinear):
            # gather back into a MuiLinear layer to simplify conversion
            new_prev_module = prev_module.to_linear()

            # delete the previous module to save memory
            del prev_module

            # trigger GC to save memory
            trigger_gc()

            prev_module = new_prev_module

        device = prev_module.weight.device if device is None else device
        dtype = prev_module.weight.dtype

        # put on the end device to accelerate things
        # (ok as we are replacing the module entirely so we can change its device)
        if device is not None:
            prev_module = prev_module.to(device)
            prev_layernorm_module = (
                prev_layernorm_module.to(device)
                if prev_layernorm_module is not None
                else None
            )

        has_bias = prev_module.bias is not None
        in_features = prev_module.in_features
        out_features = prev_module.out_features

        if isinstance(prev_module, MuiLinear):
            # due to replacement order, we might get the normalization weights already in
            # or in prev_layernorm_module
            # but not both
            if (prev_module.normalize) and (prev_layernorm_module is not None):
                raise ValueError(
                    "both norm weights in MuiLinear and layernorm module provided"
                )
            if prev_module.normalize:
                normalize = True
                variance_epsilon = prev_module.variance_epsilon
                norm_weights = None  # needs to be None for copy_module
            elif prev_layernorm_module is not None:
                normalize = True
                variance_epsilon = prev_layernorm_module.variance_epsilon
                norm_weights = prev_layernorm_module.weight
            else:
                normalize = False
                variance_epsilon = 0.0
                norm_weights = None

        elif isinstance(prev_module, nn.Linear):
            normalize = prev_layernorm_module is not None
            variance_epsilon = (
                prev_layernorm_module.variance_epsilon if normalize else 0.0
            )
            norm_weights = prev_layernorm_module.weight if normalize else None
        else:
            raise ValueError(
                f"Unsupported replacement to MuiParallelLinear: {prev_module.__class__.__name__}"
            )

        new_module = MuiParallelLinear(
            engine_config=engine_config,
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            variance_epsilon=variance_epsilon,
            normalize=normalize,
            dtype=dtype,
            device=device,
        )
        new_module.copy_module(
            prev_module=prev_module, norm_weights=norm_weights, device=device
        )

        # delete the previous module to save memory
        if isinstance(prev_module, MuiParallelLinear) or isinstance(
            prev_module, MuiLinear
        ):
            prev_module._severe_ties()
        else:
            prev_weights = prev_module.weight
            prev_module.weight = None
            del prev_weights

            if prev_module.bias is not None:
                prev_bias = prev_module.bias
                prev_module.bias = None
                del prev_bias

        if prev_layernorm_module is not None:
            prev_layernorm_weights = prev_layernorm_module.weight
            prev_layernorm_module.weight = None
            del prev_layernorm_weights

        del prev_module

        # trigger GC to save memory
        trigger_gc()

        return new_module

    def _set_norm_weights(
        self, norm_weights: torch.Tensor, requires_grads: Optional[bool] = None
    ) -> None:
        if norm_weights is None:
            self.norm_weights = nn.ParameterList([None])
            return

        if not hasattr(self, "norm_weights") or self.norm_weights is None:
            self.norm_weights = nn.ParameterList(
                [norm_weights.contiguous().clone().detach()]
            )
        else:
            self.norm_weights[0].data.copy_(norm_weights.data)

        MuiParallelLinear._set_requires_grads(
            self.norm_weights,
            (
                requires_grads
                if requires_grads is not None
                else norm_weights.requires_grad
            ),
        )

        # re-create the cpp module
        self.finalize_init()

    def _set_weights(
        self, weights: torch.Tensor, requires_grads: Optional[bool] = None
    ) -> None:
        if not hasattr(self, "weights") or self.weights is None:
            self.weights = nn.ParameterList([weights.contiguous().clone().detach()])
        else:
            self.weights[0].data.copy_(weights.data)

        MuiParallelLinear._set_requires_grads(
            self.weights,
            requires_grads if requires_grads is not None else weights.requires_grad,
        )

    def _set_bias(
        self, bias: torch.Tensor, requires_grads: Optional[bool] = None
    ) -> None:
        if bias is None:
            self.biases = nn.ParameterList([None])
            return

        if not hasattr(self, "biases") or self.biases is None:
            self.biases = nn.ParameterList([bias.contiguous().clone().detach()])
        else:
            self.biases[0].data.copy_(bias.data)

        MuiParallelLinear._set_requires_grads(
            self.biases,
            requires_grads if requires_grads is not None else bias.requires_grad,
        )

    def copy_module(
        self,
        prev_module: Union["MuiParallelLinear", nn.Linear, MuiLinear],
        norm_weights: torch.Tensor = None,
        variance_epsilon: float = 0.0,
        device=None,
    ):
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiParallelLinear):
            # gather back into a MuiLinear layer to simplify conversion
            prev_module = prev_module.to_linear()

        has_bias = prev_module.bias is not None

        self._set_weights(self.__shard_weigths(prev_module.weight))

        if has_bias:
            self._set_bias(self.__shard_bias(prev_module.bias))

        if isinstance(prev_module, MuiLinear):
            # MuiLinear inherits nn.Linear, so need to check first
            if norm_weights is not None:
                raise ValueError("norm_weights should be None")
            norm_weights = prev_module.norm_weights
        elif isinstance(prev_module, nn.Linear):
            # norm_weights need to be set in calling args if needed
            pass
        else:
            raise ValueError(
                f"Unsupported replacement: {prev_module.__class__.__name__}"
            )

        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities
            self._set_norm_weights(norm_weights)

        # put ourselves on the right device
        self.to(device=device)

        self.finalize_init()

        # Need to synchronize after copying the tensors to make sure the transfers
        # completed
        self.__sync_all()

    def __sync_all(self):
        MuiParallelLinear._sync_all(engine_config=self.engine_config)

    @staticmethod
    def _sync_all(engine_config: MuiEngineConfig):
        torch.cuda.synchronize()

    def to_linear(self) -> MuiLinear:
        has_bias = self.biases is not None
        normalize = self.normalize

        linear = MuiLinear(
            engine_config=self.engine_config,
            in_features=self.in_features,
            out_features=self.out_features,
            bias=has_bias,
            variance_epsilon=self.variance_epsilon,
            normalize=normalize,
            device=self.device,
            dtype=self.dtype,
        )

        linear._set_weights(self._unshard_weigths(self.weights[0]))
        if has_bias:
            linear._set_bias(self._unshard_biases(self.biases[0]))
        linear._set_norm_weights(self.norm_weights[0]) if self.normalize else None

        linear.finalize_init()

        return linear

    def __shard_weigths(self, tensor: Tensor) -> Tensor:
        return MuiParallelLinear._shard_weigths(
            self.engine_config, tensor, self.tensor_parallelism, self.sharding_dim
        )

    @staticmethod
    def _shard_weigths(
        engine_config: MuiEngineConfig,
        tensor: Tensor,
        tensor_parallelism: int,
        sharding_dim: int,
    ) -> Tensor:
        rank = engine_config.comms.rank
        tensor = torch.tensor_split(tensor, tensor_parallelism, sharding_dim)[rank]
        return tensor.contiguous()

    def _unshard_weigths(self, tensor: Tensor) -> Tensor:
        # gather the weights
        weights = self.engine_config.comms.all_gather(tensor)
        # and concatenate the pieces
        weight = torch.cat(weights, dim=self.sharding_dim)

        return weight.clone().contiguous().detach()

    def __shard_bias(self, bias: Optional[Tensor]) -> Optional[Tensor]:
        return MuiParallelLinear._shard_bias(
            self.engine_config,
            bias,
            tensor_parallelism=self.tensor_parallelism,
            sharding_dim=self.sharding_dim,
        )

    @staticmethod
    def _shard_bias(
        engine_config: MuiEngineConfig,
        bias: Optional[Tensor],
        tensor_parallelism: int,
        sharding_dim: int,
    ) -> Optional[Tensor]:
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

    def _unshard_biases(self, bias: Optional[Tensor]) -> Optional[Tensor]:
        rank = self.engine_config.comms.rank
        sharding_dim = self.sharding_dim

        if sharding_dim == 0:
            # if we shard by rows, we need to gather the bias
            biases = self.engine_config.comms.all_gather(bias)
            # and concatenate the pieces
            bias = torch.cat(biases, dim=0)
        elif sharding_dim == 1:
            # only the GPU 0 has the bias
            # get it by broadcasting from the GPU 0
            bias = (
                bias
                if rank == 0
                else torch.zeros(size=self.out_features, device=self.device)
            )

            bias = self.engine_config.comms.broadcast(bias, src=0)
        else:
            raise ValueError("Unsupported sharding dimension")

        return bias.clone().contiguous().detach()

    def _shard_inputs(self, tensor: Tensor) -> Tensor:
        if self.sharding_dim == 1:
            # if we are sharding along the k-dim, we need to shard the input accordingly
            rank = self.engine_config.comms.rank
            tensor = torch.tensor_split(tensor, self.tensor_parallelism, -1)[rank]
            return tensor
        elif self.sharding_dim == 0:
            # but if we shard by row, we just need the inputs on all devices
            return tensor
        else:
            raise ValueError("Unsupported sharding dimension")

    def _shard_inputs_if_needed(
        self, tensors: Union[Tensor, List[Tensor]]
    ) -> List[Tensor]:
        # if it is a list already, it indicates it is shareded
        sharded_inputs = isinstance(tensors, list)

        if not sharded_inputs:
            return self._shard_inputs(tensors)
        else:
            # already sharded
            # unwrap
            return tensors[0]

    def __collect_outputs(self, tensor: Tensor) -> Tensor:
        return MuiParallelLinear._collect_outputs(
            self.engine_config, tensor, self.sharding_dim
        )

    @staticmethod
    def _collect_outputs(
        engine_config: MuiEngineConfig, tensor: Tensor, sharding_dim: int
    ) -> Tensor:
        if sharding_dim == 1:
            # reduce
            return engine_config.comms.all_reduce_sum(tensor)
        elif sharding_dim == 0:
            # concat all
            all_tensors = engine_config.comms.all_gather(tensor)
            return torch.cat(all_tensors, dim=-1)
        else:
            # TODO: implement for sharding_dim == 0 (needed by parallel multi linear)
            raise ValueError("Not supported")

    def parallel_forward(
        self,
        input: Union[Tensor, List[Tensor]],
        residual: Optional[Tensor] = None,
        collect_outputs: bool = True,
    ) -> Tensor:
        input = self._shard_inputs_if_needed(input)

        output = _MuiParallelLinear.apply(
            self.cpp_module, input, residual, collect_outputs
        )

        # wrap in a list to indicate that it is the output of parallel_forward
        return [output]

    def forward(self, input: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        if self.tensor_parallelism > 1:
            return self.parallel_forward(input, residual)[0]

        raise ValueError("Only parallel inference is supported")
