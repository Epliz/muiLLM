from typing import Iterable, List, Optional, Tuple, Union
from muillm.hftensorparallelism.hftensorparallelism import _to_local_module
from muillm.engineconfig import (
    MuiEngineConfig,
)
from muillm.modules.module import MuiModule
from muillm.modules.multilinear import MuiMultiLinear
from muillm.modules.parallellinear import MuiParallelLinear
import torch
from torch import Tensor
import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.linear import MuiLinear
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

import muillm_ext

from muillm.modules.norm.rmsnorm import MuiRMSNorm
from muillm.replacement.replacementcontext import MuiReplacementContext


class _MuiParallelMultiLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, x, collect_outputs):
        output = muillm_ext.muillm_parallel_multilinear_module_forward(
            module, x, collect_outputs
        )

        ctx.save_for_backward(x)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("Not implemented")


def _all_or_none(it: Iterable[bool], exception_message) -> bool:
    has_all = all([i for i in it])
    has_any = not all([not i for i in it])

    if has_any and not has_all:
        raise ValueError(exception_message)
    return has_all


class MuiParallelMultiLinear(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        in_features: int,
        out_features: List[int],
        bias: bool = True,
        variance_epsilon: float = 0.0,
        normalize: bool = False,
        sharding_dim: int = 0,
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
        self.sharding_dim = sharding_dim if sharding_dim >= 0 else sharding_dim + 2

        self.linear = MuiParallelLinear(
            engine_config=engine_config,
            in_features=in_features,
            out_features=sum(out_features),
            bias=bias,
            variance_epsilon=variance_epsilon,
            normalize=normalize,
            sharding_dim=sharding_dim,
            device=device,
            dtype=dtype,
        )

        self.in_features = in_features
        self.out_features = list(out_features)

        self.slice_starts = []
        self.slice_ends = []

        current_start = 0
        for out_feature in out_features:
            sharded_out_feature = (
                out_feature
                if sharding_dim == 1
                else self._safe_div(out_feature, self.tensor_parallelism)
            )

            current_end = current_start + sharded_out_feature

            self.slice_starts.append(current_start)
            self.slice_ends.append(current_end)

            current_start = current_end

        self.slices = list(zip(self.slice_starts, self.slice_ends))

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # Need to synchronize after copying the tensors to make sure the transfers
        # completed
        self.__sync_all()

    def finalize_init(self):
        self.linear.finalize_init()

        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_multilinear_module_deinit(self.cpp_module)

        self.cpp_module = muillm_ext.muillm_parallel_multilinear_module_init(
            self.cpp_engine,
            self.comms.comms,
            self.linear.cpp_module,
            self.slices,
            self.sharding_dim,
        )

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        self.dispatchable = self.linear.dispatchable

    def finalize_deinit(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_multilinear_module_deinit(self.cpp_module)
            self.cpp_module = None

    def _safe_div(self, a, b):
        if a % b != 0:
            raise ValueError("Not divisible entirely")

        return int(a / b)

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_modules: Union[MuiMultiLinear, List[nn.Linear]],
        prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None,
        sharding_dim: int = 1,
    ) -> "MuiParallelMultiLinear":
        engine_config = replacement_context.engine_config
        device = replacement_context.device
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_modules, MuiMultiLinear):
            return MuiParallelMultiLinear._replace_multilinear(
                replacement_context,
                prev_module=prev_modules,
                prev_layernorm_module=prev_layernorm_module,
                sharding_dim=sharding_dim,
                device=device,
            )
        elif isinstance(prev_modules, list):
            return MuiParallelMultiLinear._replace_linears(
                replacement_context,
                prev_modules=prev_modules,
                prev_layernorm_module=prev_layernorm_module,
                sharding_dim=sharding_dim,
                device=device,
            )
        else:
            raise ValueError(f"Not supported {type(prev_modules)}")

    @staticmethod
    def _replace_multilinear(
        replacement_context: MuiReplacementContext,
        prev_module: MuiMultiLinear,
        prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None,
        sharding_dim: int = 0,
        device=None,
    ) -> "MuiParallelMultiLinear":
        engine_config = replacement_context.engine_config

        if device is None:
            raise ValueError("device was None")

        if prev_layernorm_module is not None:
            raise ValueError("prev_layernorm_module should be None")

        # put on the end device to accelerate things
        # (ok as we are replacing the module entirely so we can change its device)
        if device is not None:
            prev_module = prev_module.to(device=device)
            prev_layernorm_module = (
                prev_layernorm_module.to(device=device)
                if prev_layernorm_module is not None
                else None
            )

        has_bias = prev_module.linear.bias is not None

        in_features = prev_module.in_features
        out_features = prev_module.out_features
        dtype = prev_module.linear.weight.dtype
        device = prev_module.linear.weight.device

        normalize = prev_module.linear.normalize
        variance_epsilon = prev_module.linear.variance_epsilon

        new_module = MuiParallelMultiLinear(
            engine_config=engine_config,
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            variance_epsilon=variance_epsilon,
            normalize=normalize,
            sharding_dim=sharding_dim,
            dtype=dtype,
            device=device,
        )
        new_module.copy_module(prev_module=prev_module, device=device)

        return new_module

    @staticmethod
    def _replace_linears(
        replacement_context: MuiReplacementContext,
        prev_modules: List[Union[MuiParallelLinear, nn.Linear]],
        prev_layernorm_module: Union[LlamaRMSNorm, MistralRMSNorm] = None,
        sharding_dim: int = 0,
        device=None,
    ) -> "MuiParallelMultiLinear":
        engine_config = replacement_context.engine_config

        if device is None:
            raise ValueError("device was None")

        if len(prev_modules) < 1:
            raise ValueError(
                "MuiMultiLinear needs some linear layers passed in but none were provided"
            )

        # gather back into MuiLinear layers to simplify conversion
        prev_modules = [
            (
                prev_module.to_linear()
                if isinstance(prev_module, MuiParallelLinear)
                else replacement_context.to_local_module(prev_module)
            )
            for prev_module in prev_modules
        ]

        if prev_layernorm_module is not None:
            if not isinstance(prev_layernorm_module, MuiRMSNorm):
                prev_layernorm_module = replacement_context.to_local_module(
                    prev_layernorm_module
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
            prev_modules = [
                prev_module.to(device=device) for prev_module in prev_modules
            ]
            prev_layernorm_module = (
                prev_layernorm_module.to(device=device)
                if prev_layernorm_module is not None
                else None
            )

        normalize = prev_layernorm_module is not None
        variance_epsilon = (
            MuiRMSNorm._extract_eps(prev_layernorm_module) if normalize else 0.0
        )
        norm_weights = prev_layernorm_module.weight if normalize else None

        new_module = MuiParallelMultiLinear(
            engine_config=engine_config,
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            variance_epsilon=variance_epsilon,
            normalize=normalize,
            sharding_dim=sharding_dim,
            dtype=dtype,
            device=device,
        )
        new_module.copy_modules(
            prev_modules=prev_modules, norm_weights=norm_weights, device=device
        )

        return new_module

    def copy_module(self, prev_module: MuiMultiLinear, device=None):
        if device is None:
            raise ValueError("device was None")

        prev_modules, norm_weights = prev_module.replace_back()
        self.copy_modules(
            prev_modules=prev_modules, norm_weights=norm_weights, device=device
        )

    def _cat_dim0(self, tensors: List[Optional[torch.Tensor]]) -> torch.Tensor:
        if all(t is None for t in tensors):
            return None

        return torch.cat(tensors, dim=0)

    def copy_modules(
        self,
        prev_modules: List[nn.Linear],
        norm_weights: torch.Tensor = None,
        variance_epsilon: float = 0.0,
        device=None,
    ):
        if device is None:
            raise ValueError("device was None")

        has_bias = self.linear.biases is not None

        num_linears = len(prev_modules)

        # For each linear, shard it, then make the per device tensor by concatenating all the pieces
        # We need to do it this way so that for example the sharded attention works by having the heads
        # well split
        all_weights = [
            MuiParallelLinear._shard_weigths(
                self.engine_config,
                prev_module.weight,
                self.tensor_parallelism,
                self.sharding_dim,
            )
            for prev_module in prev_modules
        ]
        concat_weights = torch.cat([all_weights[i] for i in range(num_linears)], dim=0)
        concat_weights_requires_grad = _all_or_none(
            [prev_module.weight.requires_grad for prev_module in prev_modules],
            "all or none weights must required grads but got a mix",
        )

        self.linear._set_weights(concat_weights, concat_weights_requires_grad)

        if has_bias:
            all_biases = [
                MuiParallelLinear._shard_bias(
                    self.engine_config,
                    prev_module.bias,
                    self.tensor_parallelism,
                    self.sharding_dim,
                )
                for prev_module in prev_modules
            ]
            concat_biases = self._cat_dim0([all_biases[i] for i in range(num_linears)])
            concat_biases_requires_grad = _all_or_none(
                [prev_module.bias.requires_grad for prev_module in prev_modules],
                "all or none biases must required grads but got a mix",
            )

            self.linear._set_bias(concat_biases, concat_biases_requires_grad)

        # norm_weights are not sharded
        if norm_weights is not None:
            # the rescaling weights are not fused in the matrices due to instabilities
            self.linear._set_norm_weights(norm_weights)

        # put ourselves on the right device
        self.to(device=device)

        self.finalize_init()

        # Need to synchronize after copying the tensors to make sure the transfers
        # completed
        self.__sync_all()

    def __sync_all(self):
        MuiParallelLinear._sync_all(engine_config=self.engine_config)

    def _shard_inputs_if_needed(
        self, tensors: Union[Tensor, List[Tensor]]
    ) -> List[Tensor]:
        return self.linear._shard_inputs_if_needed(tensors)

    def __slice_outputs(self, all_outputs: torch.Tensor) -> List[torch.Tensor]:
        return [
            all_outputs[..., slice_start:slice_end]
            for slice_start, slice_end in self.slices
        ]

    def __collect_output(self, tensor: torch.Tensor) -> Tuple[List[torch.Tensor], ...]:
        return MuiParallelLinear._collect_outputs(
            self.engine_config, tensor, sharding_dim=self.sharding_dim
        )

    def parallel_forward(
        self, input: Union[Tensor, List[Tensor]], collect_outputs: bool = True
    ) -> List[Tuple[Tensor, ...]]:

        if self.dispatchable:
            input = self._shard_inputs_if_needed(input)
            output = _MuiParallelMultiLinear.apply(
                self.cpp_module, input, collect_outputs
            )
            return [tuple(output)]

        # not dispatchable
        num_slices = len(self.slice_starts)

        # if we are sharding by columns, we can let MuiParallelLinear collect the results
        if self.sharding_dim == 1:
            all_outputs = self.linear.parallel_forward(
                input, collect_outputs=collect_outputs
            )[0]
            # outputs are already collected if they need to be
            all_split_outputs = self.__slice_outputs(all_outputs)
            return [tuple(all_split_outputs)]

        # but if we shard by row, we need to do it manually as tensors are interleaved and it would not give
        # the right results
        all_outputs = self.linear.parallel_forward(input, collect_outputs=False)[0]

        # output x device
        all_split_outputs = self.__slice_outputs(all_outputs=all_outputs)

        if collect_outputs:  # sharding by rows
            # collect the outputs if necessary
            for output_idx in range(num_slices):
                all_split_outputs[output_idx] = self.__collect_output(
                    # tensors need to be contiguous for the all-reduce
                    all_split_outputs[output_idx].contiguous()
                )

        return [tuple(all_split_outputs)]

    def forward(self, input: Tensor) -> Tuple[Tensor, ...]:
        if self.tensor_parallelism > 1:
            outputs = self.parallel_forward(input)[0]

            return tuple(output for output in outputs)

        raise ValueError("Only parallel inference is supported")
