from typing import Optional, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.module import MuiModule

import torch
import torch.nn as nn

import muillm_ext
from muillm.replacement.replacementcontext import MuiReplacementContext


class _MuiEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, x):
        output = muillm_ext.muillm_embedding_module_forward(module, x)

        ctx.save_for_backward(x)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("embedding backward is not implemented")


class MuiEmbedding(MuiModule, nn.Module):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
    ):
        MuiModule.__init__(self, engine_config)
        nn.Module.__init__(self)

        self.cpp_engine = engine_config.cpp_engine

        embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            device=device,
            dtype=dtype,
        )

        self.weight = embeddings.weight

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # the cpp module will be created at the end of all layer replacements
        self.cpp_module = None

    def _check_dispatchable(self):
        wdtype = self.weight.dtype
        dispatchable_type = (wdtype == torch.float16) or (wdtype == torch.bfloat16)
        dispatchable_device = self.weight.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def finalize_init(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_embedding_module_deinit(self.cpp_module)

        self.cpp_module = muillm_ext.muillm_embedding_module_init(
            self.cpp_engine,
            self.weight,
        )

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_module: Union["MuiEmbedding", nn.Embedding],
    ) -> "MuiEmbedding":

        engine_config = replacement_context.engine_config
        device = replacement_context.device
        if device is None:
            raise ValueError("device was None")

        if isinstance(prev_module, MuiEmbedding):
            # re-creating a module would change nothing, we can just avoid id
            return prev_module

        # Make sure we convert the previous module to a local module
        # so that we can safely copy its parameters
        prev_module = replacement_context.to_local_module(prev_module)

        device = prev_module.weight.device if device is None else device
        dtype = prev_module.weight.dtype

        # put on the end device to accelerate things
        # (ok as we are replacing the module entirely so we can change its device)
        if device is not None:
            prev_module = prev_module.to(device)

        new_module = MuiEmbedding(
            engine_config=engine_config,
            num_embeddings=prev_module.num_embeddings,
            embedding_dim=prev_module.embedding_dim,
            padding_idx=prev_module.padding_idx,
            max_norm=prev_module.max_norm,
            norm_type=prev_module.norm_type,
            scale_grad_by_freq=prev_module.scale_grad_by_freq,
            sparse=prev_module.sparse,
            device=device,
            dtype=dtype,
        )

        new_module.copy_module(prev_module, device=device)

        # delete the previous modules to free memory
        del prev_module.weight

        # trigger garbage collection to free memory
        trigger_gc()

        return new_module

    def copy_module(
        self,
        prev_module: nn.Embedding,
        device=None,
    ):
        if device is None:
            raise ValueError("device was None")

        requires_grads = prev_module.weight.requires_grad
        self.weight.data.copy_(prev_module.weight.data)
        self.weight.requires_grad = requires_grads

        # put ourselves on the right device
        self.to(device=device)

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.cpp_module is not None:
            return _MuiEmbedding.apply(
                self.cpp_module,
                input,
            )

        # TODO: to support training we would have to pass all the other parameters
        # (padding_idx, max norm etc)
        return torch.nn.functional.embedding(input, self.weight)
