from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.distributed as dist

import os

import muillm_ext


class Communicator(ABC):
    def __init__(
            self,
            world_size: Optional[int] = None,
            local_size: Optional[int] = None,
            rank: Optional[int] = None,
            local_rank: Optional[int] = None):

        self.world_size = world_size if world_size is not None else int(os.environ['WORLD_SIZE'])
        self.local_size = local_size if local_size is not None else int(os.environ['LOCAL_SIZE'])
        self.rank = rank if rank is not None else int(os.environ['RANK'])
        self.local_rank = local_rank if local_rank is not None else int(os.environ['LOCAL_RANK'])

        if self.world_size is None:
            # Assume not distributed
            self.world_size = 1
            self.local_size = 1
            self.rank = 0
            self.local_rank = 0

    def is_multi_node(self) -> bool:
        return self.world_size != self.local_size

    @abstractmethod
    def broadcast(self, tensor:torch.Tensor, src: int):
        pass

    @abstractmethod
    def all_reduce_sum(self, tensor:torch.Tensor):
        pass

class TorchCommunicator(Communicator):
    def __init__(
            self,
            backend='nccl',
            world_size: Optional[int] = None,
            local_size: Optional[int] = None,
            rank: Optional[int] = None,
            local_rank: Optional[int] = None):

        super().__init__(world_size=world_size, local_size=local_size, rank=rank, local_rank=local_rank)

        if not dist.is_available():
            raise ValueError("torch dist is not available")

        # set the current device to the GPU we need
        torch.cuda.set_device(self.local_rank)

        if not dist.is_initialized():
            # initialize the default group
            dist.init_process_group(backend, rank=self.rank, world_size=self.world_size)

    def broadcast(self, tensor:torch.Tensor, src: int):
        dist.broadcast(tensor, src=src)

    def all_reduce_sum(self, tensor:torch.Tensor):
        dist.all_reduce(tensor)

class _MuiCommBarrier(torch.autograd.Function):
    # Primarily so that in the pytorch trace, we see the operation with a name and not just HIP kernel launches

    @staticmethod
    def forward(ctx, comm):
        muillm_ext.muillm_barrier(comm)

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("not implemented")

class _MuiCommBroadcast(torch.autograd.Function):
    # Primarily so that in the pytorch trace, we see the operation with a name and not just HIP kernel launches

    @staticmethod
    def forward(ctx, comm, tensor: torch.Tensor, src: int):
        muillm_ext.muillm_broadcast(comm, tensor, src)

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("not implemented")


class _MuiCommAllReduceSum(torch.autograd.Function):
    # Primarily so that in the pytorch trace, we see the operation with a name and not just HIP kernel launches

    @staticmethod
    def forward(ctx, comm, tensor: torch.Tensor):
        muillm_ext.muillm_all_reduce_sum(comm, tensor)

    @staticmethod
    def backward(ctx, grad_output):
        raise ValueError("not implemented")

class MuiCommunicator(Communicator):
    def __init__(
            self,
            world_size: Optional[int] = None,
            local_size: Optional[int] = None,
            rank: Optional[int] = None,
            local_rank: Optional[int] = None):

        super().__init__(world_size=world_size, local_size=local_size, rank=rank, local_rank=local_rank)

        if (self.is_multi_node()):
            raise ValueError(f"Multi-node setups are not supported with MuiCommunicator, use TorchCommunicator instead. Specified WORLD_SIZE {self.world_size} and LOCAL_SIZE {self.local_size}")

        # set the current device to the GPU we need
        torch.cuda.set_device(self.local_rank)

        self.comm = muillm_ext.muillm_comm_init(self.world_size, self.local_size, self.rank, self.local_rank)

    def barrier(self):
        _MuiCommBarrier.apply(self.comm)

    def broadcast(self, tensor:torch.Tensor, src: int):
        #dist.broadcast(tensor, src=src)
        _MuiCommBroadcast.apply(self.comm, tensor, src)

    def all_reduce_sum(self, tensor:torch.Tensor):
        #dist.all_reduce(tensor)
        _MuiCommAllReduceSum.apply(self.comm, tensor)