from abc import ABC, abstractmethod
from typing import List, Optional
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
    def broadcast(self, tensor:torch.Tensor, src: int) -> torch.Tensor:
        pass

    @abstractmethod
    def all_reduce_sum(self, tensor:torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def all_gather(self, tensor:torch.Tensor) -> List[torch.Tensor]:
        pass

class TorchCommunicator(Communicator):
    def __init__(
            self,
            backend='gloo',
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

        print(f"Initialized rank {self.rank}")

    def broadcast(self, tensor:torch.Tensor, src: int) -> torch.Tensor:
        dist.broadcast(tensor, src=src)
        return tensor

    def all_reduce_sum(self, tensor:torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor)
        return tensor

    def all_gather(self, tensor:torch.Tensor) -> List[torch.Tensor]:
        output_tensors = [torch.empty_like(tensor) for i in range(self.world_size)]
        dist.all_gather(output_tensors, tensor)
        return output_tensors