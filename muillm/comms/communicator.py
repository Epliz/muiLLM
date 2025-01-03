from typing import List, Optional
import torch

import muillm_ext

class Communicator:
    """
    Provides functionalities to multi-GPU communications (but single-process)
    """
    def __init__(self, tensor_parallelism: int, devices):

        self.devices = devices
        self.tensor_parallelism = tensor_parallelism

    def transfer_back(self, tensors: Optional[List[torch.Tensor]]) -> List[torch.Tensor]:
        if tensors is None:
            return None

        device = self.devices[0]
        moved_tensors = [t.to(device=device, dtype=t.dtype) if t is not None else None for t in tensors] 

        # make the stream 0 wait for the other GPUs
        # torch already inserts waits
        #self._wait_for_others(d = 0)

        return moved_tensors

    def transfer_across(self, tensors: Optional[List[torch.Tensor]]) -> Optional[List[torch.Tensor]]:
        if tensors is None:
            return None
        
        devices = self.devices
        moved_tensors = [t.to(device=devices[i], dtype=t.dtype) if t is not None else None for i, t in enumerate(tensors)] 

        # make all streams of the other devices wait on the GPU0
        # torch already inserts waits
        #self._wait_for(d = 0)

        return moved_tensors
    
    def all_reduce(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:

        # transfer all outputs back on GPU0 
        tensors = self.transfer_back(tensors)

        # reduce on GPU0
        output = tensors[0]
        for i in range(1, self.tensor_parallelism):
            output = output + tensors[i]

        outputs = [output] * self.tensor_parallelism

        # transfer to the different GPUs
        outputs = self.transfer_across(outputs)

        return outputs
    
    def concat_all(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        # transfer all outputs back on GPU0 
        tensors = self.transfer_back(tensors)

        # concatenate them all on GPU0
        # sharding the weights by row means we need to concatenate all out features
        # which is concatenating on the last dimension
        output = torch.cat(tensors, dim=-1)

        outputs = [output] * self.tensor_parallelism

        # transfer to the different GPUs
        outputs = self.transfer_across(outputs)

        return outputs