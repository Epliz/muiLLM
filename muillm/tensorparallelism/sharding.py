import torch
import torch.distributed as dist

def _shard_size(shape, N: int, tensor_parallelism:int):
    if N % tensor_parallelism != 0:
        raise ValueError(f"The provided tensor of shape {shape} cannot be sharded for tp {tensor_parallelism} as the dimension of size {N} is not divisible")

    return N // tensor_parallelism

def _shard(tensor: torch.Tensor, tensor_parallelism: int, dim=-1) -> torch.Tensor:
    if tensor_parallelism <= 1:
        return tensor
    
    rank = dist.get_rank()
    return torch.tensor_split(tensor, tensor_parallelism, dim=dim)[rank].contiguous()

