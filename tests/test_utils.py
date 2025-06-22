import os
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn


import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def tensors_equal(t1, t2, rtol=1e-04):
    same_shapes = t1.shape == t2.shape

    if not same_shapes:
        print(f"Shapes are different: {t1.shape} vs {t2.shape}")
        assert False

    # we don't care so much about absolute differences, but rather relative differences
    close_enough = torch.allclose(t1, t2, rtol=rtol, atol=1e-01)
    if not close_enough:
        abs_diff = torch.abs(t1 - t2)
        rel_diff = torch.abs(t1 - t2) / (torch.abs(t1) + 1e-8)
        print(f"Tensors are not close enough: {t1} vs {t2}")
        print(f"Max absolute difference: {abs_diff.max()})")
        print(f"Max relative difference: {rel_diff.max()})")
        assert False


def random_linear(
    in_features: int,
    out_features: int,
    bias: bool = False,
    device="cuda",
    dtype=torch.float16,
) -> nn.Linear:
    linear = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        dtype=dtype,
    )

    # We seed to have reproducible results
    torch.manual_seed(0)
    linear.weight = nn.Parameter(torch.randn_like(linear.weight))
    if linear.bias is not None:
        linear.bias = nn.Parameter(torch.randn_like(linear.bias))

    return linear


def copy_linear(linear: nn.Linear) -> nn.Linear:
    device = linear.weight.device
    dtype = linear.weight.dtype
    new_linear = nn.Linear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
        device=device,
        dtype=dtype,
    )
    new_linear.weight = nn.Parameter(linear.weight.clone().detach())
    if linear.bias is not None:
        new_linear.bias = nn.Parameter(linear.bias.clone().detach())

    return new_linear


def random_linears(
    in_features: int, out_features: List[int], device: str, dtype=torch.float16
) -> List[nn.Linear]:
    return [
        random_linear(
            in_features=in_features, out_features=out_feat, device=device, dtype=dtype
        )
        for out_feat in out_features
    ]


def copy_linears(linears: List[nn.Linear]) -> List[nn.Linear]:
    return [copy_linear(linear) for linear in linears]


def init_process(rank, fn, fn_args: Dict[str, Any]):
    local_size = torch.cuda.device_count()
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(local_size)
    os.environ["LOCAL_SIZE"] = str(local_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    local_size = torch.cuda.device_count()
    print(f"(rank {rank}) local_size = {local_size}")

    # set the current device to the GPU we need
    torch.cuda.set_device(rank)

    dist.init_process_group("nccl", rank=rank, world_size=local_size)

    fn(**fn_args)


def execute_distributed(func, **kwargs: Dict[str, Any]):
    # cf. https://github.com/pytorch/pytorch/issues/3492

    # get the number of GPUs
    size = torch.cuda.device_count()

    print(f"{size} GPUs available for testing.")

    # Spawn one subprocess per GPU
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, func, kwargs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for p in processes:
        if p.exitcode != 0:
            raise RuntimeError(f"Process {p.pid} exited with code {p.exitcode}")
