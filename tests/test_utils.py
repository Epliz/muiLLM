import os
from typing import Any, Dict, List, Tuple
import numpy
import torch
import torch.nn as nn


import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def max_diff(comp: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> Dict[str, float]:
    comp = comp.reshape(-1)
    t1 = t1.reshape(-1)
    t2 = t2.reshape(-1)

    max_val, max_idx = torch.max(comp, dim=0)
    return {"max_diff_idx": max_idx.item(), "max_diff_val": max_val.item(), "t1_val": t1[max_idx].item(), "t2_val": t2[max_idx].item()}


def tensors_equal(y, y_m, rtol=1e-02, y_highres: torch.Tensor = None):
    same_shapes = y.shape == y_m.shape

    if not same_shapes:
        print(f"Shapes are different: {y.shape} vs {y_m.shape}")
        assert False

    y = y.float()
    y_m = y_m.float()

    # we don't care so much about absolute differences, but rather relative differences
    rel_eps = 1e-08

    if y_highres is not None:
        # we prefer to compare to a high resolution (fp32 computed reference - y_highres)
        rel_diff_highres_y = 2.0 * torch.abs(y - y_highres) / (torch.abs(y) + torch.abs(y_highres) + rel_eps)
        print(f"Max relative difference y vs high-res: {max_diff(rel_diff_highres_y, y, y_highres)})")
        rel_diff_highres_ym = 2.0 * torch.abs(y_m - y_highres) / (torch.abs(y_m) + torch.abs(y_highres) + rel_eps)
        print(f"Max relative difference y_m vs high-res: {max_diff(rel_diff_highres_ym, y_m, y_highres)})")
        ym_yhighres_relatively_close = torch.all(rel_diff_highres_ym <= rtol).cpu().item()

        if not ym_yhighres_relatively_close:
            print(f"Tensors are not close enough: y_m vs high-res: {y_m} vs {y_highres}")
            print(f"Difference: {y_m - y_highres}")
            assert False
    else:
        rel_diff = 2.0 * torch.abs(y - y_m) / (torch.abs(y) + torch.abs(y_m) + rel_eps)

        relatively_close = torch.all(rel_diff <= rtol).cpu().item()
        close_enough = relatively_close

        #print(f"Max absolute difference: {max_diff(abs_diff, t1, t2)})")
        print(f"Max relative difference: {max_diff(rel_diff, y, y_m)})")
        if not close_enough:
            print(f"Tensors are not close enough: {y} vs {y_m}")
            print(f"Difference: {y - y_m}")
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
