import os
from typing import Any, Dict, List
from muillm.engineconfig import MuiEngineConfig
import torch
import torch.nn as nn

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from muillm.modules.multilinear import MuiMultiLinear
from muillm.modules.parallelmultilinear import MuiParallelMultiLinear
from .test_utils import execute_distributed, tensors_equal, copy_linears, random_linears


def _test_basic_linears(in_features: int, out_features: List[int], device: str):
    linears = random_linears(
        in_features=in_features, out_features=out_features, device=device
    )

    # replace destroys the passed linear module so we need to copy it
    linear_copies = copy_linears(linears)

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    multilinear = MuiParallelMultiLinear.replace(
        prev_modules=linear_copies,
        engine_config=engine_config,
        device=device,
    )
    multilinear.finalize_init()

    input_tensor = torch.rand(size=(4, in_features), device=device)

    q = linears[0](input_tensor)
    k = linears[1](input_tensor)
    v = linears[2](input_tensor)

    q_m, k_m, v_m = multilinear(input_tensor)

    tensors_equal(q, q_m)
    tensors_equal(k, k_m)
    tensors_equal(v, v_m)


def test_basic_linears():
    execute_distributed(
        _test_basic_linears, in_features=128, out_features=[4, 8, 8], device="cuda"
    )


# TODO tests with bias and no bias
# TODO tests with other sharding dimensions
# TODO tests with input norm
# TODO tests with other data types
