import os
from typing import Any, Dict, List, Tuple
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.linear import MuiLinear
import torch
import torch.nn as nn

from muillm.modules.multilinear import MuiMultiLinear


import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from muillm.modules.parallellinear import MuiParallelLinear
from muillm.replacement.replacementcontext import MuiReplacementContext
from .test_utils import (
    copy_linear,
    execute_distributed,
    random_linear,
    tensors_equal,
)


def _test_basic_linear(
    in_features: int, out_features: int, device: str, dtype: torch.dtype = torch.float16
):
    linear = random_linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        device=device,
        dtype=dtype,
    )

    # replace destroys the passed linear module so we need to copy it
    linear_copy = copy_linear(linear)

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )
    muilinear = MuiParallelLinear.replace(
        replacement_context=replacement_context,
        prev_module=linear_copy,
    )
    muilinear.finalize_init()

    input_tensor = torch.rand(size=(4, in_features), device=device, dtype=dtype)

    y = linear(input_tensor)

    y_m = muilinear(input_tensor)

    tensors_equal(y, y_m)


def test_basic_linear():
    execute_distributed(
        _test_basic_linear, in_features=512, out_features=32, device="cuda"
    )


def _test_linear_bias(
    in_features: int, out_features: int, device: str, dtype: torch.dtype = torch.float16
):
    linear = random_linear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        device=device,
        dtype=dtype,
    )

    # replace destroys the passed linear module so we need to copy it
    linear_copy = copy_linear(linear)

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )
    muilinear = MuiParallelLinear.replace(
        replacement_context=replacement_context,
        prev_module=linear_copy,
    )
    muilinear.finalize_init()

    input_tensor = torch.rand(size=(4, in_features), device=device, dtype=dtype)

    y = linear(input_tensor)

    y_m = muilinear(input_tensor)

    tensors_equal(y, y_m)


def test_linear_bias():
    execute_distributed(
        _test_linear_bias, in_features=512, out_features=32, device="cuda"
    )


# TODO tests with input norm
# TODO tests with other sharding dimensions
# TODO tests with other data types (fp16, bf16, etc)
# TODO test after finalizing module
