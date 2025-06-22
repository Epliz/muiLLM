from typing import List
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.linear import MuiLinear
import torch
import torch.nn as nn

from muillm.modules.multilinear import MuiMultiLinear

from .test_utils import copy_linear, random_linear, tensors_equal


def _test_basic_linear(in_features: int, device: str = "cpu", dtype=torch.float16):
    linear = random_linear(
        in_features=in_features,
        out_features=1024,
        bias=False,
        device=device,
        dtype=dtype,
    )

    # replace destroys the passed linear module so we need to copy it
    linear_copy = copy_linear(linear)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    muilinear = MuiLinear.replace(
        prev_module=linear_copy,
        engine_config=engine_config,
        device=device,
    )
    muilinear.finalize_init()

    input_tensor = torch.rand(size=(4, in_features), device=device, dtype=dtype)

    y = linear(input_tensor)

    y_m = muilinear(input_tensor)

    tensors_equal(y, y_m)


def test_basic_linear_fp32_cpu():
    device = "cpu"
    in_features = 2048
    _test_basic_linear(in_features, device, dtype=torch.float32)


def test_basic_linear_fp32_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(in_features, device, dtype=torch.float32)


def test_basic_linear_fp16_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(in_features, device, dtype=torch.float16)


def test_basic_linear_bf16_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(in_features, device, dtype=torch.bfloat16)


def _test_linear_bias(in_features: int, device: str, dtype=torch.float16):
    linear = random_linear(
        in_features=in_features,
        out_features=1024,
        bias=True,
        device=device,
        dtype=dtype,
    )

    # replace destroys the passed linear module so we need to copy it
    linear_copy = copy_linear(linear)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    muilinear = MuiLinear.replace(
        prev_module=linear_copy,
        engine_config=engine_config,
        device=device,
    )
    muilinear.finalize_init()

    input_tensor = torch.rand(size=(4, in_features), device=device, dtype=dtype)

    y = linear(input_tensor)

    y_m = muilinear(input_tensor)

    tensors_equal(y, y_m)


def test_linear_bias_fp32_cpu():
    device = "cpu"
    in_features = 2048
    _test_linear_bias(in_features, device, dtype=torch.float32)


def test_linear_bias_fp32_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(in_features, device, dtype=torch.float32)


def test_linear_bias_fp16_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(in_features, device, dtype=torch.float16)


def test_linear_bias_bf16_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(in_features, device, dtype=torch.bfloat16)


# TODO tests with input norm
