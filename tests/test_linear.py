from typing import List
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.linear import MuiLinear
import torch
import torch.nn as nn

from muillm.modules.multilinear import MuiMultiLinear

from .test_utils import copy_linear, random_linear, tensors_equal


def _test_basic_linear(in_features: int, device: str):
    linear = random_linear(
        in_features=in_features, out_features=1024, bias=False, device=device
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

    input_tensor = torch.rand(size=(4, in_features), device=device)

    y = linear(input_tensor)

    y_m = muilinear(input_tensor)

    tensors_equal(y, y_m)


def test_basic_linear_cpu():
    device = "cpu"
    in_features = 2048
    _test_basic_linear(in_features, device)


def test_basic_linear_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(in_features, device)


def _test_linear_bias(in_features: int, device: str):
    linear = random_linear(
        in_features=in_features, out_features=1024, bias=True, device=device
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

    input_tensor = torch.rand(size=(4, in_features), device=device)

    y = linear(input_tensor)

    y_m = muilinear(input_tensor)

    tensors_equal(y, y_m)


def test_linear_bias_cpu():
    device = "cpu"
    in_features = 2048
    _test_linear_bias(in_features, device)


def test_linear_bias_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(in_features, device)


# TODO tests with input norm
# TODO tests with other data types
