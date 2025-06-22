from typing import List
from muillm.engineconfig import MuiEngineConfig
import torch
import torch.nn as nn

from muillm.modules.multilinear import MuiMultiLinear
from .test_utils import copy_linears, random_linears, tensors_equal


def test_basic_linears():
    dtype = torch.float16
    device = "cpu"
    in_features = 4096
    linears = random_linears(
        in_features=in_features,
        out_features=[1024, 2048, 4096],
        device=device,
        dtype=dtype,
    )

    # replace destroys the passed linear module so we need to copy it
    linear_copies = copy_linears(linears)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    multilinear = MuiMultiLinear.replace(
        prev_modules=linear_copies,
        engine_config=engine_config,
        device=device,
    )
    multilinear.finalize_init()

    input_tensor = torch.rand(size=(4, in_features), device=device, dtype=dtype)

    q = linears[0](input_tensor)
    k = linears[1](input_tensor)
    v = linears[2](input_tensor)

    q_m, k_m, v_m = multilinear(input_tensor)

    tensors_equal(q, q_m)
    tensors_equal(k, k_m)
    tensors_equal(v, v_m)


# TODO tests with bias and no bias
# TODO tests with input norm
# TODO tests with other data types


def test_replace_back():
    dtype = torch.float16
    device = "cpu"
    in_features = 4096
    linears = random_linears(
        in_features=in_features,
        out_features=[1024, 2048, 4096],
        device=device,
        dtype=dtype,
    )

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    multilinear = MuiMultiLinear.replace(
        prev_modules=linears,
        engine_config=engine_config,
        device=device,
    )

    replaced_back_linears, _ = multilinear.replace_back()

    input_tensor = torch.rand(size=(4, in_features), device=device, dtype=dtype)

    q = linears[0](input_tensor)
    k = linears[1](input_tensor)
    v = linears[2](input_tensor)

    q_r = replaced_back_linears[0](input_tensor)
    k_r = replaced_back_linears[1](input_tensor)
    v_r = replaced_back_linears[2](input_tensor)

    tensors_equal(q, q_r)
    tensors_equal(k, k_r)
    tensors_equal(v, v_r)
