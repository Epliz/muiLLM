from typing import List
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.linear import MuiLinear
import torch
import torch.nn as nn

from muillm.modules.multilinear import MuiMultiLinear
from muillm.replacement.replacementcontext import MuiReplacementContext

from .test_utils import copy_linear, random_linear, tensors_equal


def _test_basic_linear(batch_size: int, in_features: int, device: str = "cpu", dtype=torch.float16, sharded: bool = False):
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
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )
    muilinear = MuiLinear.replace(
        replacement_context=replacement_context,
        prev_module=linear_copy,
    )
    muilinear.finalize_init()

    if sharded:
        input_tensor = torch.rand(size=(batch_size, in_features * 4), device=device, dtype=dtype)
        input_tensor = input_tensor[..., :in_features]  # Simulate sharding by slicing the input tensor
    else:
        input_tensor = torch.rand(size=(batch_size, in_features), device=device, dtype=dtype)

    y = linear(input_tensor)

    linear.to(device=device, dtype=torch.float32)
    y_highres = linear(input_tensor.to(torch.float32)).to(dtype)

    y_m = muilinear(input_tensor)

    tensors_equal(y, y_m, y_highres=y_highres)

def test_basic_linear_b1_fp32_cpu():
    device = "cpu"
    in_features = 2048
    _test_basic_linear(batch_size=1, in_features=in_features, device=device, dtype=torch.float32)


def test_basic_linear_b1_fp32_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(batch_size=1, in_features=in_features, device=device, dtype=torch.float32)


def test_basic_linear_b1_fp16_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(batch_size=1, in_features=in_features, device=device, dtype=torch.float16)


def test_basic_linear_b1_bf16_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(batch_size=1, in_features=in_features, device=device, dtype=torch.bfloat16)

def test_basic_linear_b2_fp16_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(batch_size=2, in_features=in_features, device=device, dtype=torch.float16)


def test_basic_linear_b2_bf16_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(batch_size=2, in_features=in_features, device=device, dtype=torch.bfloat16)

def test_basic_linear_b4_fp32_cpu():
    device = "cpu"
    in_features = 2048
    _test_basic_linear(batch_size=4, in_features=in_features, device=device, dtype=torch.float32)


def test_basic_linear_b4_fp32_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(batch_size=4, in_features=in_features, device=device, dtype=torch.float32)


def test_basic_linear_b4_fp16_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(batch_size=4, in_features=in_features, device=device, dtype=torch.float16)


def test_basic_linear_b4_bf16_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(batch_size=4, in_features=in_features, device=device, dtype=torch.bfloat16)

def test_basic_linear_b8_fp16_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(batch_size=8, in_features=in_features, device=device, dtype=torch.float16)


def test_basic_linear_b8_bf16_gpu():
    device = "cuda"
    in_features = 2048
    _test_basic_linear(batch_size=8, in_features=in_features, device=device, dtype=torch.bfloat16)


def _test_linear_bias(batch_size: int, in_features: int, device: str, dtype=torch.float16, sharded: bool = False):
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
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )
    muilinear = MuiLinear.replace(
        replacement_context=replacement_context,
        prev_module=linear_copy,
    )
    muilinear.finalize_init()

    if sharded:
        input_tensor = torch.rand(size=(batch_size, in_features * 4), device=device, dtype=dtype)
        input_tensor = input_tensor[..., :in_features]  # Simulate sharding by slicing the input tensor
    else:
        input_tensor = torch.rand(size=(batch_size, in_features), device=device, dtype=dtype)

    y = linear(input_tensor)

    linear.to(device=device, dtype=torch.float32)
    y_highres = linear(input_tensor.to(torch.float32)).to(dtype)

    y_m = muilinear(input_tensor)

    tensors_equal(y, y_m, y_highres=y_highres)

def test_linear_bias_b1_fp32_cpu():
    device = "cpu"
    in_features = 2048
    _test_linear_bias(batch_size=1, in_features=in_features, device=device, dtype=torch.float32)


def test_linear_bias_b1_fp32_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=1, in_features=in_features, device=device, dtype=torch.float32)


def test_linear_bias_b1_fp16_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=1, in_features=in_features, device=device, dtype=torch.float16)


def test_linear_bias_b1_bf16_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=1, in_features=in_features, device=device, dtype=torch.bfloat16)

def test_linear_bias_b2_fp16_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=2, in_features=in_features, device=device, dtype=torch.float16)


def test_linear_bias_b2_bf16_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=2, in_features=in_features, device=device, dtype=torch.bfloat16)

def test_linear_bias_b4_fp32_cpu():
    device = "cpu"
    in_features = 2048
    _test_linear_bias(batch_size=4, in_features=in_features, device=device, dtype=torch.float32)


def test_linear_bias_b4_fp32_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=4, in_features=in_features, device=device, dtype=torch.float32)


def test_linear_bias_b4_fp16_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=4, in_features=in_features, device=device, dtype=torch.float16)


def test_linear_bias_b4_bf16_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=4, in_features=in_features, device=device, dtype=torch.bfloat16)

def test_linear_bias_b8_fp16_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=8, in_features=in_features, device=device, dtype=torch.float16)


def test_linear_bias_b8_bf16_gpu():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=8, in_features=in_features, device=device, dtype=torch.bfloat16)

def test_linear_bias_b1_fp32_cpu_sharded():
    device = "cpu"
    in_features = 2048
    _test_linear_bias(batch_size=1, in_features=in_features, device=device, dtype=torch.float32, sharded=True)


def test_linear_bias_b1_fp32_gpu_sharded():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=1, in_features=in_features, device=device, dtype=torch.float32, sharded=True)


def test_linear_bias_b1_fp16_gpu_sharded():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=1, in_features=in_features, device=device, dtype=torch.float16, sharded=True)


def test_linear_bias_b1_bf16_gpu_sharded():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=1, in_features=in_features, device=device, dtype=torch.bfloat16, sharded=True)

def test_linear_bias_b4_fp32_cpu_sharded():
    device = "cpu"
    in_features = 2048
    _test_linear_bias(batch_size=4, in_features=in_features, device=device, dtype=torch.float32, sharded=True)


def test_linear_bias_b4_fp32_gpu_sharded():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=4, in_features=in_features, device=device, dtype=torch.float32, sharded=True)


def test_linear_bias_b4_fp16_gpu_sharded():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=4, in_features=in_features, device=device, dtype=torch.float16, sharded=True)


def test_linear_bias_b4_bf16_gpu_sharded():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=4, in_features=in_features, device=device, dtype=torch.bfloat16, sharded=True)

def test_linear_bias_b8_fp16_gpu_sharded():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=8, in_features=in_features, device=device, dtype=torch.float16, sharded=True)


def test_linear_bias_b8_bf16_gpu_sharded():
    device = "cuda"
    in_features = 2048
    _test_linear_bias(batch_size=8, in_features=in_features, device=device, dtype=torch.bfloat16, sharded=True)

# TODO tests with input norm
