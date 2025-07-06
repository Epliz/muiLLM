from typing import List
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
import torch
import torch.nn as nn


from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from transformers.models.llama.modeling_llama import LlamaRMSNorm


from transformers.models.llama4.modeling_llama4 import Llama4TextRMSNorm

from muillm.modules.norm.rmsnorm import MuiRMSNorm
from muillm.replacement.replacementcontext import MuiReplacementContext
from .test_utils import tensors_equal


def random_mistral_rmsnorm(
    hidden_size: int,
    eps=1e-5,
    dtype=torch.float16,
    device="cpu",
) -> MistralRMSNorm:
    norm = MistralRMSNorm(hidden_size=hidden_size, eps=eps)
    norm = norm.to(dtype=dtype, device=device)

    # initialize weights
    torch.nn.init.uniform_(norm.weight)

    return norm


def copy_mistral_rmsnorm(norm: MistralRMSNorm) -> MistralRMSNorm:
    hidden_size = norm.weight.shape[0]
    new_norm = MistralRMSNorm(hidden_size, eps=norm.variance_epsilon)

    new_norm.weight = nn.Parameter(norm.weight.clone().detach())

    return new_norm


def _test_basic_mistral_rmsnorm(dtype: torch.dtype, device: str):
    hidden_size = 256
    norm = random_mistral_rmsnorm(hidden_size=hidden_size, dtype=dtype, device=device)

    # replace destroys the passed linear module so we need to copy it
    norm_copy = copy_mistral_rmsnorm(norm)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )
    muinorm = MuiRMSNorm.replace(
        replacement_context=replacement_context,
        prev_module=norm_copy,
    )

    input_tensor = torch.rand(size=(4, hidden_size), device=device, dtype=dtype)

    y = norm(input_tensor)

    y_m = muinorm(input_tensor)

    tensors_equal(y, y_m)


def test_basic_mistral_rmsnorm_fp32_cpu():
    _test_basic_mistral_rmsnorm(dtype=torch.float32, device="cpu")


def test_basic_mistral_rmsnorm_fp32_gpu():
    _test_basic_mistral_rmsnorm(dtype=torch.float32, device="cuda")


def test_basic_mistral_rmsnorm_fp16_gpu():
    _test_basic_mistral_rmsnorm(dtype=torch.float16, device="cuda")


def test_basic_mistral_rmsnorm_bf16_gpu():
    _test_basic_mistral_rmsnorm(dtype=torch.bfloat16, device="cuda")


def random_llama3_rmsnorm(hidden_size: int, eps=1e-5) -> LlamaRMSNorm:
    norm = LlamaRMSNorm(hidden_size=hidden_size, eps=eps)

    # initialize weights
    torch.nn.init.uniform_(norm.weight)

    return norm


def copy_llama3_rmsnorm(norm: LlamaRMSNorm) -> LlamaRMSNorm:
    hidden_size = norm.weight.shape[0]
    new_norm = LlamaRMSNorm(hidden_size, eps=norm.variance_epsilon)

    new_norm.weight = nn.Parameter(norm.weight.clone().detach())

    return new_norm


def test_basic_llama3_rmsnorm():
    hidden_size = 256
    norm = random_llama3_rmsnorm(hidden_size=hidden_size)

    # replace destroys the passed linear module so we need to copy it
    norm_copy = copy_llama3_rmsnorm(norm)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device="cpu",
    )
    muinorm = MuiRMSNorm.replace(
        replacement_context=replacement_context,
        prev_module=norm_copy,
    )

    input_tensor = torch.rand(size=(4, hidden_size))

    y = norm(input_tensor)

    y_m = muinorm(input_tensor)

    tensors_equal(y, y_m)


def random_llama4_rmsnorm(hidden_size: int, eps=1e-5) -> Llama4TextRMSNorm:
    norm = Llama4TextRMSNorm(hidden_size=hidden_size, eps=eps)

    # initialize weights
    torch.nn.init.uniform_(norm.weight)

    return norm


def copy_llama4_rmsnorm(norm: Llama4TextRMSNorm) -> Llama4TextRMSNorm:
    hidden_size = norm.weight.shape[0]
    new_norm = Llama4TextRMSNorm(hidden_size, eps=norm.eps)

    new_norm.weight = nn.Parameter(norm.weight.clone().detach())

    return new_norm


def test_basic_llama4_rmsnorm():
    hidden_size = 256
    norm = random_llama4_rmsnorm(hidden_size=hidden_size)

    # replace destroys the passed linear module so we need to copy it
    norm_copy = copy_llama4_rmsnorm(norm)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device="cpu",
    )
    muinorm = MuiRMSNorm.replace(
        replacement_context=replacement_context,
        prev_module=norm_copy,
    )

    input_tensor = torch.rand(size=(4, hidden_size))

    y = norm(input_tensor)

    y_m = muinorm(input_tensor)

    tensors_equal(y, y_m)


# TODO tests with other data types
