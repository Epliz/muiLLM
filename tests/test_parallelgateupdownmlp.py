from typing import List
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
import torch
import torch.nn as nn


from transformers.models.mistral.modeling_mistral import MistralMLP
from transformers.models.mistral.configuration_mistral import MistralConfig

from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from transformers.models.llama4.modeling_llama4 import Llama4TextMLP
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

from muillm.replacement.replacementcontext import MuiReplacementContext

from .test_utils import execute_distributed, tensors_equal


def random_mistral_mlp(
    hidden_size: int,
    intermediate_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> MistralMLP:
    config = MistralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    mlp = MistralMLP(config)

    # initialize weights
    torch.nn.init.xavier_uniform_(mlp.gate_proj.weight)
    torch.nn.init.xavier_uniform_(mlp.up_proj.weight)
    torch.nn.init.xavier_uniform_(mlp.down_proj.weight)

    mlp = mlp.to(device=device, dtype=dtype)

    return mlp


def copy_mistral_mlp(mlp: MistralMLP) -> MistralMLP:
    new_mlp = MistralMLP(config=mlp.config)

    new_mlp.gate_proj.weight = nn.Parameter(mlp.gate_proj.weight.clone().detach())
    new_mlp.up_proj.weight = nn.Parameter(mlp.up_proj.weight.clone().detach())
    new_mlp.down_proj.weight = nn.Parameter(mlp.down_proj.weight.clone().detach())

    return new_mlp


def _test_basic_mistral_mlp_inner(
    device: str = "cuda", dtype: torch.dtype = torch.float32
):
    hidden_size = 256
    mlp = random_mistral_mlp(
        hidden_size=hidden_size, intermediate_size=1024, device=device, dtype=dtype
    )

    # replace destroys the passed linear module so we need to copy it
    mlp_copy = copy_mistral_mlp(mlp)

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )
    muimlp = MuiGateUpDownMLP.replace(
        replacement_context=replacement_context,
        prev_module=mlp_copy,
    )
    muimlp.finalize_init()

    input_tensor = torch.rand(size=(4, hidden_size), device=device, dtype=dtype)

    y = mlp(input_tensor)

    y_m = muimlp(input_tensor)

    tensors_equal(y, y_m)


def _test_basic_mistral_mlp(device: str = "cuda", dtype: torch.dtype = torch.float32):
    execute_distributed(_test_basic_mistral_mlp_inner, device=device, dtype=dtype)


def test_basic_mistral_mlp_fp32_gpu():
    _test_basic_mistral_mlp(device="cuda", dtype=torch.float32)


def test_basic_mistral_mlp_fp16_gpu():
    _test_basic_mistral_mlp(device="cuda", dtype=torch.float16)


def test_basic_mistral_mlp_bf16_gpu():
    _test_basic_mistral_mlp(device="cuda", dtype=torch.bfloat16)


def random_llama3_mlp(
    hidden_size: int,
    intermediate_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LlamaMLP:
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    mlp = LlamaMLP(config)

    mlp = mlp.to(device=device, dtype=dtype)

    # initialize weights
    torch.nn.init.xavier_uniform_(mlp.gate_proj.weight)
    torch.nn.init.xavier_uniform_(mlp.up_proj.weight)
    torch.nn.init.xavier_uniform_(mlp.down_proj.weight)

    return mlp


def copy_llama3_mlp(mlp: LlamaMLP) -> LlamaMLP:
    new_mlp = LlamaMLP(config=mlp.config)

    new_mlp.gate_proj.weight = nn.Parameter(mlp.gate_proj.weight.clone().detach())
    new_mlp.up_proj.weight = nn.Parameter(mlp.up_proj.weight.clone().detach())
    new_mlp.down_proj.weight = nn.Parameter(mlp.down_proj.weight.clone().detach())

    return new_mlp


def _test_basic_llama3_mlp_inner(
    device: str = "cuda", dtype: torch.dtype = torch.float32
):
    hidden_size = 256
    mlp = random_llama3_mlp(
        hidden_size=hidden_size, intermediate_size=1024, device=device, dtype=dtype
    )

    # replace destroys the passed linear module so we need to copy it
    mlp_copy = copy_llama3_mlp(mlp)

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )
    muimlp = MuiGateUpDownMLP.replace(
        replacement_context=replacement_context,
        prev_module=mlp_copy,
    )
    muimlp.finalize_init()

    input_tensor = torch.rand(size=(4, hidden_size), device=device, dtype=dtype)

    y = mlp(input_tensor)

    y_m = muimlp(input_tensor)

    tensors_equal(y, y_m)


def _test_basic_llama3_mlp(device: str = "cuda", dtype: torch.dtype = torch.float32):
    execute_distributed(_test_basic_llama3_mlp_inner, device=device, dtype=dtype)


def test_basic_llama3_mlp_gpu_fp32():
    _test_basic_llama3_mlp(device="cuda", dtype=torch.float32)


def test_basic_llama3_mlp_gpu_fp16():
    _test_basic_llama3_mlp(device="cuda", dtype=torch.float16)


def test_basic_llama3_mlp_gpu_bf16():
    _test_basic_llama3_mlp(device="cuda", dtype=torch.bfloat16)


def random_llama4_mlp(
    hidden_size: int,
    intermediate_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Llama4TextMLP:
    config = Llama4TextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    mlp = Llama4TextMLP(config)

    mlp = mlp.to(device=device, dtype=dtype)

    # initialize weights
    torch.nn.init.xavier_uniform_(mlp.gate_proj.weight)
    torch.nn.init.xavier_uniform_(mlp.up_proj.weight)
    torch.nn.init.xavier_uniform_(mlp.down_proj.weight)

    return mlp


def copy_llama4_mlp(mlp: Llama4TextMLP) -> Llama4TextMLP:
    new_mlp = Llama4TextMLP(config=mlp.config)

    new_mlp.gate_proj.weight = nn.Parameter(mlp.gate_proj.weight.clone().detach())
    new_mlp.up_proj.weight = nn.Parameter(mlp.up_proj.weight.clone().detach())
    new_mlp.down_proj.weight = nn.Parameter(mlp.down_proj.weight.clone().detach())

    return new_mlp


def _test_basic_llama4_mlp_inner(
    device: str = "cuda", dtype: torch.dtype = torch.float32
):
    hidden_size = 256
    mlp = random_llama4_mlp(
        hidden_size=hidden_size, intermediate_size=1024, device=device, dtype=dtype
    )

    # replace destroys the passed linear module so we need to copy it
    mlp_copy = copy_llama4_mlp(mlp)

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )
    muimlp = MuiGateUpDownMLP.replace(
        replacement_context=replacement_context,
        prev_module=mlp_copy,
    )
    muimlp.finalize_init()

    input_tensor = torch.rand(size=(4, hidden_size), device=device, dtype=dtype)

    y = mlp(input_tensor)

    y_m = muimlp(input_tensor)

    tensors_equal(y, y_m)


def test_basic_llama4_mlp(device: str = "cuda", dtype: torch.dtype = torch.float32):
    execute_distributed(_test_basic_llama4_mlp_inner, device=device, dtype=dtype)


def test_basic_llama4_mlp_fp32_gpu():
    test_basic_llama4_mlp(device="cuda", dtype=torch.float32)


def test_basic_llama4_mlp_fp16_gpu():
    test_basic_llama4_mlp(device="cuda", dtype=torch.float16)


def test_basic_llama4_mlp_bf16_gpu():
    test_basic_llama4_mlp(device="cuda", dtype=torch.bfloat16)


def random_gemma3_mlp(
    hidden_size: int, intermediate_size: int, device: str, dtype: torch.dtype
) -> Gemma3MLP:
    config = Gemma3TextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=16,
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    mlp = Gemma3MLP(config)

    mlp = mlp.to(device=device, dtype=dtype)

    # initialize weights
    torch.nn.init.xavier_uniform_(mlp.gate_proj.weight)
    torch.nn.init.xavier_uniform_(mlp.up_proj.weight)
    torch.nn.init.xavier_uniform_(mlp.down_proj.weight)

    return mlp


def copy_gemma3_mlp(mlp: Gemma3MLP) -> Gemma3MLP:
    new_mlp = Gemma3MLP(config=mlp.config)

    new_mlp.gate_proj.weight = nn.Parameter(mlp.gate_proj.weight.clone().detach())
    new_mlp.up_proj.weight = nn.Parameter(mlp.up_proj.weight.clone().detach())
    new_mlp.down_proj.weight = nn.Parameter(mlp.down_proj.weight.clone().detach())

    return new_mlp


def _test_basic_gemma3_mlp_inner(
    device: str = "cuda", dtype: torch.dtype = torch.float32
):
    hidden_size = 256
    mlp = random_gemma3_mlp(
        hidden_size=hidden_size, intermediate_size=1024, device=device, dtype=dtype
    )

    # replace destroys the passed linear module so we need to copy it
    mlp_copy = copy_gemma3_mlp(mlp)

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )
    muimlp = MuiGateUpDownMLP.replace(
        replacement_context=replacement_context,
        prev_module=mlp_copy,
    )
    muimlp.finalize_init()

    input_tensor = torch.rand(size=(4, hidden_size), device=device, dtype=dtype)

    y = mlp(input_tensor)

    y_m = muimlp(input_tensor)

    tensors_equal(y, y_m)


def _test_basic_gemma3_mlp(device: str = "cuda", dtype: torch.dtype = torch.float32):
    execute_distributed(_test_basic_gemma3_mlp_inner, device=device, dtype=dtype)


def test_basic_gemma3_mlp_fp32_gpu():
    _test_basic_gemma3_mlp(device="cuda", dtype=torch.float32)


def test_basic_gemma3_mlp_fp16_gpu():
    _test_basic_gemma3_mlp(device="cuda", dtype=torch.float16)


def test_basic_gemma3_mlp_bf16_gpu():
    _test_basic_gemma3_mlp(device="cuda", dtype=torch.bfloat16)


# TODO tests with bias and no bias
# TODO tests with input norm
