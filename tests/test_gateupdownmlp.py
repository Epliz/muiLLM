from typing import List
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
import torch
import torch.nn as nn


from transformers.models.mistral.modeling_mistral import MistralMLP
from transformers.models.mistral.configuration_mistral import MistralConfig

from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.configuration_llama import LlamaConfig


from transformers.models.llama4.modeling_llama4 import Llama4TextMLP
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

from .test_utils import tensors_equal


def random_mistral_mlp(
    hidden_size: int,
    intermediate_size: int,
    device: str = "cpu",
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


def _test_basic_mistral_mlp(device: str = "cpu", dtype: torch.dtype = torch.float32):
    hidden_size = 256
    mlp = random_mistral_mlp(
        hidden_size=hidden_size, intermediate_size=1024, device=device, dtype=dtype
    )

    # replace destroys the passed linear module so we need to copy it
    mlp_copy = copy_mistral_mlp(mlp)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    muimlp = MuiGateUpDownMLP.replace(
        prev_module=mlp_copy,
        engine_config=engine_config,
        device=device,
    )
    muimlp.finalize_init()

    input_tensor = torch.rand(size=(4, hidden_size), device=device, dtype=dtype)

    y = mlp(input_tensor)

    y_m = muimlp(input_tensor)

    tensors_equal(y, y_m)


def test_basic_mistral_mlp_fp32_cpu():
    _test_basic_mistral_mlp(device="cpu", dtype=torch.float32)


def test_basic_mistral_mlp_fp32_gpu():
    _test_basic_mistral_mlp(device="cuda", dtype=torch.float32)


def test_basic_mistral_mlp_fp16_gpu():
    _test_basic_mistral_mlp(device="cuda", dtype=torch.float16)


def test_basic_mistral_mlp_bf16_gpu():
    _test_basic_mistral_mlp(device="cuda", dtype=torch.bfloat16)


def random_llama3_mlp(hidden_size: int, intermediate_size: int) -> LlamaMLP:
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


def test_basic_llama3_mlp():
    hidden_size = 256
    mlp = random_llama3_mlp(hidden_size=hidden_size, intermediate_size=1024)

    # replace destroys the passed linear module so we need to copy it
    mlp_copy = copy_llama3_mlp(mlp)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    muimlp = MuiGateUpDownMLP.replace(
        prev_module=mlp_copy,
        engine_config=engine_config,
        device="cpu",
    )
    muimlp.finalize_init()

    input_tensor = torch.rand(size=(4, hidden_size))

    y = mlp(input_tensor)

    y_m = muimlp(input_tensor)

    tensors_equal(y, y_m)


def random_llama4_mlp(hidden_size: int, intermediate_size: int) -> Llama4TextMLP:
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


def test_basic_llama4_mlp():
    hidden_size = 256
    mlp = random_llama4_mlp(hidden_size=hidden_size, intermediate_size=1024)

    # replace destroys the passed linear module so we need to copy it
    mlp_copy = copy_llama4_mlp(mlp)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    muimlp = MuiGateUpDownMLP.replace(
        prev_module=mlp_copy,
        engine_config=engine_config,
        device="cpu",
    )
    muimlp.finalize_init()

    input_tensor = torch.rand(size=(4, hidden_size))

    y = mlp(input_tensor)

    y_m = muimlp(input_tensor)

    tensors_equal(y, y_m)


# TODO tests with bias and no bias
# TODO tests with input norm
