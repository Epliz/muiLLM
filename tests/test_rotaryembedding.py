from typing import List, Tuple
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.attention.llama4attention import apply_rotary_emb
from muillm.modules.attention.rotaryembedding import MuiRotaryEmbedding
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
import torch
import torch.nn as nn


from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
from transformers.models.mistral.configuration_mistral import MistralConfig

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig


from transformers.models.llama4.modeling_llama4 import Llama4TextRotaryEmbedding
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

from .test_utils import tensors_equal


def random_mistral_rotary(
    max_position_embeddings: int, device: str, dtype: torch.dtype
) -> MistralRotaryEmbedding:
    config = MistralConfig(
        max_position_embeddings=max_position_embeddings,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    rotary_emb = MistralRotaryEmbedding(config, device=device)

    return rotary_emb


def copy_mistral_rotary(
    rotary_emb: MistralRotaryEmbedding,
) -> MistralRotaryEmbedding:
    new_rotary_emb = MistralRotaryEmbedding(
        config=rotary_emb.config, device=rotary_emb.inv_freq.device
    )

    return new_rotary_emb


def _test_basic_mistral_rotary(device: str, dtype: torch.dtype):
    max_position_embeddings = 256
    rotary_emb = random_mistral_rotary(
        max_position_embeddings=max_position_embeddings, device=device, dtype=dtype
    )

    # replace destroys the passed linear module so we need to copy it
    rotary_emb_copy = copy_mistral_rotary(rotary_emb)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    mui_rotary = MuiRotaryEmbedding.replace(
        prev_module=rotary_emb_copy,
        engine_config=engine_config,
        device=device,
    )

    input_position_ids = torch.arange(
        start=0, end=max_position_embeddings, device=device
    ).unsqueeze(0)

    # x is just used for the type and device
    x = torch.randn((1, 1, 256), device=device, dtype=dtype)

    # mistral returns cos, sin
    cos, sin = rotary_emb(x, input_position_ids)

    cos_m, sin_m = mui_rotary(x, input_position_ids)

    tensors_equal(cos, cos_m)
    tensors_equal(sin, sin_m)


def test_basic_mistral_rotary_fp32_cpu():
    _test_basic_mistral_rotary(device="cpu", dtype=torch.float32)


def test_basic_mistral_rotary_fp32_gpu():
    _test_basic_mistral_rotary(device="cuda", dtype=torch.float32)


def test_basic_mistral_rotary_fp16_gpu():
    _test_basic_mistral_rotary(device="cuda", dtype=torch.float16)


def test_basic_mistral_rotary_bf16_gpu():
    _test_basic_mistral_rotary(device="cuda", dtype=torch.bfloat16)


def random_llama3_rotary(
    max_position_embeddings: int, device: str, dtype: torch.dtype
) -> LlamaRotaryEmbedding:
    config = LlamaConfig(
        max_position_embeddings=max_position_embeddings,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    rotary_emb = LlamaRotaryEmbedding(config, device=device)

    return rotary_emb


def copy_llama3_rotary(
    rotary_emb: Llama4TextRotaryEmbedding,
) -> LlamaRotaryEmbedding:
    new_rotary_emb = LlamaRotaryEmbedding(
        config=rotary_emb.config, device=rotary_emb.inv_freq.device
    )

    return new_rotary_emb


def _test_basic_llama3_rotary(device: str, dtype: torch.dtype):
    max_position_embeddings = 256
    rotary_emb = random_llama3_rotary(
        max_position_embeddings=max_position_embeddings, device=device, dtype=dtype
    )

    # replace destroys the passed linear module so we need to copy it
    rotary_emb_copy = copy_llama3_rotary(rotary_emb)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    mui_rotary = MuiRotaryEmbedding.replace(
        prev_module=rotary_emb_copy,
        engine_config=engine_config,
        device=device,
    )

    input_position_ids = torch.arange(
        start=0, end=max_position_embeddings, device=device
    ).unsqueeze(0)

    # x is just used for the type and device
    x = torch.randn((1, 1, 256), device=device, dtype=dtype)

    # llama3 returns cos, sin
    cos, sin = rotary_emb(x, input_position_ids)

    cos_m, sin_m = mui_rotary(x, input_position_ids)

    tensors_equal(cos, cos_m)
    tensors_equal(sin, sin_m)


def test_basic_llama3_rotary_fp32_cpu():
    _test_basic_llama3_rotary(device="cpu", dtype=torch.float32)


def test_basic_llama3_rotary_fp32_gpu():
    _test_basic_llama3_rotary(device="cuda", dtype=torch.float32)


def test_basic_llama3_rotary_fp16_gpu():
    _test_basic_llama3_rotary(device="cuda", dtype=torch.float16)


def test_basic_llama3_rotary_bf16_gpu():
    _test_basic_llama3_rotary(device="cuda", dtype=torch.bfloat16)


def random_llama4_rotary(
    max_position_embeddings: int, device: str, dtype: torch.dtype
) -> Llama4TextRotaryEmbedding:
    config = Llama4TextConfig(
        max_position_embeddings=max_position_embeddings,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    rotary_emb = Llama4TextRotaryEmbedding(config, device=device)

    return rotary_emb


def copy_llama4_rotary(
    rotary_emb: Llama4TextRotaryEmbedding,
) -> Llama4TextRotaryEmbedding:
    new_rotary_emb = Llama4TextRotaryEmbedding(
        config=rotary_emb.config, device=rotary_emb.inv_freq.device
    )

    return new_rotary_emb


def _test_basic_llama4_rotary(device: str, dtype: torch.dtype):
    max_position_embeddings = 256
    rotary_emb = random_llama4_rotary(
        max_position_embeddings=max_position_embeddings, device=device, dtype=dtype
    )

    # replace destroys the passed linear module so we need to copy it
    rotary_emb_copy = copy_llama4_rotary(rotary_emb)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    mui_rotary = MuiRotaryEmbedding.replace(
        prev_module=rotary_emb_copy,
        engine_config=engine_config,
        device=device,
    )

    input_position_ids = torch.arange(
        start=0, end=max_position_embeddings, device=device
    ).unsqueeze(0)

    # x is just used for the type and device
    x = torch.randn((1, 1, 256), device=device, dtype=dtype)

    # llama4 returns a complex tensor of floats
    z = rotary_emb(x, input_position_ids)

    z_m = mui_rotary(x, input_position_ids)

    tensors_equal(z, z_m)


def test_basic_llama4_rotary_fp32_cpu():
    _test_basic_llama4_rotary(device="cpu", dtype=torch.float32)


def test_basic_llama4_rotary_fp32_gpu():
    _test_basic_llama4_rotary(device="cuda", dtype=torch.float32)


def test_basic_llama4_rotary_fp16_gpu():
    _test_basic_llama4_rotary(device="cuda", dtype=torch.float16)


def test_basic_llama4_rotary_bf16_gpu():
    _test_basic_llama4_rotary(device="cuda", dtype=torch.bfloat16)


# copied from Llama4, added fix for bfloat16
def ref_apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, None, :, :]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, None, :, :]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _test_apply_rotary_emb(device: str, dtype: torch.dtype):
    T = 5
    B = 1
    H = 128
    num_q_heads = 5
    num_k_heads = 10

    # x is just used for the type and device
    xq = torch.randn((B, num_q_heads, T, H), dtype=dtype, device=device)
    xk = torch.randn((B, num_k_heads, T, H), dtype=dtype, device=device)

    freqs_cis = torch.randn((B, T, H), dtype=torch.float, device=device)

    freqs_cis = torch.view_as_complex(freqs_cis.reshape(*freqs_cis.shape[:-1], -1, 2))

    xq_out, xk_out = ref_apply_rotary_emb(xq, xk, freqs_cis)

    xq_out_m, xk_out_m = apply_rotary_emb(xq, xk, freqs_cis)

    tensors_equal(xq_out, xq_out_m)
    tensors_equal(xk_out, xk_out_m)


def test_apply_rotary_emb_fp32_cpu():
    _test_apply_rotary_emb(device="cpu", dtype=torch.float32)


def test_apply_rotary_emb_fp32_gpu():
    _test_apply_rotary_emb(device="cuda", dtype=torch.float32)


def test_apply_rotary_emb_fp16_gpu():
    _test_apply_rotary_emb(device="cuda", dtype=torch.float16)


def test_apply_rotary_emb_bf16_gpu():
    _test_apply_rotary_emb(device="cuda", dtype=torch.bfloat16)
