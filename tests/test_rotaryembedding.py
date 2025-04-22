from typing import List
from muillm.engineconfig import MuiEngineConfig
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


def random_mistral_rotary(max_position_embeddings: int) -> MistralRotaryEmbedding:
    config = MistralConfig(
        max_position_embeddings=max_position_embeddings,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    rotary_emb = MistralRotaryEmbedding(config)

    return rotary_emb


def copy_mistral_rotary(
    rotary_emb: MistralRotaryEmbedding,
) -> MistralRotaryEmbedding:
    new_rotary_emb = MistralRotaryEmbedding(config=rotary_emb.config)

    return new_rotary_emb


def test_basic_mistral_rotary():
    max_position_embeddings = 256
    rotary_emb = random_mistral_rotary(max_position_embeddings=max_position_embeddings)

    # replace destroys the passed linear module so we need to copy it
    rotary_emb_copy = copy_mistral_rotary(rotary_emb)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    mui_rotary = MuiRotaryEmbedding.replace(
        prev_module=rotary_emb_copy,
        engine_config=engine_config,
        device="cpu",
    )

    input_position_ids = torch.arange(start=0, end=max_position_embeddings).unsqueeze(0)

    # x is just used for the type and device
    x = torch.randn((1, 1, 256), dtype=torch.float32)

    # llama3 returns cos, sin
    cos, sin = rotary_emb(x, input_position_ids)

    cos_m, sin_m = mui_rotary(x, input_position_ids)

    tensors_equal(torch.complex(cos, sin), torch.complex(cos_m, sin_m))


def random_llama3_rotary(max_position_embeddings: int) -> LlamaRotaryEmbedding:
    config = LlamaConfig(
        max_position_embeddings=max_position_embeddings,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    rotary_emb = LlamaRotaryEmbedding(config)

    return rotary_emb


def copy_llama3_rotary(
    rotary_emb: Llama4TextRotaryEmbedding,
) -> LlamaRotaryEmbedding:
    new_rotary_emb = LlamaRotaryEmbedding(config=rotary_emb.config)

    return new_rotary_emb


def test_basic_llama3_rotary():
    max_position_embeddings = 256
    rotary_emb = random_llama3_rotary(max_position_embeddings=max_position_embeddings)

    # replace destroys the passed linear module so we need to copy it
    rotary_emb_copy = copy_llama3_rotary(rotary_emb)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    mui_rotary = MuiRotaryEmbedding.replace(
        prev_module=rotary_emb_copy,
        engine_config=engine_config,
        device="cpu",
    )

    input_position_ids = torch.arange(start=0, end=max_position_embeddings).unsqueeze(0)

    # x is just used for the type and device
    x = torch.randn((1, 1, 256), dtype=torch.float32)

    # llama3 returns cos, sin
    cos, sin = rotary_emb(x, input_position_ids)

    cos_m, sin_m = mui_rotary(x, input_position_ids)

    tensors_equal(torch.complex(cos, sin), torch.complex(cos_m, sin_m))


def random_llama4_rotary(max_position_embeddings: int) -> Llama4TextRotaryEmbedding:
    config = Llama4TextConfig(
        max_position_embeddings=max_position_embeddings,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    rotary_emb = Llama4TextRotaryEmbedding(config)

    return rotary_emb


def copy_llama4_rotary(
    rotary_emb: Llama4TextRotaryEmbedding,
) -> Llama4TextRotaryEmbedding:
    new_rotary_emb = Llama4TextRotaryEmbedding(config=rotary_emb.config)

    return new_rotary_emb


def test_basic_llama4_rotary():
    max_position_embeddings = 256
    rotary_emb = random_llama4_rotary(max_position_embeddings=max_position_embeddings)

    # replace destroys the passed linear module so we need to copy it
    rotary_emb_copy = copy_llama4_rotary(rotary_emb)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    mui_rotary = MuiRotaryEmbedding.replace(
        prev_module=rotary_emb_copy,
        engine_config=engine_config,
        device="cpu",
    )

    input_position_ids = torch.arange(start=0, end=max_position_embeddings).unsqueeze(0)

    # x is just used for the type and device
    x = torch.randn((1, 1, 256), dtype=torch.float32)

    # llama4 returns a complex tensor
    z = rotary_emb(x, input_position_ids)

    z_m = mui_rotary(x, input_position_ids)

    tensors_equal(z, z_m)


# TODO tests with bias and no bias
# TODO tests with input norm
# TODO tests with other data types
