from typing import List
from muillm.engineconfig import MuiEngineConfig
import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.configuration_llama import LlamaConfig

from muillm.modules.kvcache.cache_utils import MuiDynamicCache

from .test_utils import tensors_equal

from transformers.cache_utils import DynamicCache


def llama3_model_config(hidden_size: int, intermediate_size: int) -> LlamaMLP:
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    return config


def _test_dynamic_kv_cache(batch_size: int, dtype: torch.dtype, device: str):
    max_cache_len = 128
    hidden_size = 256
    model_config = llama3_model_config(hidden_size=hidden_size, intermediate_size=1024)

    engine_config = MuiEngineConfig(tensor_parallelism=1)

    hf_cache = DynamicCache()
    cache = MuiDynamicCache(engine_config=engine_config)

    # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
    head_dim = (
        model_config.head_dim
        if hasattr(model_config, "head_dim")
        else model_config.hidden_size // model_config.num_attention_heads
    )

    num_key_value_heads = (
        model_config.num_attention_heads
        if getattr(model_config, "num_key_value_heads", None) is None
        else model_config.num_key_value_heads
    )

    num_layers = model_config.num_hidden_layers

    # Prefill
    prefill_size = 64

    for l in range(num_layers):
        prefill_input_tensor = torch.rand(
            size=(batch_size, num_key_value_heads, prefill_size, head_dim),
            device=device,
            dtype=dtype,
        )

        cache_position = torch.arange(
            start=0, end=prefill_size, device=device, dtype=torch.int64
        )

        k_out, v_out = cache.update(
            key_states=prefill_input_tensor,
            value_states=prefill_input_tensor,
            layer_idx=l,
            cache_kwargs={"cache_position": cache_position},
        )

        hf_k_out, hf_v_out = hf_cache.update(
            key_states=prefill_input_tensor,
            value_states=prefill_input_tensor,
            layer_idx=l,
            cache_kwargs={"cache_position": cache_position},
        )

        tensors_equal(hf_k_out, k_out)
        tensors_equal(hf_v_out, v_out)

    current_seq_length = prefill_size
    assert current_seq_length == cache.get_seq_length(layer_idx=0)

    # Decode
    decode_size = 1

    for _ in range(16):
        for l in range(num_layers):
            decode_input_tensor = torch.rand(
                size=(batch_size, num_key_value_heads, decode_size, head_dim),
                device=device,
                dtype=dtype,
            )

            cache_position = torch.arange(
                start=current_seq_length,
                end=current_seq_length + decode_size,
                device=device,
                dtype=torch.int64,
            )

            k_out, v_out = cache.update(
                key_states=decode_input_tensor,
                value_states=decode_input_tensor,
                layer_idx=l,
                cache_kwargs={"cache_position": cache_position},
            )

            hf_k_out, hf_v_out = hf_cache.update(
                key_states=decode_input_tensor,
                value_states=decode_input_tensor,
                layer_idx=l,
                cache_kwargs={"cache_position": cache_position},
            )

            tensors_equal(hf_k_out, k_out)
            tensors_equal(hf_v_out, v_out)

        current_seq_length = current_seq_length + decode_size
        assert current_seq_length == cache.get_seq_length(layer_idx=0)


def test_dynamic_kv_cache_batch_size1_fp32_cpu():
    dtype = torch.float32
    device = "cpu"
    batch_size = 1
    _test_dynamic_kv_cache(batch_size=batch_size, dtype=dtype, device=device)


def test_dynamic_kv_cache_batch_size4_fp32_cpu():
    dtype = torch.float32
    device = "cpu"
    batch_size = 4
    _test_dynamic_kv_cache(batch_size=batch_size, dtype=dtype, device=device)


def test_dynamic_kv_cache_batch_size1_fp32_gpu():
    dtype = torch.float32
    device = "cuda"
    batch_size = 1
    _test_dynamic_kv_cache(batch_size=batch_size, dtype=dtype, device=device)


def test_dynamic_kv_cache_batch_size4_fp32_gpu():
    dtype = torch.float32
    device = "cuda"
    batch_size = 4
    _test_dynamic_kv_cache(batch_size=batch_size, dtype=dtype, device=device)


def test_dynamic_kv_cache_batch_size1_fp16_gpu():
    dtype = torch.float16
    device = "cuda"
    batch_size = 1
    _test_dynamic_kv_cache(batch_size=batch_size, dtype=dtype, device=device)


def test_dynamic_kv_cache_batch_size4_fp16_gpu():
    dtype = torch.float16
    device = "cuda"
    batch_size = 4
    _test_dynamic_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
