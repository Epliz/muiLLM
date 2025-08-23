from typing import List
from muillm.engineconfig import MuiEngineConfig
import torch
import torch.nn as nn

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.cache_utils import HybridCache

from muillm.modules.kvcache.cache_utils import MuiHybridCache

from .test_utils import tensors_equal


def gemma3_model_config(
    hidden_size: int, intermediate_size: int, sliding_window: int
) -> Gemma3TextConfig:
    config = Gemma3TextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=8,  # enough layers to get some sliding window ones
        num_attention_heads=16,
        sliding_window=sliding_window,
    )

    return config


def _is_sliding_layer(layer_index: int, config: Gemma3TextConfig) -> bool:
    if hasattr(config.get_text_config(), "no_rope_layers"):
        return config.no_rope_layers[layer_index]
    else:
        layer_switch = getattr(config, "sliding_window_pattern", 2)
        return bool((layer_index + 1) % layer_switch)


def _last_chunk(tensor: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if tensor.shape[-1] <= chunk_size:
        return tensor
    return tensor[:, :, -chunk_size:, :]


def _test_hybrid_kv_cache(batch_size: int, dtype: torch.dtype, device: str):
    max_cache_len = 128
    hidden_size = 256
    sliding_window = 32
    model_config = gemma3_model_config(
        hidden_size=hidden_size,
        intermediate_size=1024,
        sliding_window=sliding_window,
    )

    engine_config = MuiEngineConfig(tensor_parallelism=1)

    hf_cache = HybridCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        dtype=dtype,
        device=device,
    )

    cache = MuiHybridCache(
        engine_config=engine_config,
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        dtype=dtype,
        device=device,
        tensor_parallelism=1,
        narrow_output=False,
    )

    num_layers = model_config.num_hidden_layers

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

    # Prefill
    prefill_size = 30  # chosen prefill size to be smaller than sliding_window
    for l in range(num_layers):
        prefill_input_tensor = torch.rand(
            size=(batch_size, num_key_value_heads, prefill_size, head_dim),
            dtype=dtype,
        ).to(
            device=device
        )  # move after for platforms without rand implemented

        cache_position = torch.arange(start=0, end=prefill_size, dtype=torch.int64).to(
            device=device
        )  # move after for platforms without arange implemented

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

        # During prefill, we return the K,V input tensors for the sliding layers
        # but HF returns the cache ones, that are bigger
        tensors_equal(hf_k_out[:, :, :prefill_size, :], k_out[:, :, :prefill_size, :])
        tensors_equal(hf_v_out[:, :, :prefill_size, :], v_out[:, :, :prefill_size, :])

    current_seq_length = prefill_size
    assert current_seq_length == cache.get_seq_length(layer_idx=0)

    # Decode
    decode_size = 1

    for _ in range(16):
        cache_position = torch.arange(
            start=current_seq_length,
            end=current_seq_length + decode_size,
            dtype=torch.int64,
        ).to(
            device=device
        )  # move after for platforms without arange implemented

        for l in range(num_layers):
            decode_input_tensor = torch.rand(
                size=(batch_size, num_key_value_heads, decode_size, head_dim),
                dtype=dtype,
            ).to(
                device=device
            )  # move after for platforms without rand implemented

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


def test_hybrid_kv_cache_batch_size1_fp32_cpu():
    dtype = torch.float32
    device = "cpu"
    batch_size = 1
    _test_hybrid_kv_cache(batch_size=batch_size, dtype=dtype, device=device)


def test_hybrid_kv_cache_batch_size4_fp32_cpu():
    dtype = torch.float32
    device = "cpu"
    batch_size = 4
    _test_hybrid_kv_cache(batch_size=batch_size, dtype=dtype, device=device)


def test_hybrid_kv_cache_batch_size1_fp32_gpu():
    dtype = torch.float32
    device = "cuda"
    batch_size = 1
    _test_hybrid_kv_cache(batch_size=batch_size, dtype=dtype, device=device)


def test_hybrid_kv_cache_batch_size4_fp32_gpu():
    dtype = torch.float32
    device = "cuda"
    batch_size = 4
    _test_hybrid_kv_cache(batch_size=batch_size, dtype=dtype, device=device)


def test_hybrid_kv_cache_batch_size1_fp16_gpu():
    dtype = torch.float16
    device = "cuda"
    batch_size = 1
    _test_hybrid_kv_cache(batch_size=batch_size, dtype=dtype, device=device)


def test_hybrid_kv_cache_batch_size4_fp16_gpu():
    dtype = torch.float16
    device = "cuda"
    batch_size = 4
    _test_hybrid_kv_cache(batch_size=batch_size, dtype=dtype, device=device)


# TODO: test prefill larger than attention_chunk_size
# TODO: test prefill smaller than attention_chunk_size but next chunk going over attention_chunk_size
