from typing import List
from muillm.engineconfig import MuiEngineConfig
import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
from transformers.cache_utils import HybridChunkedCache

from muillm.modules.kvcache.cache_utils import MuiHybridChunkedCache
from muillm.modules.rope.ropeops import apply_complex_rotary_emb, apply_rotary_pos_emb

from .test_utils import tensors_equal


def llama4_model_config(
    hidden_size: int, intermediate_size: int, attention_chunk_size: int
) -> LlamaMLP:
    config = Llama4TextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=8,  # enough layers to get some sliding window ones
        num_attention_heads=16,
        attention_chunk_size=attention_chunk_size,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    return config


def _is_sliding_layer(layer_index: int, config: Llama4TextConfig) -> bool:
    if hasattr(config.get_text_config(), "no_rope_layers"):
        return config.no_rope_layers[layer_index]
    else:
        layer_switch = getattr(config, "sliding_window_pattern", 2)
        return bool((layer_index + 1) % layer_switch)


def _last_chunk(tensor: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if tensor.shape[-1] <= chunk_size:
        return tensor
    return tensor[:, :, -chunk_size:, :]


def _test_hybrid_kv_cache(
    batch_size: int,
    dtype: torch.dtype,
    device: str,
    attention_chunk_size=32,
    prefill_size=30,
):
    max_cache_len = 128
    hidden_size = 256
    model_config = llama4_model_config(
        hidden_size=hidden_size,
        intermediate_size=1024,
        attention_chunk_size=attention_chunk_size,
    )

    engine_config = MuiEngineConfig(tensor_parallelism=1)

    hf_cache = HybridChunkedCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        dtype=dtype,
        device=device,
    )

    cache = MuiHybridChunkedCache(
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
    current_seq_length = prefill_size
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
        tensors_equal(
            hf_k_out[:, :, :current_seq_length, :], k_out[:, :, :current_seq_length, :]
        )
        tensors_equal(
            hf_v_out[:, :, :current_seq_length, :], v_out[:, :, :current_seq_length, :]
        )

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

        current_seq_length = current_seq_length + decode_size

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

            tensors_equal(
                hf_k_out[:, :, :current_seq_length, :],
                k_out[:, :, :current_seq_length, :],
            )
            tensors_equal(
                hf_v_out[:, :, :current_seq_length, :],
                v_out[:, :, :current_seq_length, :],
            )

        assert current_seq_length == cache.get_seq_length(layer_idx=0)


def test_hybrid_kv_cache_batch_size1_fp32_cpu():
    dtype = torch.float32
    device = "cpu"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_batch_size4_fp32_cpu():
    dtype = torch.float32
    device = "cpu"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_batch_size1_fp32_gpu():
    dtype = torch.float32
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_batch_size4_fp32_gpu():
    dtype = torch.float32
    device = "cuda"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_batch_size1_fp16_gpu():
    dtype = torch.float16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_batch_size4_fp16_gpu():
    dtype = torch.float16
    device = "cuda"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_batch_size1_bf16_gpu():
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_batch_size4_bf16_gpu():
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


# large prefill


def test_hybrid_kv_cache_batch_size1_fp32_cpu_large_prefill():
    dtype = torch.float32
    device = "cpu"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_batch_size1_fp32_gpu_large_prefill():
    dtype = torch.float32
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_batch_size1_fp16_gpu_large_prefill():
    dtype = torch.float16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_batch_size1_bf16_gpu_large_prefill():
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48
    _test_hybrid_kv_cache(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def _test_hybrid_kv_cache_rope_update(
    batch_size: int,
    dtype: torch.dtype,
    device: str,
    attention_chunk_size=32,
    prefill_size=30,
):
    max_cache_len = 128
    hidden_size = 256
    model_config = llama4_model_config(
        hidden_size=hidden_size,
        intermediate_size=1024,
        attention_chunk_size=attention_chunk_size,
    )

    engine_config = MuiEngineConfig(tensor_parallelism=1)

    hf_cache = HybridChunkedCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        dtype=dtype,
        device=device,
    )

    cache = MuiHybridChunkedCache(
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
    # chosen prefill size to be smaller than attention_chunk_size
    current_seq_length = prefill_size
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

        q, k, v = prefill_input_tensor, prefill_input_tensor, prefill_input_tensor

        cos = torch.rand(
            size=(batch_size, prefill_size, head_dim), device=device, dtype=dtype
        )
        sin = torch.rand(
            size=(batch_size, prefill_size, head_dim), device=device, dtype=dtype
        )

        position_embeds = cos, sin

        q_out, k_out, v_out = cache.rope_update(
            query_states=q,
            key_states=k,
            value_states=v,
            position_embeds=position_embeds,
            layer_idx=l,
            cache_kwargs={"cache_position": cache_position},
        )

        hf_q_out, hf_k_out = apply_rotary_pos_emb(
            q=q,
            k=k,
            cos=cos,
            sin=sin,
        )

        hf_k_out, hf_v_out = hf_cache.update(
            key_states=hf_k_out,
            value_states=v,
            layer_idx=l,
            cache_kwargs={"cache_position": cache_position},
        )

        tensors_equal(
            hf_q_out[:, :, :current_seq_length, :], q_out[:, :, :current_seq_length, :]
        )
        tensors_equal(
            hf_k_out[:, :, :current_seq_length, :], k_out[:, :, :current_seq_length, :]
        )
        tensors_equal(
            hf_v_out[:, :, :current_seq_length, :], v_out[:, :, :current_seq_length, :]
        )

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

        current_seq_length = current_seq_length + decode_size

        for l in range(num_layers):
            decode_input_tensor = torch.rand(
                size=(batch_size, num_key_value_heads, decode_size, head_dim),
                dtype=dtype,
            ).to(
                device=device
            )  # move after for platforms without rand implemented

            q, k, v = decode_input_tensor, decode_input_tensor, decode_input_tensor

            cos = torch.rand(
                size=(batch_size, decode_size, head_dim), device=device, dtype=dtype
            )
            sin = torch.rand(
                size=(batch_size, decode_size, head_dim), device=device, dtype=dtype
            )

            position_embeds = cos, sin

            q_out, k_out, v_out = cache.rope_update(
                query_states=q,
                key_states=k,
                value_states=v,
                position_embeds=position_embeds,
                layer_idx=l,
                cache_kwargs={"cache_position": cache_position},
            )

            hf_q_out, hf_k_out = apply_rotary_pos_emb(
                q=q,
                k=k,
                cos=cos,
                sin=sin,
            )

            hf_k_out, hf_v_out = hf_cache.update(
                key_states=hf_k_out,
                value_states=v,
                layer_idx=l,
                cache_kwargs={"cache_position": cache_position},
            )

            tensors_equal(
                hf_q_out[:, :, :current_seq_length, :],
                q_out[:, :, :current_seq_length, :],
            )
            tensors_equal(
                hf_k_out[:, :, :current_seq_length, :],
                k_out[:, :, :current_seq_length, :],
            )
            tensors_equal(
                hf_v_out[:, :, :current_seq_length, :],
                v_out[:, :, :current_seq_length, :],
            )

        assert current_seq_length == cache.get_seq_length(layer_idx=0)


def test_hybrid_kv_cache_rope_update_batch_size1_fp32_cpu():
    dtype = torch.float32
    device = "cpu"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_rope_update_batch_size4_fp32_cpu():
    dtype = torch.float32
    device = "cpu"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_rope_update_batch_size1_fp32_gpu():
    dtype = torch.float32
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_rope_update_batch_size4_fp32_gpu():
    dtype = torch.float32
    device = "cuda"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_rope_update_batch_size1_fp16_gpu():
    dtype = torch.float16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_rope_update_batch_size4_fp16_gpu():
    dtype = torch.float16
    device = "cuda"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_rope_update_batch_size1_bf16_gpu():
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_rope_update_batch_size4_bf16_gpu():
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


# large prefill


def test_hybrid_kv_cache_rope_update_batch_size1_fp32_cpu_large_prefill():
    dtype = torch.float32
    device = "cpu"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_rope_update_batch_size1_fp32_gpu_large_prefill():
    dtype = torch.float32
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_rope_update_batch_size1_fp16_gpu_large_prefill():
    dtype = torch.float16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_rope_update_batch_size1_bf16_gpu_large_prefill():
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48
    _test_hybrid_kv_cache_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def _test_hybrid_kv_cache_complex_rope_update(
    batch_size: int,
    dtype: torch.dtype,
    device: str,
    attention_chunk_size=32,
    prefill_size=30,
):
    max_cache_len = 128
    hidden_size = 256
    model_config = llama4_model_config(
        hidden_size=hidden_size,
        intermediate_size=1024,
        attention_chunk_size=attention_chunk_size,
    )

    engine_config = MuiEngineConfig(tensor_parallelism=1)

    hf_cache = HybridChunkedCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        dtype=dtype,
        device=device,
    )

    cache = MuiHybridChunkedCache(
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
    # chosen prefill size to be smaller than attention_chunk_size
    current_seq_length = prefill_size
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

        q, k, v = prefill_input_tensor, prefill_input_tensor, prefill_input_tensor

        # position embeds need to be in float32
        position_embeds = torch.rand(
            size=(batch_size, prefill_size, head_dim),
            device=device,
            dtype=torch.float32,
        )
        position_embeds = torch.view_as_complex(
            position_embeds.view(batch_size, prefill_size, head_dim // 2, 2)
        )

        q_out, k_out, v_out = cache.rope_update(
            query_states=q,
            key_states=k,
            value_states=v,
            position_embeds=position_embeds,
            layer_idx=l,
            cache_kwargs={"cache_position": cache_position},
            complex_rope=True,
        )

        hf_q_out, hf_k_out = apply_complex_rotary_emb(
            xk=k, xq=q, freqs_cis=position_embeds
        )

        hf_k_out, hf_v_out = hf_cache.update(
            key_states=hf_k_out,
            value_states=v,
            layer_idx=l,
            cache_kwargs={"cache_position": cache_position},
        )

        tensors_equal(
            hf_q_out[:, :, :current_seq_length, :], q_out[:, :, :current_seq_length, :]
        )
        tensors_equal(
            hf_k_out[:, :, :current_seq_length, :], k_out[:, :, :current_seq_length, :]
        )
        tensors_equal(
            hf_v_out[:, :, :current_seq_length, :], v_out[:, :, :current_seq_length, :]
        )

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

        current_seq_length = current_seq_length + decode_size

        for l in range(num_layers):
            decode_input_tensor = torch.rand(
                size=(batch_size, num_key_value_heads, decode_size, head_dim),
                dtype=dtype,
            ).to(
                device=device
            )  # move after for platforms without rand implemented

            q, k, v = decode_input_tensor, decode_input_tensor, decode_input_tensor

            # position embeds need to be in float32
            position_embeds = torch.rand(
                size=(batch_size, decode_size, head_dim),
                device=device,
                dtype=torch.float32,
            )
            position_embeds = torch.view_as_complex(
                position_embeds.view(batch_size, decode_size, head_dim // 2, 2)
            )

            q_out, k_out, v_out = cache.rope_update(
                query_states=q,
                key_states=k,
                value_states=v,
                position_embeds=position_embeds,
                layer_idx=l,
                cache_kwargs={"cache_position": cache_position},
                complex_rope=True,
            )

            hf_q_out, hf_k_out = apply_complex_rotary_emb(
                xk=k, xq=q, freqs_cis=position_embeds
            )

            hf_k_out, hf_v_out = hf_cache.update(
                key_states=hf_k_out,
                value_states=v,
                layer_idx=l,
                cache_kwargs={"cache_position": cache_position},
            )

            tensors_equal(
                hf_q_out[:, :, :current_seq_length, :],
                q_out[:, :, :current_seq_length, :],
            )
            tensors_equal(
                hf_k_out[:, :, :current_seq_length, :],
                k_out[:, :, :current_seq_length, :],
            )
            tensors_equal(
                hf_v_out[:, :, :current_seq_length, :],
                v_out[:, :, :current_seq_length, :],
            )

        assert current_seq_length == cache.get_seq_length(layer_idx=0)


def test_hybrid_kv_cache_complex_rope_update_batch_size1_fp32_cpu():
    dtype = torch.float32
    device = "cpu"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30

    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_complex_rope_update_batch_size4_fp32_cpu():
    dtype = torch.float32
    device = "cpu"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_complex_rope_update_batch_size1_fp32_gpu():
    dtype = torch.float32
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_complex_rope_update_batch_size4_fp32_gpu():
    dtype = torch.float32
    device = "cuda"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_complex_rope_update_batch_size1_fp16_gpu():
    dtype = torch.float16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_complex_rope_update_batch_size4_fp16_gpu():
    dtype = torch.float16
    device = "cuda"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_complex_rope_update_batch_size1_bf16_gpu():
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_complex_rope_update_batch_size4_bf16_gpu():
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 4
    attention_chunk_size = 32
    prefill_size = 30
    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


# large prefill


def test_hybrid_kv_cache_complex_rope_update_batch_size1_fp32_cpu_large_prefill():
    dtype = torch.float32
    device = "cpu"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48

    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_complex_rope_update_batch_size1_fp32_gpu_large_prefill():
    dtype = torch.float32
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48
    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_complex_rope_update_batch_size1_fp16_gpu_large_prefill():
    dtype = torch.float16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48
    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )


def test_hybrid_kv_cache_complex_rope_update_batch_size1_bf16_gpu_large_prefill():
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 1
    attention_chunk_size = 32
    prefill_size = 48
    _test_hybrid_kv_cache_complex_rope_update(
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        attention_chunk_size=attention_chunk_size,
        prefill_size=prefill_size,
    )
