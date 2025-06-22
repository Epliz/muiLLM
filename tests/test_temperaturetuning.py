from typing import Tuple
import torch

from muillm.modules.attention.llama4attention import apply_temperature_tuning
from tests.test_utils import tensors_equal


# copied from Llama4
def ref_apply_temperature_tuning(
    query_states: torch.Tensor,  # [B, num_attention_heads, T, head_dim]
    cache_position: torch.LongTensor,  # [T]
    attn_scale: float,
    floor_scale: float,
) -> torch.Tensor:
    bsz, num_attention_heads, q_len, head_dim = query_states.shape

    attn_scales = (
        torch.log(torch.floor((cache_position.float() + 1.0) / floor_scale) + 1.0)
        * attn_scale
        + 1.0
    )
    attn_scales = attn_scales.view((1, 1, q_len, 1)).expand(
        (bsz, 1, q_len, 1)
    )  # batch size > 1
    query_states = (query_states * attn_scales).to(query_states.dtype)

    return query_states


def _test_apply_rotary_emb(device: str, dtype: torch.dtype):
    T = 5
    B = 1
    H = 128
    num_q_heads = 5

    # x is just used for the type and device
    xq = torch.randn((B, num_q_heads, T, H), dtype=dtype, device=device)
    cache_positions = torch.arange(T, dtype=torch.long, device=device)

    attn_scale = 0.5
    floor_scale = 0.75

    xq_out = ref_apply_temperature_tuning(xq, cache_positions, attn_scale, floor_scale)

    xq_out_m = apply_temperature_tuning(
        xq,
        cache_positions,
        attn_scale,
        floor_scale,
    )

    tensors_equal(xq_out, xq_out_m)


def test_apply_rotary_emb_fp32_cpu():
    device = "cpu"
    dtype = torch.float32

    _test_apply_rotary_emb(device=device, dtype=dtype)


def test_apply_rotary_emb_fp32_gpu():
    device = "cuda"
    dtype = torch.float32

    _test_apply_rotary_emb(device=device, dtype=dtype)


def test_apply_rotary_emb_fp16_gpu():
    device = "cuda"
    dtype = torch.float16

    _test_apply_rotary_emb(device=device, dtype=dtype)


def test_apply_rotary_emb_bf16_gpu():
    device = "cuda"
    dtype = torch.bfloat16

    _test_apply_rotary_emb(device=device, dtype=dtype)
