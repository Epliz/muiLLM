from typing import Tuple
import torch

import muillm_ext


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _MuiRotaryNoCache(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions_ids, cos_cached, sin_cached, q, k):
        output = muillm_ext.muillm_rope_forward_no_cache(
            positions_ids, cos_cached, sin_cached, q, k
        )

        ctx.save_for_backward(positions_ids, cos_cached, sin_cached, q, k)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("rotary backward not implemented")


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    dtype = q.dtype
    if (q.is_cuda) and ((dtype == torch.float16) or (dtype == torch.bfloat16)):
        cos = cos.contiguous()
        sin = sin.contiguous()
        # can dispatch to the custom kernel
        return _MuiRotaryNoCache.apply(
            None,
            cos,
            sin,
            q,
            k,
        )
    else:
        # positions_ids  torch.Size([B, T])
        # cos  torch.Size([54, 128])
        # sin  torch.Size([54, 128])
        # q  torch.Size([B, 32, T, 128])
        # k  torch.Size([B, 8, T, 128])
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        # cos[position_ids]  torch.Size([B, 1, T, 128])
        # sin[position_ids]  torch.Size([B, 1, T, 128])
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class _MuiComplexRotaryNoCache(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, position_embeds):
        output = muillm_ext.muillm_complex_rope_forward_no_cache(q, k, position_embeds)

        ctx.save_for_backward(q, k, position_embeds)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("complex rotary backward not implemented")


def apply_complex_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = xq.dtype
    if (xq.is_cuda) and ((dtype == torch.float16) or (dtype == torch.bfloat16)):
        freqs_cis = freqs_cis.contiguous()
        # can dispatch to the custom kernel
        return _MuiComplexRotaryNoCache.apply(
            xq,
            xk,
            freqs_cis,
        )
    else:
        # freqs_cis is always a complex tensor of floats
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xq_out = torch.view_as_real(xq_ * freqs_cis[:, None, :, :]).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis[:, None, :, :]).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
