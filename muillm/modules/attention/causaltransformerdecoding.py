import math
import torch
import torch.nn as nn

import muillm_ext

class _MuiCausalDecoding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        output = muillm_ext.muillm_causal_transformer_decoding_no_mask(q, k, v)

        ctx.save_for_backward(q, k, v)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("causal decoding backward not implemented")

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def mui_causally_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor :
    # Expected shapes:
    #  q: [B, num_q_heads, T, embed_dim]
    #  k: [B, num_k_heads, NEW_T, embed_dim]
    #  v: [B, num_v_heads, NEW_T, embed_dim]
    return _MuiCausalDecoding.apply(q, k, v)

class _MuiMaskedCausalDecoding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, m):
        output = muillm_ext.muillm_causal_transformer_decoding_masked(q, k, v, m)

        ctx.save_for_backward(q, k, v, m)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("causal decoding backward not implemented")


def mui_causally_decode_masked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor :
    # Expected shapes:
    #  q: [B, num_q_heads, T, embed_dim]
    #  k: [B, num_k_heads, NEW_T, embed_dim]
    #  v: [B, num_v_heads, NEW_T, embed_dim]
    #  m: [B, 1, NEW_T, T]
    return _MuiMaskedCausalDecoding.apply(q, k, v, m)