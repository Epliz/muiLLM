from typing import List
import torch

import muillm_ext

class _MuiParallelCausalDecoding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qs, ks, vs):
        output = muillm_ext.muillm_parallel_causal_transformer_decoding_no_mask(qs, ks, vs)

        ctx.save_for_backward(qs, ks, vs)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("causal decoding backward not implemented")

def mui_parallel_causally_decode(qs: List[torch.Tensor], ks: List[torch.Tensor], vs: List[torch.Tensor]) -> List[torch.Tensor] :
    # Expected shapes:
    #  q: [B, num_q_heads, T, embed_dim]
    #  k: [B, num_k_heads, NEW_T, embed_dim]
    #  v: [B, num_v_heads, NEW_T, embed_dim]
    return _MuiParallelCausalDecoding.apply(qs, ks, vs)

class _MuiParallelMaskedCausalDecoding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qs, ks, vs, ms):
        output = muillm_ext.muillm_parallel_causal_transformer_decoding_masked(qs, ks, vs, ms)

        ctx.save_for_backward(qs, ks, vs, ms)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("causal decoding backward not implemented")


def mui_parallel_causally_decode_masked(qs: List[torch.Tensor], ks: List[torch.Tensor], vs: List[torch.Tensor], ms: List[torch.Tensor]) -> List[torch.Tensor] :
    # Expected shapes:
    #  q: [B, num_q_heads, T, embed_dim]
    #  k: [B, num_k_heads, NEW_T, embed_dim]
    #  v: [B, num_v_heads, NEW_T, embed_dim]
    #  m: [B, 1, NEW_T, T]
    return _MuiParallelMaskedCausalDecoding.apply(qs, ks, vs, ms)