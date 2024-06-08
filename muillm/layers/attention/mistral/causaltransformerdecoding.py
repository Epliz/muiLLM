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


class _MuiCausalSoftmaxScoreComputation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        output = muillm_ext.muillm_causal_transformer_compute_softmax_scores_no_mask(q, k)

        ctx.save_for_backward(q, k)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("causal decoding backward not implemented")

class _MuiCausalSoftmaxScoreApplication(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attention_weights, v):
        output = muillm_ext.muillm_causal_transformer_apply_softmax_scores(attention_weights, v)

        ctx.save_for_backward(attention_weights, v)

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
    if False:
        return _MuiCausalDecoding.apply(q, k, v)
    else:
        # Non fully fused
        attn_weights =  _MuiCausalSoftmaxScoreComputation.apply(q, k)

        #v = repeat_kv(v, int(q.shape[1] / v.shape[1]))
        #return torch.matmul(attn_weights, v)
        return _MuiCausalSoftmaxScoreApplication.apply(attn_weights, v)