#include "half_fused_decoding.cuh"

#include <torch/extension.h>

at::Tensor muillm_causal_transformer_decoding_no_mask(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v  // [B, num_v_heads, S, embed_dim]
) {
  return causally_decode_no_mask(
    q,
    k,
    v
  );
}

at::Tensor muillm_causal_transformer_decoding_masked(
    torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
    torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
    torch::Tensor& v,  // [B, num_v_heads, S, embed_dim]
    torch::Tensor& m  // [B, 1, S, T]
) {
  return causally_decode_masked(
    q,
    k,
    v,
    m
  );
}