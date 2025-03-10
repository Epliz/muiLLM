#ifndef __MUILLM_HALF_FUSED_DECODING_KERNELS_CUH__
#define __MUILLM_HALF_FUSED_DECODING_KERNELS_CUH__

#include <cuda_fp16.h>

#include <torch/extension.h>

// kernel wrappers

void causally_compute_transformer_softmax_scores(
  hipStream_t stream,
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, S, embed_dim]
  float* __restrict__ temp_scores_f32, // [B, num_q_heads, T, S]
  half* __restrict__ scores_out, // [B, num_q_heads, T, S]
  // tensor dimension sizes
  unsigned B,
  unsigned T, // num tokens to decode
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_k_heads,
  float attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
  // k strides
  unsigned k_batch_stride,
  unsigned k_head_stride,
  unsigned k_tok_stride
);

void causally_compute_transformer_softmax_scores_masked(
  hipStream_t stream,
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, S, embed_dim]
  const half* __restrict__ m_in, // shape [B, 1, T, S]
  float* __restrict__ temp_scores_f32, // [B, num_q_heads, T, S]
  half* __restrict__ scores_out, // [B, num_q_heads, T, S]
  // tensor dimension sizes
  unsigned B,
  unsigned T, // num tokens to decode
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_k_heads,
  float attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
  // k strides
  unsigned k_batch_stride,
  unsigned k_head_stride,
  unsigned k_tok_stride
);

void causally_apply_transformer_softmax_scores(
  hipStream_t stream,
  const half* __restrict__ attention_weights_in, // shape [B, num_q_heads, T, S]
  const half* __restrict__ v_in, // shape [B, num_v_heads, S, embed_dim]
  half* __restrict__ hidden_out, // [B, num_q_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned B,
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_v_heads,
  // v strides
  unsigned v_batch_stride,
  unsigned v_head_stride,
  unsigned v_tok_stride
);

// torch wrappers

at::Tensor causally_decode_no_mask(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v  // [B, num_v_heads, S, embed_dim]
);

at::Tensor causally_decode_masked(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v,  // [B, num_v_heads, S, embed_dim]
  torch::Tensor& m  // [B, 1, S, T]
);

#endif /* __MUILLM_HALF_FUSED_DECODING_KERNELS_CUH__ */