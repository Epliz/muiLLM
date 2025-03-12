#ifndef __MUILLM_FLASH_DECODING_KERNELS_CUH__
#define __MUILLM_FLASH_DECODING_KERNELS_CUH__

#include <cuda_fp16.h>

#include <torch/extension.h>

void flash_decoding_partially_aggregate(
  hipStream_t stream,
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, S, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, S, embed_dim]
  const half* __restrict__ m_in, // shape [B, 1, T, S]
  float* __restrict__ partial_vectors, // [B, num_q_heads, G, T, embed_dim]
  half* __restrict__ partial_softmax_denoms, // [B, num_q_heads, G, T]
  half* __restrict__ partial_maxes, // [B, num_q_heads, G, T]
  // tensor dimension sizes
  unsigned B,
  unsigned G,
  unsigned T, // num tokens to decode
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_kv_heads,
  float attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
  // k strides
  unsigned kv_batch_stride,
  unsigned kv_head_stride,
  unsigned kv_tok_stride
);
  
void flash_decoding_final_reduce(
  hipStream_t stream,
  const half* __restrict__ partial_vectors, // shape [B, num_q_heads, G, T, embed_dim]
  const half* __restrict__ partial_softmax_denoms, // shape [B, num_q_heads, G, T]
  half* __restrict__ partial_maxes, // [B, num_q_heads, G, T]
  half* __restrict__ hidden_out, // [B, num_q_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned B,
  unsigned G, // number of groups
  unsigned T, // num tokens to decode
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim
);

// torch wrappers

at::Tensor flash_decode_no_mask(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v  // [B, num_v_heads, S, embed_dim]
);

at::Tensor flash_decode_masked(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v,  // [B, num_v_heads, S, embed_dim]
  torch::Tensor& m  // [B, 1, S, T]
);

#endif /* __MUILLM_FLASH_DECODING_KERNELS_CUH__ */