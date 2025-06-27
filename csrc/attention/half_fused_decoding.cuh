#ifndef __MUILLM_HALF_FUSED_DECODING_KERNELS_CUH__
#define __MUILLM_HALF_FUSED_DECODING_KERNELS_CUH__

#include <torch/extension.h>

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