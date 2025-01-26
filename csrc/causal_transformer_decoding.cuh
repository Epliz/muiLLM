#ifndef __MUILLM_CAUSAL_TRANSFORMER_DECODING_KERNELS_CUH__
#define __MUILLM_CAUSAL_TRANSFORMER_DECODING_KERNELS_CUH__

#include <torch/extension.h>

at::Tensor muillm_causal_transformer_compute_softmax_scores_no_mask(
    torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
    torch::Tensor& k // [B, num_k_heads, NEW_T, embed_dim]
);

at::Tensor muillm_causal_transformer_apply_softmax_scores(
    torch::Tensor& attention_weights, // [B, num_q_heads, T, NEW_T]
    torch::Tensor& v // [B, num_v_heads, NEW_T, embed_dim]
);

at::Tensor muillm_causal_transformer_decoding_no_mask(
    torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
    torch::Tensor& k, // [B, num_k_heads, NEW_T, embed_dim]
    torch::Tensor& v  // [B, num_v_heads, NEW_T, embed_dim]
);

at::Tensor muillm_causal_transformer_decoding_masked(
    torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
    torch::Tensor& k, // [B, num_k_heads, NEW_T, embed_dim]
    torch::Tensor& v,  // [B, num_v_heads, NEW_T, embed_dim]
    torch::Tensor& m  // [B, 1, NEW_T, T]
);

#endif // __MUILLM_CAUSAL_TRANSFORMER_DECODING_KERNELS_CUH__