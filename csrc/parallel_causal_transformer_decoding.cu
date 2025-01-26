#include "causal_transformer_decoding.cuh"

#include <vector>

std::vector<at::Tensor> muillm_parallel_causal_transformer_decoding_no_mask(
    std::vector<torch::Tensor>& qs, // [B, num_q_heads, T, embed_dim]
    std::vector<torch::Tensor>& ks, // [B, num_k_heads, NEW_T, embed_dim]
    std::vector<torch::Tensor>& vs  // [B, num_v_heads, NEW_T, embed_dim]
) {
  size_t tp_level = qs.size();

  std::vector<at::Tensor> outputs;

  for (size_t t = 0; t < tp_level; t++) {
    outputs.push_back(
        muillm_causal_transformer_decoding_no_mask(
            qs[t],
            ks[t],
            vs[t]
        )
    );
  }

  return outputs;
}

std::vector<at::Tensor> muillm_parallel_causal_transformer_decoding_masked(
    std::vector<torch::Tensor>& qs, // [B, num_q_heads, T, embed_dim]
    std::vector<torch::Tensor>& ks, // [B, num_k_heads, NEW_T, embed_dim]
    std::vector<torch::Tensor>& vs,  // [B, num_v_heads, NEW_T, embed_dim]
    std::vector<torch::Tensor>& ms  // [B, 1, NEW_T, T]
) {
  size_t tp_level = qs.size();

  std::vector<at::Tensor> outputs;

  for (size_t t = 0; t < tp_level; t++) {
    outputs.push_back(
        muillm_causal_transformer_decoding_masked(
            qs[t],
            ks[t],
            vs[t],
            ms[t]
        )
    );
  }

  return outputs;
}