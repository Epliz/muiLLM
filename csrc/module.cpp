#include <torch/extension.h>

#include <vector>

#include "linear_kernels.cuh"

at::Tensor muillm_linear_forward_trampoline(
    torch::Tensor x,
    torch::Tensor weights,
    std::optional<torch::Tensor> norm_weights_,
    float epsilon,
    std::optional<torch::Tensor> mul_bias_,
    std::optional<torch::Tensor> add_bias_) {
    torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : torch::Tensor();
    torch::Tensor mul_bias = mul_bias_.has_value() ? mul_bias_.value() : torch::Tensor();
    torch::Tensor add_bias = add_bias_.has_value() ? add_bias_.value() : torch::Tensor();
    return muillm_linear_activ_forward(
        norm_weights,
        epsilon,
        weights,
        mui_activation::Identity,
        mul_bias,
        add_bias,
        x
    );
}

at::Tensor muillm_gateupsilu_forward(
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor x);

at::Tensor muillm_rmsnorm_forward(
    torch::Tensor weights,
    torch::Tensor inputs,
    float epsilon);


std::vector<at::Tensor> muillm_rope_forward_no_cache(
    torch::Tensor& position_ids,
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in
);

std::vector<at::Tensor> muillm_rope_forward_dynamic_cache(
    torch::Tensor& position_ids,
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& prev_k_cache,
    torch::Tensor& prev_v_cache
);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("muillm_linear_forward", &muillm_linear_forward_trampoline, "muillm linear forward", py::arg("x"), py::arg("weights"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none());
  m.def("muillm_gateupsilu_forward", &muillm_gateupsilu_forward, "muillm gate up silu forward");
  m.def("muillm_rmsnorm_forward", &muillm_rmsnorm_forward, "muillm rmsnorm forward");
  // rotary
  m.def("muillm_rope_forward_no_cache", &muillm_rope_forward_no_cache, "muillm rotary forward no cache");
  m.def("muillm_rope_forward_dynamic_cache", &muillm_rope_forward_dynamic_cache, "muillm rotary forward dynamic cache");
  // causal transformer decoding
  m.def("muillm_causal_transformer_compute_softmax_scores_no_mask", &muillm_causal_transformer_compute_softmax_scores_no_mask, "muillm causal transformer compute softmax scores no mask");
  m.def("muillm_causal_transformer_apply_softmax_scores", &muillm_causal_transformer_apply_softmax_scores, "muillm causal transformer apply softmax scores");
  m.def("muillm_causal_transformer_decoding_no_mask", &muillm_causal_transformer_decoding_no_mask, "muillm causal transformer decoding no mask");
}
