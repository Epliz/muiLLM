#include <torch/extension.h>

#include <vector>

at::Tensor muillm_linear_forward(
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor x);

at::Tensor muillm_linear_forward_no_bias(
    torch::Tensor weights,
    torch::Tensor x);

at::Tensor muillm_gateupsilu_forward(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("muillm_linear_forward", &muillm_linear_forward, "muillm linear forward");
  m.def("muillm_linear_forward_no_bias", &muillm_linear_forward_no_bias, "muillm linear forward no bias");
  m.def("muillm_gateupsilu_forward", &muillm_gateupsilu_forward, "muillm gate up silu forward");
  m.def("muillm_rmsnorm_forward", &muillm_rmsnorm_forward, "muillm rmsnorm forward");
  // rotary
  m.def("muillm_rope_forward_no_cache", &muillm_rope_forward_no_cache, "muillm rotary forward no cache");
  m.def("muillm_rope_forward_dynamic_cache", &muillm_rope_forward_dynamic_cache, "muillm rotary forward dynamic cache");
}
