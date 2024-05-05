#include <torch/extension.h>

at::Tensor muillm_linear_forward(
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor x);

at::Tensor muillm_linear_forward_no_bias(
    torch::Tensor weights,
    torch::Tensor x);

at::Tensor muillm_rmsnorm_forward(
    torch::Tensor weights,
    torch::Tensor inputs,
    float epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("muillm_linear_forward", &muillm_linear_forward, "muillm linear forward");
  m.def("muillm_linear_forward_no_bias", &muillm_linear_forward_no_bias, "muillm linear forward no bias");
  m.def("muillm_rmsnorm_forward", &muillm_rmsnorm_forward, "muillm rmsnorm forward");
}
