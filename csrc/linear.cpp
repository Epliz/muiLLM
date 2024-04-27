#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor muillm_linear_forward_cuda(
    torch::Tensor& weights,
    torch::Tensor* bias,
    torch::Tensor& x);

at::Tensor muillm_linear_forward(
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor x) {
  //return torch::addmm(bias, x, weights.transpose(0, 1));
  return muillm_linear_forward_cuda(weights, &bias, x);
}

at::Tensor muillm_linear_forward_no_bias(
    torch::Tensor weights,
    torch::Tensor x) {
  CHECK_INPUT(weights);
  CHECK_INPUT(x);

  return muillm_linear_forward_cuda(weights, nullptr, x);
  //return torch::matmul(x, weights.transpose(0, 1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("muillm_linear_forward", &muillm_linear_forward, "muillm linear forward");
  m.def("muillm_linear_forward_no_bias", &muillm_linear_forward_no_bias, "muillm linear forward no bias");
}
