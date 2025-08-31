#include "multilinear_module.h"

MuiLLMMultiLinear::MuiLLMMultiLinear(
  muillm_engine_t* engine,
  MuiLLMLinear* linear,
  std::vector<std::tuple<int, int>>& slices
) {
  this->engine = engine;

  this->linear = linear;
  this->slices = slices;
}

MuiLLMMultiLinear::~MuiLLMMultiLinear() {
  // nothing to do
}

std::vector<torch::Tensor> MuiLLMMultiLinear::forward(
  torch::Tensor& input
) {
  auto undef_tensor = torch::Tensor();

  int num_slices = this->slices.size();

  auto all_outputs = this->linear->forward(input, /*residual*/ undef_tensor);
  auto all_split_outputs = this->slice_outputs(all_outputs);
  return all_split_outputs;
};

std::vector<torch::Tensor> MuiLLMMultiLinear::slice_outputs(
  torch::Tensor& all_outputs
) {
  int last_dim = all_outputs.dim() - 1;
  int num_slices = this->slices.size();
  std::vector<torch::Tensor> split_outputs(num_slices);

  for (int output_idx = 0; output_idx < num_slices; output_idx++) {
    auto slice = this->slices[output_idx];
    auto slice_start = std::get<0>(slice);
    auto slice_end = std::get<1>(slice);
    split_outputs[output_idx] = all_outputs.slice(last_dim, slice_start, slice_end);
  }

  return split_outputs;
}

muillm_multilinear_module_ptr_t muillm_multilinear_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_linear_module_ptr_t linear,
  std::vector<std::tuple<int, int>> slices
) {
  MuiLLMMultiLinear* multilinear_module = new MuiLLMMultiLinear(
    engine.engine_ptr,
    linear.ptr,
    slices
  );

  muillm_multilinear_module_ptr_t ptr;
  ptr.ptr = multilinear_module;

  return ptr;
}

void muillm_multilinear_module_deinit_trampoline(
  muillm_multilinear_module_ptr_t ptr
) {
  delete ptr.ptr;
}

std::vector<torch::Tensor> muillm_multilinear_module_forward_trampoline(
  muillm_multilinear_module_ptr_t ptr,
  torch::Tensor input
) {
  MuiLLMMultiLinear* multilinear_module = ptr.ptr;
  return multilinear_module->forward(input);
}