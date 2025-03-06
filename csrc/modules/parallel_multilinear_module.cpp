#include "parallel_multilinear_module.h"

MuiLLMParallelMultiLinear::MuiLLMParallelMultiLinear(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  MuiLLMParallelLinear* linear,
  std::vector<std::tuple<int, int>>& slices,
  int sharding_dim
) {
  this->engine = engine;
  this->comm = comm;

  this->linear = linear;
  this->slices = slices;
  this->sharding_dim = sharding_dim;
}

MuiLLMParallelMultiLinear::~MuiLLMParallelMultiLinear() {
  // nothing to do
}

std::vector<torch::Tensor> MuiLLMParallelMultiLinear::forward(
  torch::Tensor& input,
  bool collect_outputs
) {
  auto undef_tensor = torch::Tensor();

  int num_slices = this->slices.size();
  if (this->sharding_dim == 1) {
    // if we are sharding by columns, we can let MuiParallelLinear collect the results
    auto all_outputs = this->linear->forward(input, /*residual*/ undef_tensor, collect_outputs);
    auto all_split_outputs = this->slice_outputs(all_outputs);
    return all_split_outputs;
  }

  // but if we shard by row, we need to do it manually as tensors are interleaved and it would not give
  // the right results
  auto all_outputs = this->linear->forward(input, /*residual*/ undef_tensor, /* collect_outputs */ false);
  auto all_split_outputs = this->slice_outputs(all_outputs);

  if (collect_outputs) {
    // collect the outputs if necessary
    for (int output_idx = 0; output_idx < num_slices; output_idx++) {
      all_split_outputs[output_idx] = this->collect_output(all_split_outputs[output_idx]);
    }
  }

  return all_split_outputs;
};

std::vector<torch::Tensor> MuiLLMParallelMultiLinear::slice_outputs(
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

torch::Tensor MuiLLMParallelMultiLinear::collect_output(
  torch::Tensor& output
) {
  return MuiLLMParallelLinear::collect_output(this->comm, output, this->sharding_dim);
}

muillm_parallel_multilinear_module_ptr_t muillm_parallel_multilinear_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_linear_module_ptr_t linear,
  std::vector<std::tuple<int, int>> slices,
  int sharding_dim
) {
  MuiLLMParallelMultiLinear* multilinear_module = new MuiLLMParallelMultiLinear(
    engine.engine_ptr,
    comm.comm_ptr,
    linear.ptr,
    slices,
    sharding_dim
  );

  muillm_parallel_multilinear_module_ptr_t ptr;
  ptr.ptr = multilinear_module;

  return ptr;
}

void muillm_parallel_multilinear_module_deinit_trampoline(
  muillm_parallel_multilinear_module_ptr_t ptr
) {
  delete ptr.ptr;
}

std::vector<torch::Tensor> muillm_parallel_multilinear_module_forward_trampoline(
  muillm_parallel_multilinear_module_ptr_t ptr,
  torch::Tensor input,
  bool collect_outputs
) {
  MuiLLMParallelMultiLinear* multilinear_module = ptr.ptr;
  return multilinear_module->forward(input, collect_outputs);
}