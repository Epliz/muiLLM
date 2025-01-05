#include "gateup_kernels.cuh"
#include "comm_torch.cuh"

#include <ATen/cuda/CUDAContext.h>

std::vector<at::Tensor> muillm_parallel_gateupsilu_forward(
    muillm_comm_t* comm,
    std::vector<torch::Tensor>& norm_weights,
    float epsilon,
    std::vector<torch::Tensor>& gate_weights,
    std::vector<torch::Tensor>& up_weights,
    std::vector<torch::Tensor>& down_weights,
    torch::Tensor& residual,
    std::vector<torch::Tensor>& x) {
  size_t tp_level = gate_weights.size();

  std::vector<at::Tensor> outputs;

  auto undef_tensor = torch::Tensor();

  if (tp_level > MUILLM_COMM_MAX_GPUS) {
    TORCH_CHECK(false, "tensor level parallelism higher than what is supported by comms");
  }

  // we fuse part of the reduction with the GEMV operation
  // by making the GEMV operation write into the reduction buffers
  muillm_comm_error_t muillm_error;
  muillm_comm_buffer_set_t* buffer_set = nullptr;

  const auto N = down_weights[0].size(0);

  size_t count = N;
  muillm_comm_datatype_t datatype = MUILLM_COMM_FP16;

  if ((muillm_error = muillm_comm_get_buffer_set(comm, count, datatype, &buffer_set)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "failed to get reduction buffers");
  }

  // get pointers on the output tensors to pass to the reduction
  void* output_ptrs[MUILLM_COMM_MAX_GPUS];

  for (size_t t = 0; t < tp_level; t++) {
    auto device = x[t].device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

    auto dtype = torch::kFloat16;
    auto output_options = at::TensorOptions()
                              .dtype(dtype)
                              .layout(at::kStrided)
                              .device(device) // same output device as inputs
                              .requires_grad(false);

    // y has the same dimensions as x, except the last dim that is given by
    // the out_features of weights
    auto output_sizes = x[t].sizes().vec();
    output_sizes[output_sizes.size() - 1] = N;

    auto output = torch::empty(output_sizes, output_options);
    void* output_ptr = output.data_ptr();

    muillm_gateupsilu_forward_placed_output(
      norm_weights[t],
      epsilon,
      gate_weights[t],
      up_weights[t],
      down_weights[t],
      t == 0 ? residual : undef_tensor,
      x[t],
      buffer_set->buffers[t]
    );

    outputs.push_back(output);
    output_ptrs[t] = output_ptr;
  }

  // finish the reduction
  if ((muillm_error = muillm_comm_all_reduce_sum(
    comm,
    (const void**) buffer_set->buffers,
    (void**) output_ptrs,
    count,
    datatype
    )) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "reduction failed");
  }

  return outputs;
}

std::vector<at::Tensor> muillm_parallel_gateupsilu_split_forward(
    muillm_comm_t* comm,
    std::vector<torch::Tensor>& norm_weights,
    float epsilon,
    std::vector<torch::Tensor>& gate_weights,
    std::vector<torch::Tensor>& up_weights,
    std::vector<torch::Tensor>& down_weights,
    torch::Tensor& residual,
    std::vector<torch::Tensor>& x) {
  size_t tp_level = gate_weights.size();

  std::vector<at::Tensor> outputs;

  auto undef_tensor = torch::Tensor();

  if (tp_level > MUILLM_COMM_MAX_GPUS) {
    TORCH_CHECK(false, "tensor level parallelism higher than what is supported by comms");
  }

  // we fuse part of the reduction with the GEMV operation
  // by making the GEMV operation write into the reduction buffers
  muillm_comm_error_t muillm_error;
  muillm_comm_buffer_set_t* buffer_set = nullptr;

  const auto N = down_weights[0].size(0);

  size_t count = N;
  muillm_comm_datatype_t datatype = MUILLM_COMM_FP16;

  if ((muillm_error = muillm_comm_get_buffer_set(comm, count, datatype, &buffer_set)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "failed to get reduction buffers");
  }

  // get pointers on the output tensors to pass to the reduction
  void* output_ptrs[MUILLM_COMM_MAX_GPUS];

  for (size_t t = 0; t < tp_level; t++) {
    auto device = x[t].device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

    auto dtype = torch::kFloat16;
    auto output_options = at::TensorOptions()
                              .dtype(dtype)
                              .layout(at::kStrided)
                              .device(device) // same output device as inputs
                              .requires_grad(false);

    // y has the same dimensions as x, except the last dim that is given by
    // the out_features of weights
    auto output_sizes = x[t].sizes().vec();
    output_sizes[output_sizes.size() - 1] = N;

    auto output = torch::empty(output_sizes, output_options);
    void* output_ptr = output.data_ptr();

    muillm_gateupsilu_split_forward_placed_output(
      norm_weights[t],
      epsilon,
      gate_weights[t],
      up_weights[t],
      down_weights[t],
      t == 0 ? residual : undef_tensor,
      x[t],
      buffer_set->buffers[t]
    );

    outputs.push_back(output);
    output_ptrs[t] = output_ptr;
  }

  // finish the reduction
  if ((muillm_error = muillm_comm_all_reduce_sum(
    comm,
    (const void**) buffer_set->buffers,
    (void**) output_ptrs,
    count,
    datatype
    )) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "reduction failed");
  }

  return outputs;
}