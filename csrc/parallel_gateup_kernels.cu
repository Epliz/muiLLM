#include "gateup_kernels.cuh"
#include "comm.h"

#include <ATen/cuda/CUDAContext.h>

at::Tensor muillm_parallel_gateupsilu_forward(
    muillm_comm_t* comm,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x) {
  int rank = comm->rank;
  // TODO: change this once we support interleaving
  size_t tp_level = comm->local_size;

  auto undef_tensor = torch::Tensor();

  if (tp_level > MUILLM_COMM_MAX_GPUS) {
    TORCH_CHECK(false, "tensor level parallelism higher than what is supported by comms");
  }

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  // we fuse part of the reduction with the GEMV operation
  // by making the GEMV operation write into the reduction buffers
  muillm_comm_error_t muillm_error;

  const auto N = down_weights.size(0);

  size_t count = N;
  muillm_comm_datatype_t datatype = MUILLM_COMM_FP16;

  void** buffers;
  if ((muillm_error = muillm_comm_get_buffers(comm, count, datatype, &buffers, stream)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "failed to get reduction buffers");
  }

  auto dtype = torch::kFloat16;
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // y has the same dimensions as x, except the last dim that is given by
  // the out_features of weights
  auto output_sizes = x.sizes().vec();
  output_sizes[output_sizes.size() - 1] = N;

  auto output = torch::empty(output_sizes, output_options);
  void* output_ptr = output.data_ptr();

  muillm_gateupsilu_forward_placed_output(
    norm_weights,
    epsilon,
    gate_weights,
    up_weights,
    down_weights,
    // we apply the residual only on device 0
    rank == 0 ? residual : undef_tensor,
    x,
    buffers[rank]
  );

  // finish the reduction
  if ((muillm_error = muillm_comm_placed_all_reduce_sum(
    comm,
    (const void**) buffers,
    output_ptr,
    count,
    datatype,
    stream
    )) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "reduction failed");
  }

  return output;
}

at::Tensor muillm_parallel_gateupsilu_split_forward(
    muillm_comm_t* comm,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x) {
  int rank = comm->rank;
  // TODO: change this once we support interleaving
  size_t tp_level = comm->local_size;

  auto undef_tensor = torch::Tensor();

  if (tp_level > MUILLM_COMM_MAX_GPUS) {
    TORCH_CHECK(false, "tensor level parallelism higher than what is supported by comms");
  }

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  // we fuse part of the reduction with the GEMV operation
  // by making the GEMV operation write into the reduction buffers
  muillm_comm_error_t muillm_error;

  const auto N = down_weights.size(0);

  size_t count = N;
  muillm_comm_datatype_t datatype = MUILLM_COMM_FP16;

  void** buffers;
  if ((muillm_error = muillm_comm_get_buffers(comm, count, datatype, &buffers, stream)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "failed to get reduction buffers");
  }


  auto dtype = torch::kFloat16;
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // y has the same dimensions as x, except the last dim that is given by
  // the out_features of weights
  auto output_sizes = x.sizes().vec();
  output_sizes[output_sizes.size() - 1] = N;

  auto output = torch::empty(output_sizes, output_options);
  void* output_ptr = output.data_ptr();

  muillm_gateupsilu_split_forward_placed_output(
    norm_weights,
    epsilon,
    gate_weights,
    up_weights,
    down_weights,
    // we apply the residual only on device 0
    rank == 0 ? residual : undef_tensor,
    x,
    buffers[rank]
  );

  // finish the reduction
  if ((muillm_error = muillm_comm_placed_all_reduce_sum(
    comm,
    (const void**) buffers,
    output_ptr,
    count,
    datatype,
    stream
    )) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "reduction failed");
  }

  return output;
}