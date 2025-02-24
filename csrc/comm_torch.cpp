#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "comm.h"
#include "comm_torch.h"

#include <stdio.h>

//
// Python trampolines
//

muillm_comm_ptr muillm_comm_init_trampoline(
  muillm_engine_ptr engine,
  int world_size,
  int local_size,
  int rank,
  int local_rank
) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  muillm_comm_t* ptr = nullptr;
  muillm_comm_error_t error = muillm_comm_init(engine.engine_ptr, world_size, local_size, rank, local_rank, &ptr, stream);

  TORCH_CHECK(error == MUILLM_COMM_SUCCESS, "an error happened when initializing mui comm");

  muillm_comm_ptr ret;
  ret.comm_ptr = ptr;
  return ret;
}

void muillm_all_reduce_sum_trampoline(
  muillm_comm_ptr comm,
  torch::Tensor& tensor
) {
  muillm_comm_error_t error = muillm_all_reduce_sum(comm.comm_ptr, tensor);
  TORCH_CHECK(error == MUILLM_COMM_SUCCESS, "an error happened during all_reduce");
}

void muillm_broadcast_trampoline(
  muillm_comm_ptr comm,
  torch::Tensor& tensor,
  int src
) {
  muillm_comm_error_t error = muillm_broadcast(comm.comm_ptr, tensor, src);
  TORCH_CHECK(error == MUILLM_COMM_SUCCESS, "an error happened during broadcast");
}




//
//
//

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

muillm_comm_error_t muillm_all_reduce_sum(
    muillm_comm_t* comm,
    torch::Tensor& tensor
) {
  CHECK_INPUT(tensor);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // TODO: is numel slow?
  auto count = tensor.numel();

  muillm_comm_datatype_t datatype;
  auto dtype = tensor.dtype();

  if (dtype == torch::kFloat16) {
      datatype = MUILLM_COMM_FP16;
  } else if (dtype == torch::kFloat32) {
      datatype = MUILLM_COMM_FP32;
  } else {
    // error
    TORCH_CHECK(false, "Unsupported dtype for all_reduce_sum");
  }

  void* src_ptr = (void*)tensor.data_ptr();

  // src is also dest
  return muillm_comm_all_reduce_sum(comm, src_ptr, src_ptr, count, datatype, stream);
}


muillm_comm_error_t muillm_broadcast(
  muillm_comm_t* comm,
  torch::Tensor& tensor,
  int src
) {
CHECK_INPUT(tensor);

cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // TODO: is numel slow?
auto count = tensor.numel();

muillm_comm_datatype_t datatype;
auto dtype = tensor.dtype();

if (dtype == at::kBool) {
  datatype = MUILLM_COMM_BOOL;
} else if (dtype == at::kChar) {
  datatype = MUILLM_COMM_INT8;
} else if (dtype == at::kShort) {
  datatype = MUILLM_COMM_INT16;
} else if (dtype == at::kInt) {
  datatype = MUILLM_COMM_INT32;
} else if (dtype == at::kLong) {
  datatype = MUILLM_COMM_INT64;
} else if (dtype == at::kHalf) {
  datatype = MUILLM_COMM_FP16;
} else if (dtype == at::kFloat) {
  datatype = MUILLM_COMM_FP32;
} else if (dtype == at::kDouble) {
  datatype = MUILLM_COMM_FP64;
} else {
  // error
  TORCH_CHECK(false, "Unsupported dtype for broadcast");
}

void* src_ptr = (void*)tensor.data_ptr();

// src is also dest
return muillm_comm_broadcast(comm, src, src_ptr, src_ptr, count, datatype, stream);
}