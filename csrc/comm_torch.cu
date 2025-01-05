#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "comm.h"

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define MAX_GPUS 16

void muillm_all_reduce_sum(
    muillm_comm_t* comm,
    std::vector<torch::Tensor>& tensors
) {
  size_t num_tensors = tensors.size();
  if (num_tensors > MAX_GPUS) {
    TORCH_CHECK(false, "too many tensors were provided to reduce");
  }
  if (num_tensors != comm->local_size) {
    TORCH_CHECK(false, "incorrect number of tensors passed");
  }

  void* src_ptrs[MAX_GPUS];
  for (size_t i = 0; i < num_tensors; i++) {
    CHECK_INPUT(tensors[i]);
    src_ptrs[i] = (void*)tensors[i].data_ptr();
  }

  // use the main stream as stream 0
  for (int d = 0; d < num_tensors; d++) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(d);
    comm->streams[d] = stream;
  }

  auto count = tensors[0].numel();

  muillm_comm_datatype_t datatype;
  auto dtype = tensors[0].dtype();

  if (dtype == torch::kFloat16) {
      datatype = MUILLM_COMM_FP16;
  } else if (dtype == torch::kFloat32) {
      datatype = MUILLM_COMM_FP32;
  } else {
    // error
    TORCH_CHECK(false, "Unsupported dtype for all_reduce_sum");
    return;
  }

  // src is also dest
  if (muillm_comm_all_reduce_sum(comm, (const void**) src_ptrs, (void**) src_ptrs, count, datatype) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "reduction failed");
  }
}