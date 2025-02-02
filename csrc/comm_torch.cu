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

  // src is also dest
  if (muillm_comm_all_reduce_sum(comm, (const void**) src_ptrs, (void**) src_ptrs, count, datatype) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "reduction failed");
  }
}


std::vector<torch::Tensor> muillm_broadcast(
    muillm_comm_t* comm,
    torch::Tensor& tensor
) {

  size_t num_tensors = comm->local_size;
  if (num_tensors > MAX_GPUS) {
    TORCH_CHECK(false, "too many tensors were provided to reduce");
  }
  CHECK_INPUT(tensor);

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

  // create output tensors and get the streams
  auto output_sizes = tensor.sizes().vec();

  std::vector<at::Tensor> outputs;
  void* dst_ptrs[MAX_GPUS];
  for (size_t i = 0; i < num_tensors; i++) {
    // get the streams
    auto wrapped_stream = at::cuda::getCurrentCUDAStream(i);
    cudaStream_t stream = (cudaStream_t) wrapped_stream;
    comm->streams[i] = stream;

    // create the output tensor
    auto device = wrapped_stream.device();

    auto output_options = at::TensorOptions()
                              .dtype(dtype)
                              .layout(at::kStrided)
                              .device(device)
                              .requires_grad(false);

    auto output = torch::empty(output_sizes, output_options);
  
    dst_ptrs[i] = (void*)output.data_ptr();

    outputs.push_back(output);
  }

  auto count = tensor.numel();

  const void* src_ptr = (const void*)tensor.data_ptr();
  
  if (muillm_comm_broadcast(comm, (const void*) src_ptr, (void**) dst_ptrs, count, datatype) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "broadcast failed");
  }

  return outputs;
}