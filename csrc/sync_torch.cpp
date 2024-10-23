#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "sync.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static void muillm_item(
    muillm_synchronizer_t* sync,
    torch::Tensor& tensor,
    void* dst,
    size_t type_size
) {
  CHECK_INPUT(tensor);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto count = tensor.numel();

  TORCH_CHECK(count == 1, "count was not a single element");

  const void* src_ptr = (const void*)tensor.data_ptr();

  muillm_sync_copy(sync, stream, dst, src_ptr, type_size);
}

bool muillm_item_bool(
    muillm_synchronizer_t* sync,
    torch::Tensor& tensor
) {
  bool ret;

  muillm_item(sync, tensor, &ret, sizeof(bool));

  return ret;
}

half muillm_item_f16(
    muillm_synchronizer_t* sync,
    torch::Tensor& tensor
) {
  half ret;

  muillm_item(sync, tensor, &ret, sizeof(half));

  return ret;
}

float muillm_item_f32(
    muillm_synchronizer_t* sync,
    torch::Tensor& tensor
) {
  float ret;

  muillm_item(sync, tensor, &ret, sizeof(float));

  return ret;
}

at::Tensor muillm_to_cpu(
    muillm_synchronizer_t* sync,
    torch::Tensor& tensor
) {
  CHECK_INPUT(tensor);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto dtype = tensor.dtype();
  auto output_sizes = tensor.sizes().vec();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(at::kCPU)
                            .requires_grad(false);

  auto y = torch::empty(output_sizes, output_options);


  auto count = tensor.numel();
  auto type_size = tensor.element_size();
  auto tensor_size = count * type_size;

  const void* src_ptr = (const void*)tensor.data_ptr();
  void* dst_ptr = (void*)y.data_ptr();

  muillm_sync_copy(sync, stream, dst_ptr, src_ptr, tensor_size);

  return y;
}