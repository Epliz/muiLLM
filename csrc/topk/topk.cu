#include "topk.cuh"
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#define ELEMENTS_PER_BLOCK 4096

void muillm_topk_sigmoid_fp16(
    hipStream_t stream,
    const half* x, // input values = size BxM
    half* y, // output values = size Bxk
    int64_t* indices, // output indices = size Bxk
    int B,
    int M,
    int k
);

void muillm_topk_sigmoid_bf16(
    hipStream_t stream,
    const __hip_bfloat16* x, // input values = size BxM
    __hip_bfloat16* y, // output values = size Bxk
    int64_t* indices, // output indices = size Bxk
    int B,
    int M,
    int k
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// return top-k values and indices
std::tuple<at::Tensor, at::Tensor> muillm_topk_sigmoid_forward(
    torch::Tensor x, // shape BxM
    int k) {
  CHECK_INPUT(x);

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  // M is the last dimensions
  const auto M = x.size(x.dim() - 1);
  const auto B = x.numel() / M;

  TORCH_CHECK(k > 0 && k <= M, "k must be in the range (0, M] where M is the last dimension of x");
  TORCH_CHECK(M <= ELEMENTS_PER_BLOCK, "M must be less than ELEMENTS_PER_BLOCK");

  auto output_sizes = x.sizes().vec();
  output_sizes[x.dim() - 1] = k; // change last dimension to k

  auto values_dtype = x.dtype();
  auto values_output_options = at::TensorOptions()
                            .dtype(values_dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto indices_dtype = torch::kInt64;
  auto indices_output_options = at::TensorOptions()
                            .dtype(indices_dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto values = torch::empty(output_sizes, values_output_options);
  auto indices = torch::empty(output_sizes, indices_output_options);

  if (values_dtype == torch::kFloat16) {
    muillm_topk_sigmoid_fp16(
        stream,
        (const half*)x.data_ptr(),
        (half*)values.data_ptr(),
        (int64_t*)indices.data_ptr(),
        B,
        M,
        k
    );
  } else if (values_dtype == torch::kBFloat16) {
    muillm_topk_sigmoid_bf16(
        stream,
        (__hip_bfloat16*)x.data_ptr(),
        (__hip_bfloat16*)values.data_ptr(),
        (int64_t*)indices.data_ptr(),
        B,
        M,
        k
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for top-k operation");
  }

  return std::make_tuple(values, indices);
}