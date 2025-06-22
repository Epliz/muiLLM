#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

void muillm_l2norm_fp16(
  hipStream_t stream,
  unsigned B,
  unsigned K,
  const half* x,
  half* y,
  float epislon
);

void muillm_l2norm_bf16(
  hipStream_t stream,
  unsigned B,
  unsigned K,
  const __hip_bfloat16* x,
  __hip_bfloat16* y,
  float epislon
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor muillm_l2norm_forward(
    torch::Tensor x,
    float epsilon) {
  CHECK_INPUT(x);

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  const auto K = x.size(x.dim() - 1);
  // batch size
  // TODO: is numel slow?
  const auto B = x.numel() / K;

  auto output_sizes = x.sizes().vec();

  auto dtype = x.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto y = torch::empty(output_sizes, output_options);

  if (dtype == torch::kBFloat16) {
    muillm_l2norm_bf16(
        stream,
        B,
        K,
        (__hip_bfloat16*)x.data_ptr(),
        (__hip_bfloat16*)y.data_ptr(),
        epsilon
    );
    return y;
  } else if (dtype == torch::kFloat16) {
    muillm_l2norm_fp16(
          stream,
          B,
          K,
          (const half*)x.data_ptr(),
          (half*)y.data_ptr(),
          epsilon
      );
  } else {
    TORCH_CHECK(false, "muillm_l2norm_forward: unsupported dtype ");
  }

  return y;
}