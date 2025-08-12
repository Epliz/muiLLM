#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

void muillm_rmsnorm_fp16(
  hipStream_t stream,
  unsigned B,
  unsigned K,
  const half* __restrict__ W, // weight matrix - size K
  const half* __restrict__ X, // input = size BxK
  half* __restrict__ Y, // output = size BxK
  float epsilon,
  float weight_offset
);

void muillm_rmsnorm_bf16(
  hipStream_t stream,
  unsigned B,
  unsigned K,
  const __hip_bfloat16* __restrict__ W, // weight matrix - size K
  const __hip_bfloat16* __restrict__ X, // input = size BxK
  __hip_bfloat16* __restrict__ Y, // output = size BxK
  float epsilon,
  float weight_offset
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor muillm_rmsnorm_forward(
    torch::Tensor weights,
    torch::Tensor x,
    float epsilon,
    float weight_offset
) {
  CHECK_INPUT(weights);
  CHECK_INPUT(x);


  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  // TODO: is numel slow?
  const auto K = weights.numel();
  // batch size
  // TODO: is numel slow?
  const auto B = x.numel() / K;

  auto output_sizes = x.sizes().vec();

  auto dtype = weights.dtype();

  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto y = torch::empty(output_sizes, output_options);

  if (dtype == torch::kFloat16) {
    muillm_rmsnorm_fp16(
      stream,
      B,
      K,
      (const half*)weights.data_ptr(),
      (const half*)x.data_ptr(),
      (half*)y.data_ptr(),
      epsilon,
      weight_offset
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_rmsnorm_bf16(
      stream,
      B,
      K,
      (const __hip_bfloat16*)weights.data_ptr(),
      (const __hip_bfloat16*)x.data_ptr(),
      (__hip_bfloat16*)y.data_ptr(),
      epsilon,
      weight_offset
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for muillm_rmsnorm_forward");
  }

  return y;
}