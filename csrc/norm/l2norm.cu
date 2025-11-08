#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

void muillm_rmsnorm_fp16(
  hipStream_t stream,
  unsigned B,
  unsigned K,
  const half* __restrict__ W, // weight matrix - size K
  const half* __restrict__ X, // input = size BxK
  const half* __restrict__ RB, // optional residual = size BxK
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
  const __hip_bfloat16* __restrict__ RB, // optional residual = size BxK
  __hip_bfloat16* __restrict__ Y, // output = size BxK
  float epsilon,
  float weight_offset
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor muillm_l2norm_forward(
    torch::Tensor x,
    torch::Tensor residual, // optional
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
    muillm_rmsnorm_bf16(
        stream,
        B,
        K,
        /* W */ nullptr,
        (__hip_bfloat16*)x.data_ptr(),
        residual.defined() ? (__hip_bfloat16*)residual.data_ptr() : nullptr,
        (__hip_bfloat16*)y.data_ptr(),
        epsilon,
        0.0f
    );
    return y;
  } else if (dtype == torch::kFloat16) {
    muillm_rmsnorm_fp16(
          stream,
          B,
          K,
          /* W */ nullptr,
          (const half*)x.data_ptr(),
          residual.defined() ? (const half*)residual.data_ptr() : nullptr,
          (half*)y.data_ptr(),
          epsilon,
          0.0f
      );
  } else {
    TORCH_CHECK(false, "muillm_l2norm_forward: unsupported dtype ");
  }

  return y;
}

// python trampoline implementation
at::Tensor muillm_l2norm_forward_trampoline(
    torch::Tensor x,
    std::optional<torch::Tensor> residual_,
    float epsilon
) {
  torch::Tensor residual = residual_.has_value() ? residual_.value() : torch::Tensor();
  return muillm_l2norm_forward(
      x,
      residual,
      epsilon
  );
}