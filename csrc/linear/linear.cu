#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "linear.cuh"
#include "../utils/torch_utils.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// Python trampoline

at::Tensor muillm_linear_forward_trampoline(
  muillm_engine_ptr engine,
  torch::Tensor x,
  torch::Tensor weights,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  float norm_weights_offset,
  std::optional<torch::Tensor> add_bias_,
  std::optional<torch::Tensor> mul_residual_,
  std::optional<torch::Tensor> residual_) {
  auto undef_tensor = torch::Tensor();

  torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : undef_tensor;
  torch::Tensor mul_residual = mul_residual_.has_value() ? mul_residual_.value() : undef_tensor;
  torch::Tensor add_bias = add_bias_.has_value() ? add_bias_.value() : undef_tensor;
  torch::Tensor residual = residual_.has_value() ? residual_.value() : undef_tensor;
  return muillm_linear_activ_forward(
      engine.engine_ptr,
      norm_weights,
      epsilon,
      norm_weights_offset,
      weights,
      mui_activation::Identity,
      add_bias,
      mul_residual,
      residual,
      x
  );
}


void muillm_linear_activ_forward_fp16(
  hipStream_t stream,
  unsigned B,
  unsigned N,
  unsigned K,
  unsigned xK, // stride of the input X (in case it is not contiguous)
  const half* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const half* weights,
  mui_activation activ,
  const half* add_bias,
  const half* mul_residual,
  const half* residual,
  const half* x,
  half* y,
  int warp_size
);

void muillm_linear_activ_forward_bf16(
  hipStream_t stream,
  unsigned B,
  unsigned N,
  unsigned K,
  unsigned xK, // stride of the input X (in case it is not contiguous)
  const __hip_bfloat16* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const __hip_bfloat16* weights,
  mui_activation activ,
  const __hip_bfloat16* add_bias,
  const __hip_bfloat16* mul_residual,
  const __hip_bfloat16* residual,
  const __hip_bfloat16* x,
  __hip_bfloat16* y,
  int warp_size
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void muillm_linear_activ_forward_placed_output(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor& weights, // N x K
    mui_activation activ,
    torch::Tensor& add_bias,
    torch::Tensor& mul_residual,
    torch::Tensor& residual,
    torch::Tensor& x, // B x K
    void* output_ptr,
    hipStream_t stream) {
  bool normalize = norm_weights.defined();
  if (normalize) {
    CHECK_INPUT(norm_weights);
  }
  CHECK_INPUT(weights);
  if (add_bias.defined()) {
    CHECK_INPUT(add_bias);
  }
  if (mul_residual.defined()) {
    CHECK_INPUT(mul_residual);
  }
  if (residual.defined()) {
    CHECK_INPUT(residual);
  }
  CHECK_CUDA(x);

  bool x_mostly_contiguous = muillm_torch_tensor_is_mostly_contiguous(x);
  TORCH_CHECK(x_mostly_contiguous, "x must be mostly contiguous (contiguous in the last dimension)");

  auto dtype = x.dtype();

  const auto N = weights.size(0);
  const auto K = weights.size(1);

  const auto Kx = x.size(x.dim() - 1);
  TORCH_CHECK(K == Kx, "weights.size(1) must match x.size(-1)");

  const auto ndim = x.dim();
  // stride of the input X (in case it is not contiguous)
  // (if x is a vector, we can set it to K, but if x is a matrix,
  // we need to use the stride of the last dimension)
  const auto xK = ndim <= 1 ? Kx : x.stride(x.dim() - 2);

  const auto B = x.numel() / K;

  // try to occupy enough to saturate memory bandwidth
  /*
  int simd_lanes = engine->gpu_infos[0]->simd_lanes;

  while ((num_blocks * threads_per_blocks < 8 * simd_lanes) && threads_per_blocks < 256) {
    threads_per_blocks *= 2;
  }
  */

  int warp_size = engine->gpu_infos[0]->warp_size;

  if (normalize) {
    const auto NORM_K = norm_weights.size(0);
    TORCH_CHECK(K == NORM_K, "fused normalization is not supported when sharding on dim 1 (K != NORM_K)");
  }

  if (dtype == torch::kFloat16) {
    muillm_linear_activ_forward_fp16(
      stream,
      B,
      N,
      K,
      xK,
      norm_weights.defined() ? (const half*)norm_weights.data_ptr() : nullptr,
      epsilon,
      norm_weights_offset,
      (const half*)weights.data_ptr(),
      activ,
      add_bias.defined() ? (const half*)add_bias.data_ptr() : nullptr,
      mul_residual.defined() ? (const half*)mul_residual.data_ptr() : nullptr,
      residual.defined() ? (const half*)residual.data_ptr() : nullptr,
      (const half*)x.data_ptr(),
      (half*)output_ptr,
      warp_size
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_linear_activ_forward_bf16(
      stream,
      B,
      N,
      K,
      xK,
      norm_weights.defined() ? (const __hip_bfloat16*)norm_weights.data_ptr() : nullptr,
      epsilon,
      norm_weights_offset,
      (const __hip_bfloat16*)weights.data_ptr(),
      activ,
      add_bias.defined() ? (const __hip_bfloat16*)add_bias.data_ptr() : nullptr,
      mul_residual.defined() ? (const __hip_bfloat16*)mul_residual.data_ptr() : nullptr,
      residual.defined() ? (const __hip_bfloat16*)residual.data_ptr() : nullptr,
      (const __hip_bfloat16*)x.data_ptr(),
      (__hip_bfloat16*)output_ptr,
      warp_size
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for linear");
  }
}

at::Tensor muillm_linear_activ_forward(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor& weights,
    mui_activation activ,
    torch::Tensor& add_bias,
    torch::Tensor& mul_residual,
    torch::Tensor& residual,
    torch::Tensor& x) {
  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  const auto N = weights.size(0);

  auto dtype = x.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // y has the same dimensions as x, except the last dim that is given by
  // the out_features of weights
  auto output_sizes = x.sizes().vec();
  output_sizes[output_sizes.size() - 1] = N;

  auto y = torch::empty(output_sizes, output_options);

  void* output_ptr = y.data_ptr();

  muillm_linear_activ_forward_placed_output(
    engine,
    norm_weights,
    epsilon,
    norm_weights_offset,
    weights,
    activ,
    add_bias,
    mul_residual,
    residual,
    x,
    output_ptr,
    stream
  );

  return y;
}