#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include "moelinear.cuh"
#include "../reduce/reduce.cuh"

// Python trampoline

at::Tensor muillm_moelinear_forward_trampoline(
  muillm_engine_ptr engine,
  int num_shared_experts,
  int num_dynamic_experts,
  torch::Tensor x,
  torch::Tensor router_indices,
  torch::Tensor weights,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  std::optional<torch::Tensor> mul_bias_,
  std::optional<torch::Tensor> add_bias_,
  std::optional<torch::Tensor> residual_) {
  auto undef_tensor = torch::Tensor();

  torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : undef_tensor;
  torch::Tensor mul_bias = mul_bias_.has_value() ? mul_bias_.value() : undef_tensor;
  torch::Tensor add_bias = add_bias_.has_value() ? add_bias_.value() : undef_tensor;
  torch::Tensor residual = residual_.has_value() ? residual_.value() : undef_tensor;
  return muillm_moelinear_activ_forward(
      engine.engine_ptr,
      num_shared_experts,
      num_dynamic_experts,
      norm_weights,
      epsilon,
      weights,
      mui_activation::Identity,
      mul_bias,
      add_bias,
      residual,
      x,
      router_indices
  );
}

//
// actual module
//


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void muillm_moelinear_activ_forward_fp16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  unsigned num_shared_experts,
  unsigned num_computed_experts,
  const half* norm_weights,
  float epsilon,
  const half* weights, // size ((num_shared_experts + num_dynamic_experts) * N) x K
  mui_activation activ,
  const half* mul_bias,
  const half* add_bias,
  const half* residual,
  const half* x, // size B x T x (num_shared_experts + num_routed_experts) x K
  const int64_t* router_indices,
  half* linear_output_ptr, // size B x T x N
  int simd_lanes
);

void muillm_moelinear_activ_forward_bf16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  unsigned num_shared_experts,
  unsigned num_computed_experts,
  const __hip_bfloat16* norm_weights,
  float epsilon,
  const __hip_bfloat16* weights, // size ((num_shared_experts + num_dynamic_experts) * N) x K
  mui_activation activ,
  const __hip_bfloat16* mul_bias,
  const __hip_bfloat16* add_bias,
  const __hip_bfloat16* residual,
  const __hip_bfloat16* x, // size B x T x (num_shared_experts + num_routed_experts) x K
  const int64_t* router_indices,
  __hip_bfloat16* linear_output_ptr, // size B x T x N
  int simd_lanes
);


void muillm_moelinear_activ_forward_placed_output(
    muillm_engine_t* engine,
    int num_shared_experts,
    int num_dynamic_experts,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& weights, // size ((num_shared_experts + num_dynamic_experts) * N) x K
    mui_activation activ,
    torch::Tensor& mul_bias,
    torch::Tensor& add_bias,
    torch::Tensor& residual,
    torch::Tensor& x, // size B x T x (num_shared_experts + num_routed_experts) x K
    torch::Tensor& router_indices, // size B x T x num_routed_experts
    void* output_ptr, // size B x T x N
    hipStream_t stream) {

  bool normalize = norm_weights.defined();
  if (normalize) {
    CHECK_INPUT(norm_weights);
  }
  CHECK_INPUT(weights);
  if (mul_bias.defined()) {
    CHECK_INPUT(mul_bias);
  }
  if (add_bias.defined()) {
    CHECK_INPUT(add_bias);
  }
  if (residual.defined()) {
    CHECK_INPUT(residual);
  }
  CHECK_INPUT(x);

  const auto N = weights.size(0) / (num_shared_experts + num_dynamic_experts);
  const auto K = weights.size(1);
  const auto num_routed_experts = router_indices.size(2);

  const int num_computed_experts = num_shared_experts + num_routed_experts;

  auto device = x.device();
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

  int simd_lanes = engine->gpu_infos[0]->simd_lanes;

  bool needs_reduction = (num_computed_experts > 1);
  auto reduction_buffer = needs_reduction ? torch::empty(output_sizes, output_options) : torch::Tensor();

  half* linear_output_ptr = needs_reduction ? (half*) reduction_buffer.data_ptr() : (half*)output_ptr;

  if (normalize) {
    const auto NORM_K = norm_weights.size(0);
    TORCH_CHECK(K == NORM_K, "fused normalization is not supported when sharding on dim 1 (K != NORM_K)");
  }

  if (dtype == torch::kFloat16) {
    // use fp16 kernels
    muillm_moelinear_activ_forward_fp16(
      stream,
      N,
      K,
      num_shared_experts,
      num_computed_experts,
      normalize ? (const half*)norm_weights.data_ptr() : nullptr,
      epsilon,
      (const half*)weights.data_ptr(),
      activ,
      mul_bias.defined() ? (const half*) mul_bias.data_ptr() : nullptr,
      add_bias.defined() ? (const half*) add_bias.data_ptr() : nullptr,
      residual.defined() ? (const half*) residual.data_ptr() : nullptr,
      (half*)x.data_ptr(),
      (const int64_t*)router_indices.data_ptr(),
      linear_output_ptr,
      simd_lanes
    );
  } else if (dtype == torch::kBFloat16) {
    // use bf16 kernels
    muillm_moelinear_activ_forward_bf16(
      stream,
      N,
      K,
      num_shared_experts,
      num_computed_experts,
      normalize ? (__hip_bfloat16*)norm_weights.data_ptr() : nullptr,
      epsilon,
      (__hip_bfloat16*)weights.data_ptr(),
      activ,
      mul_bias.defined() ? (__hip_bfloat16*) mul_bias.data_ptr() : nullptr,
      add_bias.defined() ? (__hip_bfloat16*) add_bias.data_ptr() : nullptr,
      residual.defined() ? (__hip_bfloat16*) residual.data_ptr() : nullptr,
      (__hip_bfloat16*)x.data_ptr(),
      (const int64_t*)router_indices.data_ptr(),
      (__hip_bfloat16*)linear_output_ptr,
      simd_lanes
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for moelinear");
  }

  // reduce if necessary
  if (needs_reduction) {
    // we need to reduce the output across experts
    muillm_reduce_sum_forward_placed_output(
      reduction_buffer,
      -2 /* dim */,
      false /* keep_dim */,
      output_ptr,
      stream
    );
  }
}

at::Tensor muillm_moelinear_activ_forward(
    muillm_engine_t* engine,
    int num_shared_experts,
    int num_dynamic_experts,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& weights, // size ((num_shared_experts + num_dynamic_experts) x N) x K
    mui_activation activ,
    torch::Tensor& mul_bias,
    torch::Tensor& add_bias,
    torch::Tensor& residual,
    torch::Tensor& x, // size B x T x (num_shared_experts + num_routed_experts) x K
    torch::Tensor& router_indices // size B x T x num_routed_experts
) {
  CHECK_INPUT(x);

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  const auto N = weights.size(0) / (num_shared_experts + num_dynamic_experts);

  auto dtype = x.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // y has the same dimensions as router_indices, except the last dim that is given by
  // the out_features of weights
  auto output_sizes = router_indices.sizes().vec();
  output_sizes[output_sizes.size() - 1] = N;

  auto y = torch::empty(output_sizes, output_options);

  void* output_ptr = y.data_ptr();

  muillm_moelinear_activ_forward_placed_output(
    engine,
    num_shared_experts,
    num_dynamic_experts,
    norm_weights,
    epsilon,
    weights,
    activ,
    mul_bias,
    add_bias,
    residual,
    x,
    router_indices,
    output_ptr,
    stream
  );

  return y;
}