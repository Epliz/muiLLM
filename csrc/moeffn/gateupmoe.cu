#include "moelinear.cuh"
#include "gateupmoe.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// Python trampolines

at::Tensor muillm_gateupmlpmoe_forward_trampoline(
  muillm_engine_ptr engine,
  int num_shared_experts,
  int num_dynamic_experts,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  float norm_weights_offset,
  torch::Tensor gate_weights,
  torch::Tensor up_weights,
  torch::Tensor down_weights,
  std::optional<torch::Tensor> residual_,
  torch::Tensor x,
  torch::Tensor router_scores,
  torch::Tensor router_indices
) {
  torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : torch::Tensor();
  torch::Tensor residual = residual_.has_value() ? residual_.value() : torch::Tensor();
  return muillm_gateupmlpmoe_forward(
      engine.engine_ptr,
      num_shared_experts,
      num_dynamic_experts,
      norm_weights,
      epsilon,
      norm_weights_offset,
      gate_weights,
      up_weights,
      down_weights,
      residual,
      x,
      router_scores,
      router_indices
  );
}

at::Tensor muillm_gateupmlpmoe_split_forward_trampoline(
  muillm_engine_ptr engine,
  int num_shared_experts,
  int num_dynamic_experts,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  float norm_weights_offset,
  torch::Tensor gate_weights,
  torch::Tensor up_weights,
  torch::Tensor down_weights,
  std::optional<torch::Tensor> residual_,
  torch::Tensor x,
  torch::Tensor router_scores,
  torch::Tensor router_indices
) {
  torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : torch::Tensor();
  torch::Tensor residual = residual_.has_value() ? residual_.value() : torch::Tensor();
  return muillm_gateupmlpmoe_split_forward(
      engine.engine_ptr,
      num_shared_experts,
      num_dynamic_experts,
      norm_weights,
      epsilon,
      norm_weights_offset,
      gate_weights,
      up_weights,
      down_weights,
      residual,
      x,
      router_scores,
      router_indices
  );
}


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void muillm_gateupmlpmoe_forward_fp16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  unsigned num_shared_experts,
  unsigned num_computed_experts,
  const half* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const half* gate_weights,
  const half* up_weights,
  const half* x,
  const half* router_scores,
  const int64_t* router_indices,
  half* y,
  int simd_lanes
);

void muillm_gateupmlpmoe_forward_bf16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  unsigned num_shared_experts,
  unsigned num_computed_experts,
  const __hip_bfloat16* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const __hip_bfloat16* gate_weights,
  const __hip_bfloat16* up_weights,
  const __hip_bfloat16* x,
  const __hip_bfloat16* router_scores,
  const int64_t* router_indices,
  __hip_bfloat16* y,
  int simd_lanes
);

void muillm_gateupmlpmoe_forward_placed_output(
    muillm_engine_t* engine,
    int num_shared_experts,
    int num_dynamic_experts,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor& gate_weights, // size ((num_shared_experts + num_dynamic_experts) * N) x K
    torch::Tensor& up_weights, // size ((num_shared_experts + num_dynamic_experts) * N) x K
    torch::Tensor& down_weights, // size ((num_shared_experts + num_dynamic_experts) * K) x N
    torch::Tensor& residual,
    torch::Tensor& x, // size B x T x K
    torch::Tensor& router_scores, // size B x T x num_routed_experts
    torch::Tensor& router_indices, // size B x T x num_routed_experts
    void* output_ptr) { // size B x T x K
  bool normalize = norm_weights.defined();
  if (normalize) {
    CHECK_INPUT(norm_weights);
  }
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(up_weights);
  CHECK_INPUT(x);
  CHECK_INPUT(router_scores);
  CHECK_INPUT(router_indices);

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = x.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  const auto B = x.size(0);
  const auto T = x.size(1);
  const auto N = gate_weights.size(0) / (num_shared_experts + num_dynamic_experts);
  const auto K = gate_weights.size(1);
  const auto num_routed_experts = router_scores.size(2);

  const int num_computed_experts = num_shared_experts + num_routed_experts;
   
  // y is the output of the gate/up part, and has shape B x T x num_computed_experts x N
  auto y = torch::empty({B, T, num_computed_experts, N}, output_options);

  int simd_lanes = engine->gpu_infos[0]->simd_lanes;

  if (dtype == torch::kFloat16) {
    muillm_gateupmlpmoe_forward_fp16(
        stream,
        N,
        K,
        num_shared_experts,
        num_computed_experts,
        normalize ? (const half*)norm_weights.data_ptr() : nullptr,
        epsilon,
        norm_weights_offset,
        (const half*)gate_weights.data_ptr(),
        (const half*)up_weights.data_ptr(),
        (const half*)x.data_ptr(),
        (const half*)router_scores.data_ptr(),
        (const int64_t*)router_indices.data_ptr(),
        (half*)y.data_ptr(),
        simd_lanes
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_gateupmlpmoe_forward_bf16(
        stream,
        N,
        K,
        num_shared_experts,
        num_computed_experts,
        normalize ? (__hip_bfloat16*)norm_weights.data_ptr() : nullptr,
        epsilon,
        norm_weights_offset,
        (__hip_bfloat16*)gate_weights.data_ptr(),
        (__hip_bfloat16*)up_weights.data_ptr(),
        (__hip_bfloat16*)x.data_ptr(),
        (__hip_bfloat16*)router_scores.data_ptr(),
        (const int64_t*)router_indices.data_ptr(),
        (__hip_bfloat16*)y.data_ptr(),
        simd_lanes
    );
  } else {
    TORCH_CHECK(false, "unsupported dtype for muillm_gateupmlpmoe_forward_placed_output");
  }

  // down proj
  auto undef_tensor = torch::Tensor();

  muillm_moelinear_activ_forward_placed_output(
      engine,
      num_shared_experts,
      num_dynamic_experts,
      undef_tensor /*norm_weights*/,
      0.f,
      down_weights,
      mui_activation::Identity,
      undef_tensor /*mul_bias*/,
      undef_tensor /*add_bias*/,
      residual,
      y,
      router_indices,
      output_ptr,
      stream
  );
}

at::Tensor muillm_gateupmlpmoe_forward(
    muillm_engine_t* engine,
    int num_shared_experts,
    int num_dynamic_experts,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x,
    torch::Tensor& router_scores,
    torch::Tensor& router_indices
) {
  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = x.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  const auto N = down_weights.size(0) / (num_shared_experts + num_dynamic_experts);

  // output has the same dimensions as x, except the last dim that is given by
  // the out_features of weights
  auto output_sizes = x.sizes().vec();
  output_sizes[output_sizes.size() - 1] = N;

  auto output = torch::empty(output_sizes, output_options);

  void* output_ptr = output.data_ptr();

  muillm_gateupmlpmoe_forward_placed_output(
    engine,
    num_shared_experts,
    num_dynamic_experts,
    norm_weights,
    epsilon,
    norm_weights_offset,
    gate_weights,
    up_weights,
    down_weights,
    residual,
    x,
    router_scores,
    router_indices,
    output_ptr
  );
  return output;
}

void muillm_gateupmlpmoe_split_forward_fp16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  unsigned num_shared_experts,
  unsigned num_computed_experts,
  const half* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const half* gate_weights,
  const half* up_weights,
  const half* x,
  const half* router_scores,
  const int64_t* router_indices,
  half* gy,
  half* uy,
  half* y,
  int simd_lanes
);

void muillm_gateupmlpmoe_split_forward_bf16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  unsigned num_shared_experts,
  unsigned num_computed_experts,
  const __hip_bfloat16* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const __hip_bfloat16* gate_weights,
  const __hip_bfloat16* up_weights,
  const __hip_bfloat16* x,
  const __hip_bfloat16* router_scores,
  const int64_t* router_indices,
  __hip_bfloat16* gy,
  __hip_bfloat16* uy,
  __hip_bfloat16* y,
  int simd_lanes
);

void muillm_gateupmlpmoe_split_forward_placed_output(
    muillm_engine_t* engine,
    int num_shared_experts,
    int num_dynamic_experts,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor& gate_weights, // size (num_shared_experts + num_dynamic_experts) x N x K
    torch::Tensor& up_weights, // size (num_shared_experts + num_dynamic_experts) x N x K
    torch::Tensor& down_weights, // size (num_shared_experts + num_dynamic_experts) x K x N
    torch::Tensor& residual,
    torch::Tensor& x, // size B x T x K
    torch::Tensor& router_scores, // size B x T x num_routed_experts
    torch::Tensor& router_indices, // size B x T x num_routed_experts
    void* output_ptr) { // size B x T x K
  bool normalize = norm_weights.defined();
  if (normalize) {
    CHECK_INPUT(norm_weights);
  }
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(up_weights);
  CHECK_INPUT(x);
  CHECK_INPUT(router_scores);
  CHECK_INPUT(router_indices);

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());


  const auto B = x.size(0);
  const auto T = x.size(1);
  const auto N = gate_weights.size(0) / (num_shared_experts + num_dynamic_experts);
  const auto K = gate_weights.size(1);
  const auto num_routed_experts = router_scores.size(2);

  const int num_computed_experts = num_shared_experts + num_routed_experts;

  auto dtype = x.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // output for gate projections
  auto gy = torch::empty({B, T, num_computed_experts, N}, output_options);
  // output for up projection
  auto uy = torch::empty({B, T, num_computed_experts, N}, output_options);
  // output for the reduction
  auto y = torch::empty({B, T, num_computed_experts, N}, output_options);

  int simd_lanes = engine->gpu_infos[0]->simd_lanes;

  if (dtype == torch::kFloat16) {
    muillm_gateupmlpmoe_split_forward_fp16(
        stream,
        N,
        K,
        num_shared_experts,
        num_computed_experts,
        normalize ? (const half*)norm_weights.data_ptr() : nullptr,
        epsilon,
        norm_weights_offset,
        (const half*)gate_weights.data_ptr(),
        (const half*)up_weights.data_ptr(),
        (const half*)x.data_ptr(),
        (const half*)router_scores.data_ptr(),
        (const int64_t*)router_indices.data_ptr(),
        (half*)gy.data_ptr(),
        (half*)uy.data_ptr(),
        (half*)y.data_ptr(),
        simd_lanes
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_gateupmlpmoe_split_forward_bf16(
        stream,
        N,
        K,
        num_shared_experts,
        num_computed_experts,
        normalize ? (__hip_bfloat16*)norm_weights.data_ptr() : nullptr,
        epsilon,
        norm_weights_offset,
        (__hip_bfloat16*)gate_weights.data_ptr(),
        (__hip_bfloat16*)up_weights.data_ptr(),
        (__hip_bfloat16*)x.data_ptr(),
        (__hip_bfloat16*)router_scores.data_ptr(),
        (const int64_t*)router_indices.data_ptr(),
        (__hip_bfloat16*)gy.data_ptr(),
        (__hip_bfloat16*)uy.data_ptr(),
        (__hip_bfloat16*)y.data_ptr(),
        simd_lanes
    );
  } else {
    TORCH_CHECK(false, "unsupported dtype for muillm_gateupmlpmoe_split_forward_placed_output");
  }

  // down proj
  auto undef_tensor = torch::Tensor();
  muillm_moelinear_activ_forward_placed_output(
      engine,
      num_shared_experts,
      num_dynamic_experts,
      undef_tensor /*norm_weights*/,
      0.f,
      down_weights,
      mui_activation::Identity,
      undef_tensor /*mul_bias*/,
      undef_tensor /*add_bias*/,
      residual,
      y,
      router_indices,
      output_ptr,
      stream
  );
}

at::Tensor muillm_gateupmlpmoe_split_forward(
    muillm_engine_t* engine,
    int num_shared_experts,
    int num_dynamic_experts,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x,
    torch::Tensor& router_scores,
    torch::Tensor& router_indices
) {

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = x.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  const auto N = down_weights.size(0) / (num_shared_experts + num_dynamic_experts);

  // output has the same dimensions as x, except the last dim that is given by
  // the out_features of weights
  auto output_sizes = x.sizes().vec();
  output_sizes[output_sizes.size() - 1] = N;

  auto output = torch::empty(output_sizes, output_options);

  void* output_ptr = output.data_ptr();
  
  muillm_gateupmlpmoe_split_forward_placed_output(
    engine,
    num_shared_experts,
    num_dynamic_experts,
    norm_weights,
    epsilon,
    norm_weights_offset,
    gate_weights,
    up_weights,
    down_weights,
    residual,
    x,
    router_scores,
    router_indices,
    output_ptr
  );

  return output;
}