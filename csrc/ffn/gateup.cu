#include "../linear/linear.cuh"
#include "gateup.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// Python trampoline

at::Tensor muillm_gateupmlp_forward_trampoline(
  muillm_engine_ptr engine,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  torch::Tensor gate_weights,
  torch::Tensor up_weights,
  torch::Tensor down_weights,
  std::optional<torch::Tensor> residual_,
  torch::Tensor x) {
  torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : torch::Tensor();
  torch::Tensor residual = residual_.has_value() ? residual_.value() : torch::Tensor();
  return muillm_gateupmlp_forward(
      engine.engine_ptr,
      norm_weights,
      epsilon,
      gate_weights,
      up_weights,
      down_weights,
      residual,
      x
  );
}

at::Tensor muillm_gateupmlp_split_forward_trampoline(
  muillm_engine_ptr engine,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  torch::Tensor gate_weights,
  torch::Tensor up_weights,
  torch::Tensor down_weights,
  std::optional<torch::Tensor> residual_,
  torch::Tensor x) {
  torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : torch::Tensor();
  torch::Tensor residual = residual_.has_value() ? residual_.value() : torch::Tensor();
  return muillm_gateupmlp_split_forward(
      engine.engine_ptr,
      norm_weights,
      epsilon,
      gate_weights,
      up_weights,
      down_weights,
      residual,
      x
  );
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void muillm_gateupmlp_forward_fp16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  const half* norm_weights,
  float epsilon,
  const half* gate_weights,
  const half* up_weights,
  const half* x,
  half* y,
  int simd_lanes
);

void muillm_gateupmlp_forward_bf16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  const __hip_bfloat16* norm_weights,
  float epsilon,
  const __hip_bfloat16* gate_weights,
  const __hip_bfloat16* up_weights,
  const __hip_bfloat16* x,
  __hip_bfloat16* y,
  int simd_lanes
);

void muillm_gateupmlp_forward_placed_output(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x,
    void* output_ptr) {
  bool normalize = norm_weights.defined();
  if (normalize) {
    CHECK_INPUT(norm_weights);
  }
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(up_weights);
  CHECK_INPUT(x);

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = x.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  const auto N = gate_weights.size(0);
  const auto K = gate_weights.size(1);

  // y has the same dimensions as x, except the last dim that is given by
  // the out_features of weights
  auto output_sizes = x.sizes().vec();
  output_sizes[output_sizes.size() - 1] = N;

  auto y = torch::empty(output_sizes, output_options);

  int simd_lanes = engine->gpu_infos[0]->simd_lanes;

  if (dtype == torch::kFloat16) {
    muillm_gateupmlp_forward_fp16(
        stream,
        N,
        K,
        normalize ? (const half*)norm_weights.data_ptr() : nullptr,
        epsilon,
        (const half*)gate_weights.data_ptr(),
        (const half*)up_weights.data_ptr(),
        (const half*)x.data_ptr(),
        (half*)y.data_ptr(),
        simd_lanes
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_gateupmlp_forward_bf16(
        stream,
        N,
        K,
        normalize ? (const __hip_bfloat16*)norm_weights.data_ptr() : nullptr,
        epsilon,
        (const __hip_bfloat16*)gate_weights.data_ptr(),
        (const __hip_bfloat16*)up_weights.data_ptr(),
        (const __hip_bfloat16*)x.data_ptr(),
        (__hip_bfloat16*)y.data_ptr(),
        simd_lanes
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for gateupmlp");
  }

  // down proj
  auto undef_tensor = torch::Tensor();

  muillm_linear_activ_forward_placed_output(
      engine,
      undef_tensor /*norm_weights*/,
      epsilon,
      down_weights,
      mui_activation::Identity,
      undef_tensor /*mul_bias*/,
      undef_tensor/*add_bias*/,
      residual,
      y,
      output_ptr,
      stream
  );
}

at::Tensor muillm_gateupmlp_forward(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x) {
  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = x.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  const auto N = down_weights.size(0);

  // output has the same dimensions as x, except the last dim that is given by
  // the out_features of weights
  auto output_sizes = x.sizes().vec();
  output_sizes[output_sizes.size() - 1] = N;

  auto output = torch::empty(output_sizes, output_options);

  void* output_ptr = output.data_ptr();

  muillm_gateupmlp_forward_placed_output(
    engine,
    norm_weights,
    epsilon,
    gate_weights,
    up_weights,
    down_weights,
    residual,
    x,
    output_ptr
  );

  return output;
}

void muillm_gateupmlp_split_forward_fp16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  const half* norm_weights,
  float epsilon,
  const half* gate_weights,
  const half* up_weights,
  const half* x,
  half* gy,
  half* uy,
  half* y,
  int simd_lanes
);

void muillm_gateupmlp_split_forward_bf16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  const __hip_bfloat16* norm_weights,
  float epsilon,
  const __hip_bfloat16* gate_weights,
  const __hip_bfloat16* up_weights,
  const __hip_bfloat16* x,
  __hip_bfloat16* gy,
  __hip_bfloat16* uy,
  __hip_bfloat16* y,
  int simd_lanes
);

void muillm_gateupmlp_split_forward_placed_output(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x,
    void* output_ptr) {
  bool normalize = norm_weights.defined();
  if (normalize) {
    CHECK_INPUT(norm_weights);
  }
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(up_weights);
  CHECK_INPUT(x);

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  const auto N = gate_weights.size(0);
  const auto K = gate_weights.size(1);

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

  // output for gate projections
  auto gy = torch::empty(output_sizes, output_options);
  // output for up projection
  auto uy = torch::empty(output_sizes, output_options);
  // output for the reduction
  auto y = torch::empty(output_sizes, output_options);

  int simd_lanes = engine->gpu_infos[0]->simd_lanes;

  if (dtype == torch::kFloat16) {
    muillm_gateupmlp_split_forward_fp16(
        stream,
        N,
        K,
        normalize ? (const half*)norm_weights.data_ptr() : nullptr,
        epsilon,
        (const half*)gate_weights.data_ptr(),
        (const half*)up_weights.data_ptr(),
        (const half*)x.data_ptr(),
        (half*)gy.data_ptr(),
        (half*)uy.data_ptr(),
        (half*)y.data_ptr(),
        simd_lanes
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_gateupmlp_split_forward_bf16(
        stream,
        N,
        K,
        normalize ? (const __hip_bfloat16*)norm_weights.data_ptr() : nullptr,
        epsilon,
        (const __hip_bfloat16*)gate_weights.data_ptr(),
        (const __hip_bfloat16*)up_weights.data_ptr(),
        (const __hip_bfloat16*)x.data_ptr(),
        (__hip_bfloat16*)gy.data_ptr(),
        (__hip_bfloat16*)uy.data_ptr(),
        (__hip_bfloat16*)y.data_ptr(),
        simd_lanes
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for split gateupmlp");
  }

  // down proj
  auto undef_tensor = torch::Tensor();
  muillm_linear_activ_forward_placed_output(
      engine,
      undef_tensor /*norm_weights*/,
      epsilon,
      down_weights,
      mui_activation::Identity,
      undef_tensor /*mul_bias*/,
      undef_tensor/*add_bias*/,
      residual,
      y,
      output_ptr,
      stream
  );
}

at::Tensor muillm_gateupmlp_split_forward(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x) {

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = x.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  const auto N = down_weights.size(0);

  // output has the same dimensions as x, except the last dim that is given by
  // the out_features of weights
  auto output_sizes = x.sizes().vec();
  output_sizes[output_sizes.size() - 1] = N;

  auto output = torch::empty(output_sizes, output_options);

  void* output_ptr = output.data_ptr();
  
  muillm_gateupmlp_split_forward_placed_output(
    engine,
    norm_weights,
    epsilon,
    gate_weights,
    up_weights,
    down_weights,
    residual,
    x,
    output_ptr
  );

  return output;
}