#include "../linear/linear.cuh"
#include "gateup.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// Python trampoline

at::Tensor muillm_gateupmlp_forward_trampoline(
  muillm_engine_ptr engine,
  int activation,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  float norm_weights_offset,
  torch::Tensor gate_weights,
  torch::Tensor up_weights,
  torch::Tensor down_weights,
  std::optional<torch::Tensor> residual_,
  torch::Tensor x) {
  torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : torch::Tensor();
  torch::Tensor residual = residual_.has_value() ? residual_.value() : torch::Tensor();
  return muillm_gateupmlp_forward(
      engine.engine_ptr,
      static_cast<MuiGateUpMLPActivation>(activation),
      norm_weights,
      epsilon,
      norm_weights_offset,
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
  MuiGateUpMLPActivation activation,
  unsigned B,
  unsigned N,
  unsigned K,
  const half* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const half* gate_weights,
  const half* up_weights,
  const half* x,
  half* y,
  int warp_size
);

void muillm_gateupmlp_forward_bf16(
  hipStream_t stream,
  MuiGateUpMLPActivation activation,
  unsigned B,
  unsigned N,
  unsigned K,
  const __hip_bfloat16* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const __hip_bfloat16* gate_weights,
  const __hip_bfloat16* up_weights,
  const __hip_bfloat16* x,
  __hip_bfloat16* y,
  int warp_size
);

void muillm_gateupmlp_forward_placed_output(
    muillm_engine_t* engine,
    MuiGateUpMLPActivation activation,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
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
  const auto Kx = x.size(x.dim() - 1);
  TORCH_CHECK(K == Kx, "gate_weights.size(1) must match x.size(-1)");
  const auto B = x.numel() / K;
  TORCH_CHECK(B <= MUILLM_GATEUP_KERNELS_MAX_BATCH_SIZE, "Unsupported batch size for fused gateup kernels");

  if (normalize) {
    const auto norm_k = norm_weights.size(0);
    TORCH_CHECK(K == norm_k, "fused normalization is not supported when sharding on dim 1 (K != norm_weights.size(0))");
  }

  // y has the same dimensions as x, except the last dim that is given by
  // the out_features of weights
  auto output_sizes = x.sizes().vec();
  output_sizes[output_sizes.size() - 1] = N;

  auto y = torch::empty(output_sizes, output_options);

  int warp_size = engine->gpu_infos[0]->warp_size;

  if (dtype == torch::kFloat16) {
    muillm_gateupmlp_forward_fp16(
        stream,
        activation,
        B,
        N,
        K,
        normalize ? (const half*)norm_weights.data_ptr() : nullptr,
        epsilon,
        norm_weights_offset,
        (const half*)gate_weights.data_ptr(),
        (const half*)up_weights.data_ptr(),
        (const half*)x.data_ptr(),
        (half*)y.data_ptr(),
        warp_size
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_gateupmlp_forward_bf16(
        stream,
        activation,
        B,
        N,
        K,
        normalize ? (const __hip_bfloat16*)norm_weights.data_ptr() : nullptr,
        epsilon,
        norm_weights_offset,
        (const __hip_bfloat16*)gate_weights.data_ptr(),
        (const __hip_bfloat16*)up_weights.data_ptr(),
        (const __hip_bfloat16*)x.data_ptr(),
        (__hip_bfloat16*)y.data_ptr(),
        warp_size
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for gateupmlp");
  }

  // down proj
  auto undef_tensor = torch::Tensor();

  muillm_linear_activ_forward_placed_output(
      engine,
      undef_tensor /*norm_weights*/,
      0.f, /* epsilon */
      0.f, /* norm_weights_offset */
      down_weights,
      mui_activation::Identity,
      undef_tensor /*add_bias*/,
      undef_tensor/*mul_residual*/,
      residual,
      y,
      output_ptr,
      stream
  );
}

at::Tensor muillm_gateupmlp_forward(
    muillm_engine_t* engine,
    MuiGateUpMLPActivation activation,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
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
    activation,
    norm_weights,
    epsilon,
    norm_weights_offset,
    gate_weights,
    up_weights,
    down_weights,
    residual,
    x,
    output_ptr
  );

  return output;
}