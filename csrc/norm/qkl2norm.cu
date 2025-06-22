#include "qkl2norm.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

void muillm_qkl2norm_fp16(
  hipStream_t stream,
  unsigned BQ,
  unsigned BK,
  unsigned N,
  const half* q,
  const half* k,
  half* q_norm,
  half* k_norm,
  float epsilon
);

void muillm_qkl2norm_bf16(
  hipStream_t stream,
  unsigned BQ,
  unsigned BK,
  unsigned N,
  const __hip_bfloat16* q,
  const __hip_bfloat16* k,
  __hip_bfloat16* q_norm,
  __hip_bfloat16* k_norm,
  float epsilon
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<at::Tensor, at::Tensor> muillm_qkl2norm_forward(
    torch::Tensor q,
    torch::Tensor k,
    float epsilon) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);

  auto device = q.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  // q and k have different sizes, but the last dimension must be the same
  const auto NQ = q.size(q.dim() - 1);
  const auto NK = q.size(q.dim() - 1);
  TORCH_CHECK(NQ == NK, "The last dimension of q and k must be the same");
  const auto N = NQ; // last dimension size, same for q and k

  // batch size
  // TODO: is numel slow?
  const auto BQ = q.numel() / N;
  const auto BK = k.numel() / N;

  // q and k have different sizes
  auto qoutput_sizes = q.sizes().vec();
  auto koutput_sizes = k.sizes().vec();

  auto dtype = q.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto q_norm = torch::empty(qoutput_sizes, output_options);
  auto k_norm = torch::empty(koutput_sizes, output_options);

  if (dtype == torch::kFloat16) {
    muillm_qkl2norm_fp16(
      stream,
      BQ,
      BK,
      N,
      (const half*)q.data_ptr(),
      (const half*)k.data_ptr(),
      (half*)q_norm.data_ptr(),
      (half*)k_norm.data_ptr(),
      epsilon
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_qkl2norm_bf16(
      stream,
      BQ,
      BK,
      N,
      (__hip_bfloat16*)q.data_ptr(),
      (__hip_bfloat16*)k.data_ptr(),
      (__hip_bfloat16*)q_norm.data_ptr(),
      (__hip_bfloat16*)k_norm.data_ptr(),
      epsilon
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type for q and k");
  }

  return std::make_tuple(q_norm, k_norm);
}