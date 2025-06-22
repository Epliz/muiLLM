#include "temperature_tuning.cuh"
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <stdint.h>

void muillm_apply_temperature_tuning_fp16(
  hipStream_t stream,
  unsigned B,
  unsigned T,
  unsigned num_heads,
  unsigned embed_dim,
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  const half* q_in,
  const int64_t* cache_position,
  half* y,
  float attn_scale,
  float floor_scale
);

void muillm_apply_temperature_tuning_bf16(
  hipStream_t stream,
  unsigned B,
  unsigned T,
  unsigned num_heads,
  unsigned embed_dim,
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  const __hip_bfloat16* q_in,
  const int64_t* cache_position,
  __hip_bfloat16* y,
  float attn_scale,
  float floor_scale
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor muillm_apply_temperature_tuning(
    at::Tensor q_in, // [B, num_heads, T, embed_dim]
    at::Tensor cache_position, // [T]
    float attn_scale,
    float floor_scale
) {
  CHECK_CUDA(q_in);
  CHECK_INPUT(cache_position);
  TORCH_CHECK(cache_position.scalar_type() == at::kLong, "cache_position must be int64");

  auto device = q_in.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = q_in.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto q_sizes = q_in.sizes();
  auto q_strides = q_in.strides().vec();

  const auto B = q_sizes[0];
  const auto num_heads = q_sizes[1];
  const auto T = q_sizes[2];
  const auto embed_dim = q_sizes[3];

  // q strides
  unsigned q_in_batch_stride = q_strides[0];
  unsigned q_in_head_stride = q_strides[1];
  unsigned q_in_tok_stride = q_strides[2];

  // y has the same dimensions as query_states
  auto y = torch::empty(q_sizes, output_options);

  if (dtype == torch::kBFloat16) {
    // bfloat16 case
    muillm_apply_temperature_tuning_bf16(
      stream,
      B,
      T,
      num_heads,
      embed_dim,
      q_in_batch_stride,
      q_in_head_stride,
      q_in_tok_stride,
      (const __hip_bfloat16*)q_in.data_ptr(),
      (const int64_t*)cache_position.data_ptr(),
      (__hip_bfloat16*)y.data_ptr(),
      attn_scale,
      floor_scale
    );
    return y;
  }
  else if (dtype == torch::kFloat16) {
    // float16 case
    muillm_apply_temperature_tuning_fp16(
      stream,
      B,
      T,
      num_heads,
      embed_dim,
      q_in_batch_stride,
      q_in_head_stride,
      q_in_tok_stride,
      (const half*)q_in.data_ptr(),
      (const int64_t*)cache_position.data_ptr(),
      (half*)y.data_ptr(),
      attn_scale,
      floor_scale
    );
    return y;
  } else {
    TORCH_CHECK(false, "Unsupported dtype for temperature tuning");
  }

  return y;
}
