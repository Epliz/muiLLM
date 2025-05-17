#include "temperature_tuning_kernels.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include <stdint.h>

#define THREADS_PER_BLOCK 256

// x block is for the token dimension
// y block is for the heads dimension
// z block is for the batch dimension
__global__ void muillm_apply_temperature_tuning_kernel(
  const half* __restrict__ q_in, // [B, num_heads, T, embed_dim]
  const int64_t* cache_position,
  half* __restrict__ q_out, // [B, num_heads, T, embed_dim]
  float attn_scale,
  float floor_scale,
  unsigned B,
  unsigned num_heads,
  unsigned T,
  unsigned embed_dim,
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride
) {
  const int tok_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int batch_idx = blockIdx.z;

  const int64_t cache_pos = cache_position[tok_idx];

  // compute temperature tuning scale
  float attention_scale = logf(floorf(float(cache_pos + 1) / floor_scale) + 1.0f) * attn_scale + 1.0f;

  // realign the pointers
  unsigned realigned_input_pos = (batch_idx * q_in_batch_stride) + (head_idx * q_in_head_stride) + (tok_idx * q_in_tok_stride);
  q_in = &q_in[realigned_input_pos];

  unsigned realigned_output_pos = ((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim;
  q_out = &q_out[realigned_output_pos];

  // apply to the query states
  for (int i = threadIdx.x; i < embed_dim; i += THREADS_PER_BLOCK) {
    // apply the temperature tuning scale
    q_out[i] = __float2half(__half2float(q_in[i]) * attention_scale);
  }
}


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
  TORCH_CHECK(q_in.scalar_type() == at::kHalf, "query_states must be float16");
  TORCH_CHECK(cache_position.scalar_type() == at::kLong, "cache_position must be int64");

  auto device = q_in.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = torch::kFloat16;
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

  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 num_blocks = dim3(T, num_heads, B);

  muillm_apply_temperature_tuning_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
    (const half*)q_in.data_ptr(),
    (const int64_t*)cache_position.data_ptr(),
    (half*)y.data_ptr(),
    attn_scale,
    floor_scale,
    B,
    num_heads,
    T,
    embed_dim,
    // q strides
    q_in_batch_stride,
    q_in_head_stride,
    q_in_tok_stride
  );

  return y;
}
