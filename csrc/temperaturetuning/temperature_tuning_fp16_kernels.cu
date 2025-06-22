#include <hip/hip_fp16.h>

#include <stdint.h>

#define THREADS_PER_BLOCK 256

// x block is for the token dimension
// y block is for the heads dimension
// z block is for the batch dimension
__global__ void muillm_apply_temperature_tuning_fp16_kernel(
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
) {
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 num_blocks = dim3(T, num_heads, B);

  muillm_apply_temperature_tuning_fp16_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
    (const half*)q_in,
    (const int64_t*)cache_position,
    (half*)y,
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
}