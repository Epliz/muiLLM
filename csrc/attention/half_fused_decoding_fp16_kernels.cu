#include <hip/hip_fp16.h>

#include <stdint.h>
#include <vector>
#include <algorithm>
#include <cmath>

#define THREADS_PER_BLOCK 256

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / ((b)))

template <typename T>
static inline const T* __device__ addr(const T* p, unsigned index) {
  // helps the AMDGPU compiler understand it can use the sgrp pair + single vgpr addressing mode
  unsigned byte_offset = sizeof(T) * index;
  const uint8_t* p8 = (const uint8_t*)p;
  return (const T*) (p8 + byte_offset);
}

template <typename T>
static inline T* __device__ addr(T* p, unsigned index) {
  // helps the AMDGPU compiler understand it can use the sgrp pair + single vgpr addressing mode
  unsigned byte_offset = sizeof(T) * index;
  uint8_t* p8 = (uint8_t*)p;
  return (T*) (p8 + byte_offset);
}

#ifdef  __CUDA_ARCH__
#define __xx_shfl_down(mask, val, offset) __shfl_down_sync(mask, val, offset)
#define __readfirstlane(v) (v)
#elif defined(__HIP_PLATFORM_AMD__) // AMD
#define __xx_shfl_down(mask, val, offset) __shfl_down(val, offset)
#define __readfirstlane(v) __builtin_amdgcn_readfirstlane(v)
#else
#error "Unsupported compiler"
#endif

static inline __device__ float warpReduce(float val) {
  if (warpSize == 32) {
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __xx_shfl_down(FULL_MASK32, val, offset);
    }
  }
  if (warpSize == 64) {
    for (int offset = 32; offset > 0; offset /= 2)
      val += __xx_shfl_down(FULL_MASK64, val, offset);

  }
  return val;
}

static inline __device__ float warpReduceMax(float val) {
  if (warpSize == 32) {
    for (int offset = 16; offset > 0; offset /= 2) {
      float v = __xx_shfl_down(FULL_MASK32, val, offset);
      val = std::max(v, val);
    }
  }
  if (warpSize == 64) {
    for (int offset = 32; offset > 0; offset /= 2) {
      float v = __xx_shfl_down(FULL_MASK64, val, offset);
      val = std::max(v, val);
    }

  }
  return val;
}

static inline unsigned __device__ uniform(unsigned v) {
  // make a value be uniform (i.e. use SGPRs)
  return __readfirstlane(v);
}

static inline float __device__ uniformf(float v) {
  // make a value be uniform (i.e. use SGPRs)
  // need to go through those casting hoops as there 
  // there is no builtin for floats
  unsigned vi = __readfirstlane(*((const unsigned*)&v));
  return *((const float*) &vi);
}


// expected block dimensions: [x=T, y=num_q_heads, z=B]
void __global__ causally_compute_transformer_softmax_scores_fp16_kernel(
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, S, embed_dim]
  float* __restrict__ temp_scores_f32, // [B, num_q_heads, T, S]
  half* __restrict__ scores_out, // [B, num_q_heads, T, S]
  // tensor dimension sizes
  unsigned T, // num tokens to decode
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_k_heads,
  float attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
  // k strides
  unsigned k_batch_stride,
  unsigned k_head_stride,
  unsigned k_tok_stride
) {
  // one block does one head of a new token
  unsigned q_head_idx = blockIdx.y;
  unsigned tok_idx = blockIdx.x;
  unsigned batch_idx = blockIdx.z;

  unsigned k_head_idx = q_head_idx / q_to_k_heads;

  unsigned warps_per_block = THREADS_PER_BLOCK / warpSize;
  unsigned warpId = uniform(threadIdx.x / warpSize);
  unsigned laneId = threadIdx.x % warpSize;

  // initialize shared memory
  __shared__ float shared_attention_score_max;
  __shared__ float shared_softmax_denom;

  if (threadIdx.x == 0) {
    shared_attention_score_max = 0.f;
    shared_softmax_denom = 0.f;
  }
  __syncthreads();

  // compute the attention scores and accumulate the v vectors into shared memory
  // each warp processes a different token from the KV cache
  // re-align q, k attention out
  q_in = addr(q_in, (((batch_idx * num_q_heads) + q_head_idx) * T + tok_idx) * embed_dim);

  // we don't need to initialize attention_out as prev_softmax_denom is set to 0 initially
  unsigned kv_tok_idx = warpId;
  k_in = addr(k_in, batch_idx * k_batch_stride + k_head_idx * k_head_stride + kv_tok_idx * k_tok_stride);

  temp_scores_f32 = addr(temp_scores_f32, (((batch_idx * num_q_heads) + q_head_idx) * T + tok_idx) * S);
  scores_out = addr(scores_out, (((batch_idx * num_q_heads) + q_head_idx) * T + tok_idx) * S);

  unsigned kv_tok_stride = warps_per_block * embed_dim;

  // reduce per thread and warp the max attention score
  float max_attention_score = -INFINITY;

  // I) computing attention scores

  // TODO: 4 tokens per warp at once to reduce sync amount
  for (; kv_tok_idx < S; kv_tok_idx += warps_per_block) {
    // compute the attention score between this q vector and this k vector
    float attention_score0 = 0.f;
    float attention_score1 = 0.f;
    // vectorized by 2
    unsigned d = 2 * laneId;
    for (; d + 1 < embed_dim; d += 2 * warpSize) {
      half2 qs = *(const half2*)addr(q_in, d);
      half2 ks = *(const half2*)addr(k_in, d);
      attention_score0 += __half2float(qs.x) * __half2float(ks.x);
      attention_score1 += __half2float(qs.y) * __half2float(ks.y);
    }
    if (d < embed_dim) {
      // remainder
      attention_score0 += __half2float(q_in[d]) * __half2float(k_in[d]);
    }

    float attention_score = warpReduce(attention_score0 + attention_score1);
  
    // write out the score
    if (laneId == 0) {
      attention_score *= attention_inv_scale;
      //scores_out[kv_tok_idx] = __float2half(attention_score);
      *addr(temp_scores_f32, kv_tok_idx) = attention_score;
      // update the max attention score
      max_attention_score = std::max(attention_score, max_attention_score);
    }

    // move the pointers
    k_in += kv_tok_stride;
  }

  // II) computing the max attention score across warps
  if (laneId == 0) {
    atomicMax(&shared_attention_score_max, max_attention_score);
  }
  __syncthreads();
  // read back on all threads
  max_attention_score = uniformf(shared_attention_score_max);

  // III) computing the logits
  // each thread can compute its own logit
  float softmax_denom = 0.f;
  for (unsigned t = threadIdx.x; t < S; t += THREADS_PER_BLOCK) {
    float attention_score = *addr(temp_scores_f32, t);
    float logit = __expf(attention_score - max_attention_score);
    softmax_denom += logit;
    *addr(temp_scores_f32, t) = logit;
  }

  // IV) computing the denominator of the softmax
  softmax_denom = warpReduce(softmax_denom);
  if (laneId == 0) {
    atomicAdd(&shared_softmax_denom, softmax_denom);
  }
  __syncthreads();
  // read back on all threads
  softmax_denom = uniformf(shared_softmax_denom);
  float softmax_inv_denom = 1.0f / softmax_denom;

  // V) normalizing with the sum
  for (unsigned t = threadIdx.x; t < S; t += THREADS_PER_BLOCK) {
    float logit = *addr(temp_scores_f32, t) * softmax_inv_denom;
    *addr(scores_out, t) = __float2half(logit);
  }
}

void causally_compute_transformer_softmax_scores_fp16(
  hipStream_t stream,
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, S, embed_dim]
  float* __restrict__ temp_scores_f32, // [B, num_q_heads, T, S]
  half* __restrict__ scores_out, // [B, num_q_heads, T, S]
  // tensor dimension sizes
  unsigned B,
  unsigned T, // num tokens to decode
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_k_heads,
  float attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
  // k strides
  unsigned k_batch_stride,
  unsigned k_head_stride,
  unsigned k_tok_stride
) {
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);

  // expected block dimensions: [x=T, y=num_q_heads, z=B]
  const dim3 num_blocks = dim3(T, num_q_heads, B);
  
  causally_compute_transformer_softmax_scores_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const half*)q_in,
    (const half*)k_in,
    (float*)temp_scores_f32,
    (half*)scores_out,
    // tensor dimension sizes
    T,
    S,
    num_q_heads,
    embed_dim,
    q_to_k_heads,
    attention_inv_scale,
    k_batch_stride,
    k_head_stride,
    k_tok_stride
  );
}

// expected block dimensions: [x=T, y=num_q_heads, z=B]
void __global__ causally_compute_transformer_softmax_scores_masked_fp16_kernel(
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, S, embed_dim]
  const half* __restrict__ m_in, // shape [B, 1, T, S]
  float* __restrict__ temp_scores_f32, // [B, num_q_heads, T, S]
  half* __restrict__ scores_out, // [B, num_q_heads, T, S]
  // tensor dimension sizes
  unsigned T, // num tokens to decode
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_k_heads,
  float attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
  // k strides
  unsigned k_batch_stride,
  unsigned k_head_stride,
  unsigned k_tok_stride
) {
  // one block does one head of a new token
  unsigned q_head_idx = blockIdx.y;
  unsigned tok_idx = blockIdx.x;
  unsigned batch_idx = blockIdx.z;

  // TODO: avoid this division
  unsigned k_head_idx = q_head_idx / q_to_k_heads;

  unsigned warps_per_block = THREADS_PER_BLOCK / warpSize;
  unsigned warpId = uniform(threadIdx.x / warpSize);
  unsigned laneId = threadIdx.x % warpSize;

  // initialize shared memory
  __shared__ float shared_attention_score_max;
  __shared__ float shared_softmax_denom;

  if (threadIdx.x == 0) {
    shared_attention_score_max = 0.f;
    shared_softmax_denom = 0.f;
  }
  __syncthreads();

  // compute the attention scores and accumulate the v vectors into shared memory
  // each warp processes a different token from the KV cache

  // re-align q, k attention out
  q_in = addr(q_in, (((batch_idx * num_q_heads) + q_head_idx) * T + tok_idx) * embed_dim);

  // we don't need to initialize attention_out as prev_softmax_denom is set to 0 initially

  unsigned kv_tok_idx = warpId;
  k_in = addr(k_in, batch_idx * k_batch_stride + k_head_idx * k_head_stride + kv_tok_idx * k_tok_stride);

  m_in = addr(m_in, ((batch_idx * T) + tok_idx) * S + 0);

  temp_scores_f32 = addr(temp_scores_f32, (((batch_idx * num_q_heads) + q_head_idx) * T + tok_idx) * S);
  scores_out = addr(scores_out, (((batch_idx * num_q_heads) + q_head_idx) * T + tok_idx) * S);

  unsigned kv_tok_stride = warps_per_block * embed_dim;

  // reduce per thread and warp the max attention score
  float max_attention_score = -INFINITY;

  // I) computing attention scores

  // TODO: 4 tokens per warp at once to reduce sync amount
  for (; kv_tok_idx < S; kv_tok_idx += warps_per_block) {
    // compute the attention score between this q vector and this k vector

    // we start the read now, but compute anyway
    // as that's faster
    float mask_val = __half2float(*addr(m_in, kv_tok_idx));

    float attention_score0 = 0.f;
    float attention_score1 = 0.f;
    // vectorized by 2
    unsigned d = 2 * laneId;
    for (; d + 1 < embed_dim; d += 2 * warpSize) {
      half2 qs = *(const half2*)addr(q_in, d);
      half2 ks = *(const half2*)addr(k_in, d);
      attention_score0 += __half2float(qs.x) * __half2float(ks.x);
      attention_score1 += __half2float(qs.y) * __half2float(ks.y);
    }
    if (d < embed_dim) {
      // remainder
      attention_score0 += __half2float(q_in[d]) * __half2float(k_in[d]);
    }
    float attention_score = warpReduce(attention_score0 + attention_score1);

    // do the masking if necessary
    attention_score = mask_val == 0.0f ? attention_score : -INFINITY;

    // write out the score
    if (laneId == 0) {
      attention_score *= attention_inv_scale;
      //scores_out[kv_tok_idx] = __float2half(attention_score);
      *addr(temp_scores_f32, kv_tok_idx) = attention_score;
      // update the max attention score
      max_attention_score = std::max(attention_score, max_attention_score);
    }

    // move the pointers
    k_in += kv_tok_stride;
  }

  // II) computing the max attention score across warps
  if (laneId == 0) {
    atomicMax(&shared_attention_score_max, max_attention_score);
  }
  __syncthreads();
  // read back on all threads
  max_attention_score = uniformf(shared_attention_score_max);

  // III) computing the logits
  // each thread can compute its own logit
  float softmax_denom = 0.f;
  for (unsigned t = threadIdx.x; t < S; t += THREADS_PER_BLOCK) {
    float attention_score = *addr(temp_scores_f32, t);
    float logit = __expf(attention_score - max_attention_score);
    softmax_denom += logit;
    *addr(temp_scores_f32, t) = logit;
  }

  // IV) computing the denominator of the softmax
  softmax_denom = warpReduce(softmax_denom);
  if (laneId == 0) {
    atomicAdd(&shared_softmax_denom, softmax_denom);
  }
  __syncthreads();
  // read back on all threads
  softmax_denom = uniformf(shared_softmax_denom);
  float softmax_inv_denom = 1.0f / softmax_denom;

  // V) normalizing with the sum
  for (unsigned t = threadIdx.x; t < S; t += THREADS_PER_BLOCK) {
    float logit = *addr(temp_scores_f32, t) * softmax_inv_denom;
    *addr(scores_out, t) = __float2half(logit);
  }
}

void causally_compute_transformer_softmax_scores_masked_fp16(
  hipStream_t stream,
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, S, embed_dim]
  const half* __restrict__ m_in, // shape [B, 1, T, S]
  float* __restrict__ temp_scores_f32, // [B, num_q_heads, T, S]
  half* __restrict__ scores_out, // [B, num_q_heads, T, S]
  // tensor dimension sizes
  unsigned B,
  unsigned T, // num tokens to decode
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_k_heads,
  float attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
  // k strides
  unsigned k_batch_stride,
  unsigned k_head_stride,
  unsigned k_tok_stride
) {
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);

  // expected block dimensions: [x=T, y=num_q_heads, z=B]
  const dim3 num_blocks = dim3(T, num_q_heads, B);
  
  causally_compute_transformer_softmax_scores_masked_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const half*)q_in,
    (const half*)k_in,
    (const half*)m_in,
    (float*)temp_scores_f32,
    (half*)scores_out,
    // tensor dimension sizes
    T,
    S,
    num_q_heads,
    embed_dim,
    q_to_k_heads,
    attention_inv_scale,
    k_batch_stride,
    k_head_stride,
    k_tok_stride
  );
}

#define COLS_PER_BLOCK 8

// expected block dimensions: [x=DIV_ROUND_UP(embed_dim, COLS_PER_BLOCK), y=num_q_heads, z=B]
void __global__ causally_apply_transformer_softmax_scores_fp16_kernel(
  const half* __restrict__ attention_weights_in, // shape [B, num_q_heads, T, S]
  const half* __restrict__ v_in, // shape [B, num_v_heads, S, embed_dim]
  half* __restrict__ hidden_out, // [B, num_q_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_v_heads,
  // v strides
  unsigned v_batch_stride,
  unsigned v_head_stride,
  unsigned v_tok_stride
) {
  // one block computes for one q head, one entry of the batch a few columns
  // of the result
  unsigned q_head_idx = blockIdx.y;
  unsigned batch_idx = blockIdx.z;

  // TODO: avoid this division
  unsigned v_head_idx = q_head_idx / q_to_v_heads;

  unsigned ROWS_PER_BLOCK = THREADS_PER_BLOCK / COLS_PER_BLOCK;
  unsigned rowIdx = threadIdx.x / COLS_PER_BLOCK;
  unsigned blockColStartIdx = blockIdx.x * COLS_PER_BLOCK;
  unsigned colIdx = threadIdx.x % COLS_PER_BLOCK;

  unsigned v_in_stride = embed_dim * ROWS_PER_BLOCK;
  v_in = addr(v_in, batch_idx * v_batch_stride + v_head_idx * v_head_stride + rowIdx * v_tok_stride + blockColStartIdx);

  // TODO: support T != 1
  attention_weights_in = &attention_weights_in[((batch_idx * num_q_heads) + q_head_idx) * S + 0];
  hidden_out = &hidden_out[((batch_idx * num_q_heads) + q_head_idx) * embed_dim + blockColStartIdx];

  // initialize shared memory
  __shared__ float rs[COLS_PER_BLOCK];
  if (threadIdx.x < COLS_PER_BLOCK) {
    rs[threadIdx.x] = 0.f;
  }
  __syncthreads();

  if (blockColStartIdx + colIdx < embed_dim) {
    // compute column value
    float res = 0.f;
    for (unsigned r = rowIdx; r < S; r+= ROWS_PER_BLOCK) {
      float v = __half2float(v_in[colIdx]);
      float a = __half2float(attention_weights_in[r]);

      res += a * v;

      v_in += v_in_stride;
    }
  
    // reduce in the block
    atomicAdd(&rs[colIdx], res);
    __syncthreads();

    // write out
    if (threadIdx.x < COLS_PER_BLOCK) {
      hidden_out[threadIdx.x] = __float2half(rs[threadIdx.x]);
    }
  }
}

void causally_apply_transformer_softmax_scores_fp16(
  hipStream_t stream,
  const half* __restrict__ attention_weights_in, // shape [B, num_q_heads, T, S]
  const half* __restrict__ v_in, // shape [B, num_v_heads, S, embed_dim]
  half* __restrict__ hidden_out, // [B, num_q_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned B,
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_v_heads,
  // v strides
  unsigned v_batch_stride,
  unsigned v_head_stride,
  unsigned v_tok_stride
) {
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);

  // expected block dimensions: [x=embed_dim/COLS_PER_BLOCK, y=num_q_heads, z=B]
  const dim3 num_blocks = dim3(DIV_ROUND_UP(embed_dim, COLS_PER_BLOCK), num_q_heads, B);

  causally_apply_transformer_softmax_scores_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const half*)attention_weights_in,
    (const half*)v_in,
    (half*)hidden_out,
    // tensor dimension sizes
    S,
    num_q_heads,
    embed_dim,
    q_to_v_heads,
    v_batch_stride,
    v_head_stride,
    v_tok_stride
  );
}