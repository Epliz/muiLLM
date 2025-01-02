#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>


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
void __global__ causally_compute_transformer_softmax_scores(
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, NEW_T, embed_dim]
  float* __restrict__ temp_scores_f32, // [B, num_q_heads, T, NEW_T]
  half* __restrict__ scores_out, // [B, num_q_heads, T, NEW_T]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num tokens to decode
  unsigned NEW_T, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned embed_dim,
  unsigned q_to_k_heads,
  float attention_inv_scale // factor to scale the attention scores, typically 1/sqrt(embed_dim)
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
    k_in = addr(k_in, (((batch_idx * num_k_heads) + k_head_idx) * NEW_T + kv_tok_idx) * embed_dim);

    temp_scores_f32 = addr(temp_scores_f32, (((batch_idx * num_q_heads) + q_head_idx) * T + tok_idx) * NEW_T);
    scores_out = addr(scores_out, (((batch_idx * num_q_heads) + q_head_idx) * T + tok_idx) * NEW_T);

    unsigned kv_tok_stride = warps_per_block * embed_dim;

    // reduce per thread and warp the max attention score
    float max_attention_score = -INFINITY;

    // I) computing attention scores

    // TODO: 4 tokens per warp at once to reduce sync amount
    for (; kv_tok_idx < NEW_T; kv_tok_idx += warps_per_block) {
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
    for (unsigned t = threadIdx.x; t < NEW_T; t += THREADS_PER_BLOCK) {
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
    for (unsigned t = threadIdx.x; t < NEW_T; t += THREADS_PER_BLOCK) {
        float logit = *addr(temp_scores_f32, t) * softmax_inv_denom;
        *addr(scores_out, t) = __float2half(logit);
    }
}

#define COLS_PER_BLOCK 8

// expected block dimensions: [x=DIV_ROUND_UP(embed_dim, COLS_PER_BLOCK), y=num_q_heads, z=B]
void __global__ causally_apply_transformer_softmax_scores(
  const half* __restrict__ attention_weights_in, // shape [B, num_q_heads, T, NEW_T]
  const half* __restrict__ v_in, // shape [B, num_v_heads, NEW_T, embed_dim]
  half* __restrict__ hidden_out, // [B, num_q_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num tokens to decode
  unsigned NEW_T, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned num_v_heads, // number of heads for k
  unsigned embed_dim,
  unsigned q_to_v_heads
) {
  // one block computes for one q head, one entry of the batch a few columns
  // of the result
  unsigned q_head_idx = blockIdx.y;
  unsigned batch_idx = blockIdx.z;

  unsigned v_head_idx = q_head_idx / q_to_v_heads;

  unsigned ROWS_PER_BLOCK = THREADS_PER_BLOCK / COLS_PER_BLOCK;
  unsigned rowIdx = threadIdx.x / COLS_PER_BLOCK;
  unsigned blockColStartIdx = blockIdx.x * COLS_PER_BLOCK;
  unsigned colIdx = threadIdx.x % COLS_PER_BLOCK;

  unsigned v_in_stride = embed_dim * ROWS_PER_BLOCK;
  v_in = &v_in[(((batch_idx * num_v_heads) + v_head_idx) * NEW_T + rowIdx) * embed_dim + blockColStartIdx];

  // TODO: support T != 1
  attention_weights_in = &attention_weights_in[((batch_idx * num_q_heads) + q_head_idx) * NEW_T + 0];
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
    for (unsigned r = rowIdx; r < NEW_T; r+= ROWS_PER_BLOCK) {
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

// expected block dimensions: [x=T, y=num_q_heads, z=B]
void __global__ causally_decode_transformer(
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, NEW_T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, NEW_T, embed_dim]
  half* __restrict__ attention_out, // [B, num_q_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num tokens to decode
  unsigned NEW_T, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim,
  unsigned q_to_k_heads,
  unsigned q_to_v_heads,
  float attention_inv_scale // factor to scale the attention scores, typically 1/sqrt(embed_dim)
) {
    // one block does one head of a new token
    unsigned q_head_idx = blockIdx.y;
    unsigned tok_idx = blockIdx.x;
    unsigned batch_idx = blockIdx.z;

    unsigned k_head_idx = q_head_idx / q_to_k_heads;
    unsigned v_head_idx = q_head_idx / q_to_v_heads;

    unsigned warps_per_block = THREADS_PER_BLOCK / warpSize;
    unsigned warpId = uniform(threadIdx.x / warpSize);
    unsigned laneId = threadIdx.x % warpSize;

    // initialize shared memory
    __shared__ float attention_score_max;
    __shared__ float softmax_denom;

    if (threadIdx.x == 0) {
      attention_score_max = 0.f;
      softmax_denom = 0.f;
    }
    __syncthreads();

    // compute the attention scores and accumulate the v vectors into shared memory
    // each warp processes a different token from the KV cache

    // re-align q, k, v, attention out
    q_in = &q_in[(((batch_idx * num_q_heads) + q_head_idx) * T + tok_idx) * embed_dim];
    attention_out = &attention_out[(((batch_idx * num_q_heads) + q_head_idx) * T + tok_idx) * embed_dim];

    // we don't need to initialize attention_out as prev_softmax_denom is set to 0 initially

    unsigned kv_tok_idx = warpId;
    k_in = &k_in[(((batch_idx * num_k_heads) + k_head_idx) * NEW_T + kv_tok_idx) * embed_dim];
    v_in = &v_in[(((batch_idx * num_v_heads) + v_head_idx) * NEW_T + kv_tok_idx) * embed_dim];

    unsigned kv_tok_stride = warps_per_block * embed_dim;

    // keep track of the max attention score to rescale V vectors
    float prev_max_attention_score = 0.0f;
    float prev_softmax_denom = 0.0f;
    // TODO: 4 tokens per warp at once to reduce sync amount
    for (; kv_tok_idx < NEW_T; kv_tok_idx += warps_per_block) {
      // compute the attention score between this q vector and this k vector
      float attention_score = 0.f;
      for (unsigned d = threadIdx.x; d < embed_dim; d += warpSize) {
        attention_score += __half2float(q_in[d]) * __half2float(k_in[d]);
      }

      attention_score *= attention_inv_scale;
      attention_score = warpReduce(attention_score);
      attention_score = uniform(attention_score);

      // determine the new max attention score
      float max_attention_score = 0.0f;
      if (laneId == 0) {
        atomicMax(&attention_score_max, attention_score);
      }
      __syncthreads();
      // all warps have put their attention scores
      // TODO: compare vs just reading the shared variable with all threads
      if (laneId == 0) {
        max_attention_score = attention_score_max;
      }
      max_attention_score = uniform(max_attention_score);

      // rescale previous vector and accumulate new vector
      float rescaling_factor = __expf(prev_max_attention_score - max_attention_score);
      float scaling_factor = __expf(attention_score - max_attention_score);

      // compute the new softmax denominator
      float denom_softmax = 0.0f;
      if (laneId == 0) {
        if (warpId == 0) {
          // do the rescaling of the previous denominator
          softmax_denom = rescaling_factor * softmax_denom;
        }
      }
        __syncthreads();
      if (laneId == 0) {
        // add the new scores in the sum
        atomicAdd(&softmax_denom, scaling_factor);
      }
      __syncthreads();
      if (laneId == 0) {
        denom_softmax = softmax_denom;
      }
      denom_softmax = uniform(denom_softmax);

      // we accumulate one warp at a time as they handle different tokens
      for (unsigned w = 0; w < warps_per_block; w++) {
        __syncthreads();
        if (w == warpId) {
          if (w == 0) {
            // rescale the previous output
            float remapping_factor = rescaling_factor * (prev_softmax_denom / denom_softmax);
            for (unsigned d = threadIdx.x; d < embed_dim; d += warpSize) {
              float rescaled_output = remapping_factor * __half2float(attention_out[d]);
              attention_out[d] = __float2half(rescaled_output);
            }
          }
          // add the new scaled vector
          float mapping_factor = scaling_factor / denom_softmax;
          for (unsigned d = threadIdx.x; d < embed_dim; d += warpSize) {
            float v = __half2float(v_in[d]);
            float a = __half2float(attention_out[d]);
            attention_out[d] = __float2half(a + mapping_factor * v);
          }
        }
      }

      // remember the previous maximum, softmax denominator
      prev_max_attention_score = max_attention_score;
      prev_softmax_denom = denom_softmax;

      // move the kv pointers
      k_in += kv_tok_stride;
      v_in += kv_tok_stride;
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor muillm_causal_transformer_compute_softmax_scores_no_mask(
    torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
    torch::Tensor& k // [B, num_k_heads, NEW_T, embed_dim]
) {
  // q, k are expected to be contiguous
  CHECK_INPUT(q);
  CHECK_INPUT(k);

  auto device = q.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto output_options_f16 = at::TensorOptions()
                            .dtype(torch::kFloat16)
                            .layout(at::kStrided)
                            .device(device) // same device as inputs
                            .requires_grad(false);
  

  auto temp_options_f32 = at::TensorOptions()
                            .dtype(torch::kFloat32)
                            .layout(at::kStrided)
                            .device(device) // same device as inputs
                            .requires_grad(false);

  auto q_sizes = q.sizes().vec();
  auto k_sizes = k.sizes().vec();

  unsigned B = q_sizes[0];
  unsigned T = q_sizes[2];
  unsigned NEW_T = k_sizes[2];
  unsigned num_q_heads = q_sizes[1];
  unsigned num_k_heads = k_sizes[1];
  unsigned embed_dim = q_sizes[3];


  auto scores_sizes = q.sizes().vec();
  scores_sizes[3] = NEW_T;

  // to compute what k/v head to target for a given q head
  unsigned q_to_k_heads = num_q_heads / num_k_heads;

  float attention_inv_scale = 1.0f / sqrtf(embed_dim);

  auto temp_scores_f32 = torch::empty(scores_sizes, temp_options_f32);
  auto scores_out = torch::empty(scores_sizes, output_options_f16);

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);

  // expected block dimensions: [x=T, y=num_q_heads, z=B]
  const dim3 num_blocks = dim3(T, num_q_heads, B);

  causally_compute_transformer_softmax_scores<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const half*)q.data_ptr(),
    (const half*)k.data_ptr(),
    (float*)temp_scores_f32.data_ptr(),
    (half*)scores_out.data_ptr(),
    // tensor dimension sizes
    B,
    T,
    NEW_T,
    num_q_heads,
    num_k_heads,
    embed_dim,
    q_to_k_heads,
    attention_inv_scale
  );

  return scores_out;
}


at::Tensor muillm_causal_transformer_apply_softmax_scores(
    torch::Tensor& attention_weights, // [B, num_q_heads, T, NEW_T]
    torch::Tensor& v // [B, num_v_heads, NEW_T, embed_dim]
) {
  // q, k are expected to be contiguous
  CHECK_INPUT(attention_weights);
  CHECK_INPUT(v);

  auto device = v.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto output_options_f16 = at::TensorOptions()
                            .dtype(torch::kFloat16)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto attention_weights_sizes = attention_weights.sizes().vec();
  auto v_sizes = v.sizes().vec();

  unsigned B = attention_weights_sizes[0];
  unsigned T = attention_weights_sizes[2];
  unsigned NEW_T = attention_weights_sizes[3];
  unsigned num_q_heads = attention_weights_sizes[1];
  unsigned num_v_heads = v_sizes[1];
  unsigned embed_dim = v_sizes[3];


  auto hidden_sizes = attention_weights.sizes().vec();
  hidden_sizes[3] = embed_dim;

  // to compute what k/v head to target for a given q head
  unsigned q_to_v_heads = num_q_heads / num_v_heads;

  auto hidden_out = torch::empty(hidden_sizes, output_options_f16);

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);

  // expected block dimensions: [x=embed_dim/COLS_PER_BLOCK, y=num_q_heads, z=B]
  const dim3 num_blocks = dim3(DIV_ROUND_UP(embed_dim, COLS_PER_BLOCK), num_q_heads, B);

  causally_apply_transformer_softmax_scores<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const half*)attention_weights.data_ptr(),
    (const half*)v.data_ptr(),
    (half*)hidden_out.data_ptr(),
    // tensor dimension sizes
    B,
    T,
    NEW_T,
    num_q_heads,
    num_v_heads,
    embed_dim,
    q_to_v_heads
  );

  return hidden_out;
}

// this one does not perform so well, but maybe is fixable?
at::Tensor muillm_causal_transformer_decoding_no_mask_fully_fused(
    torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
    torch::Tensor& k, // [B, num_k_heads, NEW_T, embed_dim]
    torch::Tensor& v  // [B, num_v_heads, NEW_T, embed_dim]
) {
  // q, k, v are expected to be contiguous
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  auto device = q.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = torch::kFloat16;
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);
  
  auto q_sizes = q.sizes().vec();
  auto k_sizes = k.sizes().vec();
  auto v_sizes = v.sizes().vec();

  unsigned B = q_sizes[0];
  unsigned T = q_sizes[2];
  unsigned NEW_T = k_sizes[2];
  unsigned num_q_heads = q_sizes[1];
  unsigned num_k_heads = k_sizes[1];
  unsigned num_v_heads = v_sizes[1];
  unsigned embed_dim = q_sizes[3];

  // to compute what k/v head to target for a given q head
  unsigned q_to_k_heads = num_q_heads / num_k_heads;
  unsigned q_to_v_heads = num_q_heads / num_v_heads;

  float attention_inv_scale = 1.0f / sqrtf(embed_dim);

  auto attention_out = torch::empty(q_sizes, output_options);

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);

  // expected block dimensions: [x=T, y=num_q_heads, z=B]
  const dim3 num_blocks = dim3(T, num_q_heads, B);

  causally_decode_transformer<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const half*)q.data_ptr(),
    (const half*)k.data_ptr(),
    (const half*)v.data_ptr(),
    (half*)attention_out.data_ptr(),
    // tensor dimension sizes
    B,
    T,
    NEW_T,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    embed_dim,
    q_to_k_heads,
    q_to_v_heads,
    attention_inv_scale
  );

  return attention_out;
}

at::Tensor muillm_causal_transformer_decoding_no_mask(
    torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
    torch::Tensor& k, // [B, num_k_heads, NEW_T, embed_dim]
    torch::Tensor& v  // [B, num_v_heads, NEW_T, embed_dim]
) {
  auto attention_scores = muillm_causal_transformer_compute_softmax_scores_no_mask(
    q,
    k
  );

  return muillm_causal_transformer_apply_softmax_scores(
    attention_scores,
    v
  );
}