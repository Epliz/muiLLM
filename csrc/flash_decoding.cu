#include "flash_decoding.cuh"

#include <ATen/cuda/CUDAContext.h>

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

struct __align__(8) half4 {
  half x;
  half y;
  half z;
  half w;
};

struct __align__(8) half8 {
  half x;
  half y;
  half z;
  half w;
  half a;
  half b;
  half c;
  half d;
};

struct __align__(8) float8 {
  float x;
  float y;
  float z;
  float w;
  float a;
  float b;
  float c;
  float d;
};


static inline void __device__ dot2(float& acc, const float2& a, const float2& b) {
  acc += a.x * b.x;
  acc += a.y * b.y;
}

static inline void __device__ dot4(float& acc, const float4& a, const float4& b) {
  acc += ((a.x * b.x) + (a.w * b.w)) + ((a.y * b.y) + (a.z * b.z));
}

static inline void __device__ dot8(float& acc, const float8& a, const float8& b) {
  acc += ((a.x * b.x) + (a.w * b.w)) + ((a.y * b.y) + (a.z * b.z));
  acc += ((a.a * b.a) + (a.c * b.c)) + ((a.b * b.b) + (a.d * b.d));
}

static inline float4 __device__ __half42float4(const half4& v) {
  float4 f;
  f.x = __half2float(v.x);
  f.y = __half2float(v.y);
  f.z = __half2float(v.z);
  f.w = __half2float(v.w);

  return f;
}

static inline float8 __device__ __half82float8(const half8& v) {
  float8 f;
  f.x = __half2float(v.x);
  f.y = __half2float(v.y);
  f.z = __half2float(v.z);
  f.w = __half2float(v.w);
  f.a = __half2float(v.a);
  f.b = __half2float(v.b);
  f.c = __half2float(v.c);
  f.d = __half2float(v.d);

  return f;
}

__device__ half2 load_nontemporal_half2(const half* p) {
  float _v = __builtin_nontemporal_load((const float*)p);
  return *((half2*)&_v);
}

__device__ half4 load_nontemporal_half4(const half* p) {
  float _v0 = __builtin_nontemporal_load(((const float*)p));
  float _v1 = __builtin_nontemporal_load(((const float*)p) + 1);

  half2 _hv0 = *((half2*)&_v0);
  half2 _hv1 = *((half2*)&_v1);

  half4 v;
  v.x = _hv0.x;
  v.y = _hv0.y;
  v.z = _hv1.x;
  v.w = _hv1.y;

  return v;
}

__device__ half8 load_nontemporal_half8(const half* p) {
  float _v0 = __builtin_nontemporal_load(((const float*)p));
  float _v1 = __builtin_nontemporal_load(((const float*)p) + 1);
  float _v2 = __builtin_nontemporal_load(((const float*)p) + 2);
  float _v3 = __builtin_nontemporal_load(((const float*)p) + 3);

  half2 _hv0 = *((half2*)&_v0);
  half2 _hv1 = *((half2*)&_v1);
  half2 _hv2 = *((half2*)&_v2);
  half2 _hv3 = *((half2*)&_v3);

  half8 v;
  v.x = _hv0.x;
  v.y = _hv0.y;
  v.z = _hv1.x;
  v.w = _hv1.y;
  v.a = _hv2.x;
  v.b = _hv2.y;
  v.c = _hv3.x;
  v.d = _hv3.y;

  return v;
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

#define MIN_TOKENS_PER_GROUP 4

#define GEMV_ROWS_PER_BLOCK 4

// expected number of blocks: [x=G, y=num_q_heads, z=B]
void __global__ flash_decoding_partially_aggregate_kernel(
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, S, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_k_heads, S, embed_dim]
  const half* __restrict__ m_in, // shape [B, 1, T, S]
  float* __restrict__ partial_vectors, // [B, num_q_heads, G, T, embed_dim]
  half* __restrict__ partial_softmax_denoms, // [B, num_q_heads, G, T]
  half* __restrict__ partial_maxes, // [B, num_q_heads, G, T]
  // tensor dimension sizes
  unsigned G, // number of groups
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_kv_heads,
  float attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
  // kv strides
  unsigned kv_batch_stride,
  unsigned kv_head_stride,
  unsigned kv_tok_stride
) {
  // one block does one head of a new token
  unsigned group_idx = blockIdx.x;
  unsigned q_head_idx = blockIdx.y;
  unsigned batch_idx = blockIdx.z;

  // TODO: avoid this division
  unsigned kv_head_idx = q_head_idx / q_to_kv_heads;

  unsigned warpCounts = THREADS_PER_BLOCK / warpSize;
  unsigned warpId = uniform(threadIdx.x / warpSize);
  unsigned laneId = threadIdx.x % warpSize;

  unsigned T = 1; // TODO: support T > 1
  unsigned q_tok_idx = 0;

  //
  // first, do the matrix multiplication between q and k
  //

  // one block computes for one q head, one entry of the batch, one group

  // TODO: avoid bank conflicts by having per warp shared memory
  __shared__ float shared_accs[GEMV_ROWS_PER_BLOCK];

  // initialize the shared memory
  if (threadIdx.x < GEMV_ROWS_PER_BLOCK) {
    shared_accs[threadIdx.x] = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  const half* X = addr(q_in, (((batch_idx * num_q_heads) + q_head_idx) * T + q_tok_idx) * embed_dim);
  k_in = addr(k_in, batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride);
  {
    int current_row = group_idx * GEMV_ROWS_PER_BLOCK + 0;
    if (current_row + 3 < S) {

      // compute the t-th element of Y. by doing the dot product with the
      // t-th row of W
      const half* W0 = &k_in[(current_row + 0) * embed_dim];
      const half* W1 = &k_in[(current_row + 1) * embed_dim];
      const half* W2 = &k_in[(current_row + 2) * embed_dim];
      const half* W3 = &k_in[(current_row + 3) * embed_dim];

      float acc0 = 0.f;
      float acc1 = 0.f;
      float acc2 = 0.f;
      float acc3 = 0.f;

      // do the dot product
      {
        unsigned k;
        //*
        for (k = threadIdx.x * 4; k + 3 < embed_dim; k += (THREADS_PER_BLOCK * 4)) {
          // vectorized
          float4 x = __half42float4(*(const half4*)(addr(X, k)));
          float4 w0 = __half42float4(load_nontemporal_half4(addr(W0, k)));
          float4 w1 = __half42float4(load_nontemporal_half4(addr(W1, k)));
          float4 w2 = __half42float4(load_nontemporal_half4(addr(W2, k)));
          float4 w3 = __half42float4(load_nontemporal_half4(addr(W3, k)));

          dot4(acc0, w0, x);
          dot4(acc1, w1, x);
          dot4(acc2, w2, x);
          dot4(acc3, w3, x);
        }
        if (k + 1 < embed_dim) {
          // remainder
          float2 x = __half22float2(*(const half2*)(addr(X,k)));
          float2 w0 = __half22float2(load_nontemporal_half2(addr(W0,k)));
          float2 w1 = __half22float2(load_nontemporal_half2(addr(W1,k)));
          float2 w2 = __half22float2(load_nontemporal_half2(addr(W2,k)));
          float2 w3 = __half22float2(load_nontemporal_half2(addr(W3,k)));

          dot2(acc0, w0, x);
          dot2(acc1, w1, x);
          dot2(acc2, w2, x);
          dot2(acc3, w3, x);

          k+= 2;
        }
        if (k < embed_dim) {
          // remainder
          float x = __half2float(*addr(X,k));
          float w0 = __half2float(*addr(W0,k));
          float w1 = __half2float(*addr(W1,k));
          float w2 = __half2float(*addr(W2,k));
          float w3 = __half2float(*addr(W3,k));
          acc0 += w0 * x;
          acc1 += w1 * x;
          acc2 += w2 * x;
          acc3 += w3 * x;
        }
      }

      // warp reduce
      acc0 = warpReduce(acc0);
      acc1 = warpReduce(acc1);
      acc2 = warpReduce(acc2);
      acc3 = warpReduce(acc3);

      // reduce accross warps
      if (laneId == 0) {
        atomicAdd(&shared_accs[0], acc0);
        atomicAdd(&shared_accs[1], acc1);
        atomicAdd(&shared_accs[2], acc2);
        atomicAdd(&shared_accs[3], acc3);
      }
    } else {
      for (int i = 0; i < GEMV_ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = group_idx * GEMV_ROWS_PER_BLOCK + i;

        if (current_row >= S) {
          // we don't have a row to compute
          // we should set the attention score to -INFINITY to
          // avoid messing up the softmax
          if (laneId == 0) {
            shared_accs[i] = -INFINITY;
          }
          // we need to continue to set all remaining values to -INFINITY
          continue;
        }

        const half* W_ = &k_in[current_row * embed_dim];
      
        // do the dot product
        float acc = 0.f;
        {
          int k = threadIdx.x  * 2;
          for (; k + 1 < embed_dim; k += THREADS_PER_BLOCK * 2) {
            float2 w = __half22float2(*(const half2*)&W_[k]);
            float2 x = __half22float2(*(const half2*)&X[k]);
            dot2(acc, w, x);
          }
          if (k < embed_dim) {
            float w = __half2float(W_[k]);
            float x = __half2float(X[k]);
            acc += w * x;
          }
        }


        // warp reduce
        acc = warpReduce(acc);

        // reduce accross warps
        if (laneId == 0) {
          atomicAdd(&shared_accs[i], acc);
        }
      }
    }
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  // scale the attention weights
  { 
    int k_tok_idx = group_idx * GEMV_ROWS_PER_BLOCK + threadIdx.x;
    if (k_tok_idx < S) {
      float attention_score = attention_inv_scale * shared_accs[threadIdx.x];
  
      if (m_in != nullptr) {
        // appply the mask
        m_in = addr(m_in, ((batch_idx * T) + q_tok_idx) * S + 0);
  
        float mask_val = __half2float(*addr(m_in, k_tok_idx));
    
        attention_score = mask_val == 0.0f ? attention_score : -INFINITY;
      }
  
      shared_accs[threadIdx.x] = attention_score;
    }
  }

 

  //
  // then compute and write out the max for the group and the softmax denominator
  //

  // compute the group max
  partial_maxes = &partial_maxes[((batch_idx * num_q_heads) + q_head_idx) * G + 0];

  __shared__ float group_max_shared;

  if (threadIdx.x == 0) {
    group_max_shared = std::max(std::max(shared_accs[0], shared_accs[1]), std::max(shared_accs[2], shared_accs[3]));
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  float group_max = uniformf(group_max_shared);
 
  if (threadIdx.x == 0) {
    // write out the max
    partial_maxes[group_idx] = __float2half(group_max);
  }

  // compute the softmax denominator
  partial_softmax_denoms = &partial_softmax_denoms[((batch_idx * num_q_heads) + q_head_idx) * G + 0];

  if (isinf(group_max) && group_max < 0.f) {
    // if the group is fully masked, we can skip the aggregation
    if (threadIdx.x == 0) {
      partial_softmax_denoms[group_idx] = __float2half(0.f);
    }

    // but it is preferable to write 0s to the partial vectors
    // to avoid NaNs in the final aggregation
    {
      unsigned kv_tok_idx = group_idx * GEMV_ROWS_PER_BLOCK + 0;

      partial_vectors = addr(partial_vectors, (((batch_idx * num_q_heads) + q_head_idx) * G + group_idx) * embed_dim + 0);

      unsigned k;
      for (k = threadIdx.x * 2; k + 1 < embed_dim; k += (THREADS_PER_BLOCK * 2)) {
        *((float2*) addr(partial_vectors, k)) = make_float2(0.f, 0.f);
      }
      if (k < embed_dim) {
        *((float*) addr(partial_vectors, k)) = 0.f;
      }
    }

    return;
  }

  // non-fully masked group

  __shared__ float softmax_denom_shared;
  __shared__ float logits_shared[GEMV_ROWS_PER_BLOCK];

  if (threadIdx.x == 0) {
    float logit0 = __expf(shared_accs[0] - group_max);
    float logit1 = __expf(shared_accs[1] - group_max);
    float logit2 = __expf(shared_accs[2] - group_max);
    float logit3 = __expf(shared_accs[3] - group_max);

    logits_shared[0] = logit0;
    logits_shared[1] = logit1;
    logits_shared[2] = logit2;
    logits_shared[3] = logit3;

    softmax_denom_shared = (logit0 + logit1) + (logit2 + logit3);
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  float softmax_denom = uniformf(softmax_denom_shared);

  if (threadIdx.x == 0) {
    // write out the softmax denominator
    partial_softmax_denoms[group_idx] = __float2half(softmax_denom);
  }

  //
  // then aggregate and write out the partial vectors
  //
  {
    unsigned kv_tok_idx = group_idx * GEMV_ROWS_PER_BLOCK + 0;

    const half* v_in0 = addr(v_in, batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride + kv_tok_idx * kv_tok_stride);
    const half* v_in1 = addr(v_in0, 1 * kv_tok_stride);
    const half* v_in2 = addr(v_in0, 2 * kv_tok_stride);
    const half* v_in3 = addr(v_in0, 3 * kv_tok_stride);

    partial_vectors = addr(partial_vectors, (((batch_idx * num_q_heads) + q_head_idx) * G + group_idx) * embed_dim + 0);

    float inv_softmax_denom = 1.f / softmax_denom;

    float scale0 = uniformf(logits_shared[0]) * inv_softmax_denom;
    float scale1 = uniformf(logits_shared[1]) * inv_softmax_denom;
    float scale2 = uniformf(logits_shared[2]) * inv_softmax_denom;
    float scale3 = uniformf(logits_shared[3]) * inv_softmax_denom;

    int current_row = group_idx * GEMV_ROWS_PER_BLOCK + 0;
    if (current_row + 3 < S) {
      // 4 rows to aggregate
      unsigned k;
      for (k = threadIdx.x * 2; k + 1 < embed_dim; k += (THREADS_PER_BLOCK * 2)) {
        float2 v0 = __half22float2(load_nontemporal_half2(addr(v_in0, k)));
        float2 v1 = __half22float2(load_nontemporal_half2(addr(v_in1, k)));
        float2 v2 = __half22float2(load_nontemporal_half2(addr(v_in2, k)));
        float2 v3 = __half22float2(load_nontemporal_half2(addr(v_in3, k)));

        float2 pv = (((scale0 * v0) + (scale1 * v1)) + (scale2 * v2)) + (scale3 * v3);

        *((float2*) addr(partial_vectors, k)) = pv;
      }
      if (k < embed_dim) {
        // remainder
        float v0 = __half2float(*addr(v_in0,k));
        float v1 = __half2float(*addr(v_in1,k));
        float v2 = __half2float(*addr(v_in2,k));
        float v3 = __half2float(*addr(v_in3,k));

        float pv = (((scale0 * v0) + (scale1 * v1)) + (scale2 * v2)) + (scale3 * v3);

        *((float*) addr(partial_vectors, k)) = pv;
      }
    } else if (current_row + 2 < S) {
      // 3 rows to aggregate
      unsigned k;
      for (k = threadIdx.x * 2; k + 1 < embed_dim; k += (THREADS_PER_BLOCK * 2)) {
        float2 v0 = __half22float2(load_nontemporal_half2(addr(v_in0, k)));
        float2 v1 = __half22float2(load_nontemporal_half2(addr(v_in1, k)));
        float2 v2 = __half22float2(load_nontemporal_half2(addr(v_in2, k)));

        float2 pv = ((scale0 * v0) + (scale1 * v1)) + (scale2 * v2);

        *((float2*) addr(partial_vectors, k)) = pv;
      }
      if (k < embed_dim) {
        // remainder
        float v0 = __half2float(*addr(v_in0,k));
        float v1 = __half2float(*addr(v_in1,k));
        float v2 = __half2float(*addr(v_in2,k));

        float pv = ((scale0 * v0) + (scale1 * v1)) + (scale2 * v2);

        *((float*) addr(partial_vectors, k)) = pv;
      }
    } else if (current_row + 1 < S) {
      // 2 rows to aggregate
      unsigned k;
      for (k = threadIdx.x * 2; k + 1 < embed_dim; k += (THREADS_PER_BLOCK * 2)) {
        float2 v0 = __half22float2(load_nontemporal_half2(addr(v_in0, k)));
        float2 v1 = __half22float2(load_nontemporal_half2(addr(v_in1, k)));

        float2 pv = (scale0 * v0) + (scale1 * v1);

        *((float2*) addr(partial_vectors, k)) = pv;
      }
      if (k < embed_dim) {
        // remainder
        float v0 = __half2float(*addr(v_in0,k));
        float v1 = __half2float(*addr(v_in1,k));

        float pv = ((scale0 * v0) + (scale1 * v1));

        *((float*) addr(partial_vectors, k)) = pv;
      }
    } else if (current_row < S) {
      // 1 row to aggregate
      unsigned k;
      for (k = threadIdx.x * 2; k + 1 < embed_dim; k += (THREADS_PER_BLOCK * 2)) {
        float2 v0 = __half22float2(load_nontemporal_half2(addr(v_in0, k)));

        *((float2*) addr(partial_vectors, k)) = v0;
      }
      if (k < embed_dim) {
        // remainder
        float v0 = __half2float(*addr(v_in0,k));

        *((float*) addr(partial_vectors, k)) = v0;
      }
    }
  }
}
    
void flash_decoding_partially_aggregate(
  hipStream_t stream,
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, S, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, S, embed_dim]
  const half* __restrict__ m_in, // shape [B, 1, T, S]
  float* __restrict__ partial_vectors, // [B, num_q_heads, G, T, embed_dim]
  half* __restrict__ partial_softmax_denoms, // [B, num_q_heads, G, T]
  half* __restrict__ partial_maxes, // [B, num_q_heads, G, T]
  // tensor dimension sizes
  unsigned B,
  unsigned G,
  unsigned T, // num tokens to decode
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_kv_heads,
  float attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
  // kv strides
  unsigned kv_batch_stride,
  unsigned kv_head_stride,
  unsigned kv_tok_stride
) {
  // TODO: support T > 1
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);

  // expected number of blocks: [x=G, y=num_q_heads, z=B]
  const dim3 num_blocks = dim3(G, num_q_heads, B);
  
  flash_decoding_partially_aggregate_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const half*)q_in,
    (const half*)k_in,
    (const half*)v_in,
    (const half*)m_in,
    (float*)partial_vectors,
    (half*)partial_softmax_denoms,
    (half*) partial_maxes,
    // tensor dimension sizes
    G,
    S,
    num_q_heads,
    embed_dim,
    q_to_kv_heads,
    attention_inv_scale,
    kv_batch_stride,
    kv_head_stride,
    kv_tok_stride
  );
}


#define COLS_PER_BLOCK 8

// expected block dimensions: [x=embed_dim/COLS_PER_BLOCK, y=num_q_heads, z=B]
void __global__ flash_decoding_final_reduce_kernel(
    const float* __restrict__ partial_vectors, // shape [B, num_q_heads, G, T, embed_dim]
    const half* __restrict__ partial_softmax_denoms, // shape [B, num_q_heads, G, T]
    half* __restrict__ partial_maxes, // [B, num_q_heads, G, T]
    half* __restrict__ hidden_out, // [B, num_q_heads, T, embed_dim]
    // tensor dimension sizes
    unsigned G, // number of groups
    unsigned num_q_heads, // number of heads for q
    unsigned embed_dim,
    // v strides
    unsigned v_batch_stride,
    unsigned v_head_stride,
    unsigned v_tok_stride
) {
  unsigned warps_per_block = THREADS_PER_BLOCK / warpSize;
  unsigned warpId = uniform(threadIdx.x / warpSize);
  unsigned laneId = threadIdx.x % warpSize;

  // one block computes for one q head, one entry of the batch a few columns
  // of the result
  unsigned q_head_idx = blockIdx.y;
  unsigned batch_idx = blockIdx.z;

  unsigned head_idx = q_head_idx;

  unsigned ROWS_PER_BLOCK = THREADS_PER_BLOCK / COLS_PER_BLOCK;
  unsigned rowIdx = threadIdx.x / COLS_PER_BLOCK;
  unsigned blockColStartIdx = blockIdx.x * COLS_PER_BLOCK;
  unsigned colIdx = threadIdx.x % COLS_PER_BLOCK;

  unsigned v_in_stride = embed_dim * ROWS_PER_BLOCK;
  partial_vectors = addr(partial_vectors, batch_idx * v_batch_stride + head_idx * v_head_stride + rowIdx * v_tok_stride + blockColStartIdx);

  // TODO: support T != 1
  partial_softmax_denoms = &partial_softmax_denoms[((batch_idx * num_q_heads) + q_head_idx) * G + 0];
  partial_maxes = &partial_maxes[((batch_idx * num_q_heads) + q_head_idx) * G + 0];
  hidden_out = &hidden_out[((batch_idx * num_q_heads) + q_head_idx) * embed_dim + blockColStartIdx];


  // initialize the shared memory for finding the final max as well as the final softmax denominator
  __shared__ float final_max_shared;
  __shared__ float final_softmax_denom_shared;

  if (threadIdx.x == 0) {
    final_max_shared = -INFINITY;
    final_softmax_denom_shared = 0.f;
  }

  // initialize shared memory for final aggregation as well
  __shared__ float rs[COLS_PER_BLOCK];
  if (threadIdx.x < COLS_PER_BLOCK) {
    rs[threadIdx.x] = 0.f;
  }
  __syncthreads();


  //
  // first find the final max by going through all the group maxes
  //

  float final_max = -INFINITY;

  for (unsigned g = threadIdx.x; g < G; g += THREADS_PER_BLOCK) {
    // note: partial_max might be -INFINITY if the group was fully masked
    float partial_max = __half2float(partial_maxes[g]);

    final_max = std::max(partial_max, final_max);
  }

  final_max = warpReduceMax(final_max);

  if (laneId == 0) {
    atomicMax(&final_max_shared, final_max);
  }

  __syncthreads();

  // load the final max from shared memory by reading back on all threads
  final_max = uniformf(final_max_shared);

  //
  // then compute the final softmax denominator
  //
  float final_softmax_denom = 0.f;

  for (unsigned g = threadIdx.x; g < G; g += THREADS_PER_BLOCK) {
    // even if the block was fully masked, the partial softmax denom will be 0
    // and not cause NaNs here
    float d = __half2float(partial_softmax_denoms[g]);

    float group_max = __half2float(partial_maxes[g]);
    float max_diff = group_max - final_max;
    float group_scale = __expf(max_diff);
  
    final_softmax_denom += group_scale * d;
  }

  final_softmax_denom = warpReduce(final_softmax_denom);

  if (laneId == 0) {
    atomicAdd(&final_softmax_denom_shared, final_softmax_denom);
  }

  __syncthreads();

  // load the final softmax denominator from shared memory by reading back on all threads
  final_softmax_denom = uniformf(final_softmax_denom_shared);

  //
  // aggregate the partial vectors into the final hidden state
  //

  if (blockColStartIdx + colIdx < embed_dim) {
    // compute column value
    float res = 0.f;
    for (unsigned r = rowIdx; r < G; r+= ROWS_PER_BLOCK) {
      // v is the aggregated vector of the group
      // v = sum(exp(a_i - partial_max_g) * v_i / (partial_softmax_denom_g))
      // so to do the final aggregation we need to bring to the final scale first
      float v = partial_vectors[colIdx];

      float d = __half2float(partial_softmax_denoms[r]);
      float group_max = __half2float(partial_maxes[r]);

      float max_diff = group_max - final_max;
      float group_scale = __expf(max_diff);

      float group_rescaling = group_scale * (d / final_softmax_denom);

      res += group_rescaling * v;

      partial_vectors += v_in_stride;
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

void flash_decoding_final_reduce(
  hipStream_t stream,
  const float* __restrict__ partial_vectors, // shape [B, num_q_heads, G, T, embed_dim]
  const half* __restrict__ partial_softmax_denoms, // shape [B, num_q_heads, G, T]
  const half* __restrict__ partial_maxes, // [B, num_q_heads, G, T]
  half* __restrict__ hidden_out, // [B, num_q_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned B,
  unsigned G, // number of groups
  unsigned T, // num tokens to decode
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim
) {
  // TODO: support T > 1

  // v strides
  unsigned v_batch_stride = num_q_heads * G * T * embed_dim;
  unsigned v_head_stride = G * T * embed_dim;
  unsigned v_tok_stride = T * embed_dim;

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // expected block dimensions: [x=embed_dim/COLS_PER_BLOCK, y=num_q_heads, z=B]
  const dim3 num_blocks = dim3(DIV_ROUND_UP(embed_dim, COLS_PER_BLOCK), num_q_heads, B);

  flash_decoding_final_reduce_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const float*)partial_vectors,
    (const half*)partial_softmax_denoms,
    (half*)partial_maxes,
    (half*)hidden_out,
    // tensor dimension sizes
    G,
    num_q_heads,
    embed_dim,
    // v strides
    v_batch_stride,
    v_head_stride,
    v_tok_stride
  );
}

// torch wrappers

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// returns partial vectors, partial softmax denoms, partial maxes
static inline std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_decoding_partially_aggregate(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v, // [B, num_v_heads, S, embed_dim]
  torch::Tensor& m // [B, 1, S, T]
) {
  // q is expected to be contiguous
  CHECK_INPUT(q);
  // but k and v might not be due to the static cache being
  // allocated for longer sequences
  // it is fine as long as the innermost stride is 1
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  // m must be contiguous
  if (m.defined()) {
    CHECK_INPUT(m);
  }

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
  auto k_strides = k.strides().vec();
  auto v_strides = v.strides().vec();

  TORCH_CHECK(k_strides[3] == 1, "k innermost stride must be 1");
  TORCH_CHECK(v_strides[3] == 1, "v innermost stride must be 1");

  unsigned B = q_sizes[0];
  unsigned num_q_heads = q_sizes[1];
  unsigned T = q_sizes[2]; // always 1 for now
  unsigned embed_dim = q_sizes[3];
  unsigned S = k_sizes[2];
  unsigned num_k_heads = k_sizes[1];

  TORCH_CHECK(T == 1, "T > 1 not supported");

  TORCH_CHECK((k_strides[0] == v_strides[0]) && (k_strides[1] == v_strides[1]) && (k_strides[2] == v_strides[2]), "k and v strides must match");

  // TODO: adapt num_groups based on number of compute units
  unsigned G = DIV_ROUND_UP(S, MIN_TOKENS_PER_GROUP);

  auto partial_vectors = torch::empty({B, num_q_heads, G, T, embed_dim}, temp_options_f32);
  auto partial_softmax_denoms = torch::empty({B, num_q_heads, G, T}, output_options_f16);
  auto partial_maxes = torch::empty({B, num_q_heads, G, T}, output_options_f16);

  // kv strides
  unsigned kv_batch_stride = k_strides[0];
  unsigned kv_head_stride = k_strides[1];
  unsigned kv_tok_stride = k_strides[2];

  // to compute what k/v head to target for a given q head
  unsigned q_to_kv_heads = num_q_heads / num_k_heads;

  float attention_inv_scale = 1.0f / sqrtf(embed_dim);

  flash_decoding_partially_aggregate(
    stream,
    (const half*) q.data_ptr(), // shape [B, num_q_heads, T, embed_dim]
    (const half*) k.data_ptr(), // shape [B, num_k_heads, S, embed_dim]
    (const half*) v.data_ptr(), // shape [B, num_v_heads, S, embed_dim]
    m.defined() ? (const half*) m.data_ptr() : nullptr, // shape [B, 1, T, S]
    (float*) partial_vectors.data_ptr(), // [B, num_q_heads, G, T, embed_dim]
    (half*) partial_softmax_denoms.data_ptr(), // [B, num_q_heads, G, T]
    (half*) partial_maxes.data_ptr(), // [B, num_q_heads, G, T]
    // tensor dimension sizes
    B,
    G,
    T, // num tokens to decode
    S, // number of total tokens
    num_q_heads, // number of heads for q
    embed_dim,
    q_to_kv_heads,
    attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
    // kv strides
    kv_batch_stride,
    kv_head_stride,
    kv_tok_stride
  );

  return std::make_tuple(partial_vectors, partial_softmax_denoms, partial_maxes);
}

// returns transformed vectors
static inline at::Tensor flash_decoding_final_reduce(
  torch::Tensor& partial_vectors, // shape [B, num_q_heads, G, T, embed_dim]
  torch::Tensor& partial_softmax_denoms, // shape [B, num_q_heads, G, T]
  torch::Tensor& partial_maxes  // [B, num_q_heads, G, T]
) {
  CHECK_INPUT(partial_vectors);
  CHECK_INPUT(partial_softmax_denoms);
  CHECK_INPUT(partial_maxes);

  auto device = partial_vectors.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto output_options_f16 = at::TensorOptions()
                            .dtype(torch::kFloat16)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto partial_vectors_sizes = partial_vectors.sizes().vec();

  unsigned B = partial_vectors_sizes[0];
  unsigned num_q_heads = partial_vectors_sizes[1];
  unsigned G = partial_vectors_sizes[2];
  unsigned T = partial_vectors_sizes[3];
  unsigned embed_dim = partial_vectors_sizes[4];

  TORCH_CHECK(T == 1, "T > 1 not supported");

  // q_len is 1, so we can avoid the transposition
  auto hidden_out = torch::empty({B, T, num_q_heads * embed_dim}, output_options_f16);

  flash_decoding_final_reduce(
    stream,
    (const float*) partial_vectors.data_ptr(), // shape [B, num_q_heads, G, T, embed_dim]
    (const half*) partial_softmax_denoms.data_ptr(), // shape [B, num_q_heads, G, T]
    (const half*) partial_maxes.data_ptr(), // [B, num_q_heads, G, T]
    (half*) hidden_out.data_ptr(), // [B, num_q_heads, T, embed_dim]
    // tensor dimension sizes
    B,
    G, // number of groups
    T, // num tokens to decode
    num_q_heads, // number of heads for q
    embed_dim
  );

  return hidden_out;
}

at::Tensor flash_decode_no_mask(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v  // [B, num_v_heads, S, embed_dim]
) {
  auto undef_tensor = torch::Tensor();

  auto partial_aggregation_results = flash_decoding_partially_aggregate(q, k, v, /*m*/ undef_tensor);

  auto partial_vectors = std::get<0>(partial_aggregation_results);
  auto partial_softmax_denoms = std::get<1>(partial_aggregation_results);
  auto partial_maxes = std::get<2>(partial_aggregation_results);

  return flash_decoding_final_reduce(
    partial_vectors,
    partial_softmax_denoms,
    partial_maxes
  );
}

at::Tensor flash_decode_masked(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v,  // [B, num_v_heads, S, embed_dim]
  torch::Tensor& m  // [B, 1, S, T]
) {
  auto partial_aggregation_results = flash_decoding_partially_aggregate(q, k, v, m);

  auto partial_vectors = std::get<0>(partial_aggregation_results);
  auto partial_softmax_denoms = std::get<1>(partial_aggregation_results);
  auto partial_maxes = std::get<2>(partial_aggregation_results);

  return flash_decoding_final_reduce(
    partial_vectors,
    partial_softmax_denoms,
    partial_maxes
  );
}