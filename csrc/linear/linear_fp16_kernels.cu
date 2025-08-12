#include "activation.h"

#include <hip/hip_fp16.h>

//
// actual module
//

#define ROWS_PER_BLOCK 4
#define GEMV_THREADS_PER_BLOCK 256

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define FULL_MASK32 0xffffffff
#define FULL_MASK64 0xffffffffffffffff

#ifdef  __CUDA_ARCH__
#define __xx_shfl_down(mask, val, offset) __shfl_down_sync(mask, val, offset)
#elif defined(__HIP_PLATFORM_AMD__) // AMD
#define __xx_shfl_down(mask, val, offset) __shfl_down(val, offset)
#else
#error "Unsupported compiler"
#endif

__device__ float warpReduce(float val) {
  if (warpSize == 32) {
    for (int offset = 16; offset > 0; offset /= 2)
      val += __xx_shfl_down(FULL_MASK32, val, offset);
  }
  if (warpSize == 64) {
    for (int offset = 32; offset > 0; offset /= 2)
      val += __xx_shfl_down(FULL_MASK64, val, offset);

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

__device__ inline float8 operator+(const float8& a, const float b) {
  float8 r;
  r.x = a.x + b;
  r.y = a.y + b;
  r.z = a.z + b;
  r.w = a.w + b;
  r.a = a.a + b;
  r.b = a.b + b;
  r.c = a.c + b;
  r.d = a.d + b;
  return r;
}

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

template <typename T>
static inline const T* __device__ addr(const T* p, unsigned index) {
  // helps the AMDGPU compiler understand it can use the sgrp pair + single vgpr addressing mode
  unsigned byte_offset = sizeof(T) * index;
  const uint8_t* p8 = (const uint8_t*)p;
  return (const T*) (p8 + byte_offset);
}

static inline float __device__ silu(float x) {
  return x / (1.0f + expf(-x));
}

template<int THREADS_PER_BLOCK>
__global__ void muillm_gemv_fp16_kernel(
    const half* __restrict__ W, // weight matrix - size N x K
    const half* __restrict__ X, // input = size K
    mui_activation activation, // activation function 
    const half* __restrict__ MB, // optional multiplicative bias - size N (applied before additive bias)
    const half* __restrict__ AB, // optional additive bias - size N
    const half* __restrict__ RB, // optional residual - size N
    half* __restrict__ Y, // output - size N
    unsigned N,
    unsigned K
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  // can process ROWS_PER_BLOCK rows
  // shared state to do the reductions

  // TODO: avoid bank conflicts by having per warp shared memory
  __shared__ float shared_accs[ROWS_PER_BLOCK];

  // initialize the shared memory
  if (threadIdx.x < ROWS_PER_BLOCK) {
    shared_accs[threadIdx.x] = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  {
    int current_row = blockIdx.x * ROWS_PER_BLOCK + 0;
    if (current_row + 3 < N) {

      // compute the t-th element of Y. by doing the dot product with the
      // t-th row of W
      const half* W0 = &W[(current_row + 0) * K];
      const half* W1 = &W[(current_row + 1) * K];
      const half* W2 = &W[(current_row + 2) * K];
      const half* W3 = &W[(current_row + 3) * K];

      float acc0 = 0.f;
      float acc1 = 0.f;
      float acc2 = 0.f;
      float acc3 = 0.f;

      // do the dot product
      {
        unsigned k;
        //*
        for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
          // vectorized
          float8 x = __half82float8(*(const half8*)(addr(X, k)));

          float8 w0 = __half82float8(load_nontemporal_half8(addr(W0, k)));
          float8 w1 = __half82float8(load_nontemporal_half8(addr(W1, k)));
          float8 w2 = __half82float8(load_nontemporal_half8(addr(W2, k)));
          float8 w3 = __half82float8(load_nontemporal_half8(addr(W3, k)));

          dot8(acc0, w0, x);
          dot8(acc1, w1, x);
          dot8(acc2, w2, x);
          dot8(acc3, w3, x);
        }
        if (k + 3 < K) {
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

          k += 4;
        }
        if (k + 1 < K) {
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
        if (k < K) {
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
      for (int i = 0; i < ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const half* W_ = &W[current_row * K];
      
        // do the dot product
        float acc = 0.f;
        {
          int k = threadIdx.x  * 2;
          for (; k + 1 < K; k += THREADS_PER_BLOCK * 2) {
            float2 w = __half22float2(*(const half2*)&W_[k]);
            float2 x = __half22float2(*(const half2*)&X[k]);
            dot2(acc, w, x);
          }
          if (k < K) {
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

  // write out the results
  {
    if (threadIdx.x >= ROWS_PER_BLOCK)
      return;

    int current_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x;

    if (current_row < N) {
      float acc = shared_accs[threadIdx.x]; // read the fully reduced value

      if (activation == mui_activation::Silu) {
        // apply the activation if there is one
        acc = silu(acc);
      }

      if (MB != nullptr) { // apply the multipicative bias if there is one
        acc *= __half2float(MB[current_row]);
      }

      if (AB != nullptr) { // apply the additive bias if there is one
        acc += __half2float(AB[current_row]);
      }
      if (RB != nullptr) { // apply the residual if there is one
        acc += __half2float(RB[current_row]);
      }
      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}

template<int THREADS_PER_BLOCK>
__global__ void muillm_gemv_norm_inputs_fp16_kernel(
    const half* __restrict__ NW, // input normalization weights matrix - size K
    const half* __restrict__ W, // weight matrix - size N x K
    const half* __restrict__ X, // input = size K
    mui_activation activation, // activation function 
    const half* __restrict__ MB, // optional multiplicative bias - size N (applied before additive bias)
    const half* __restrict__ AB, // optional additive bias - size N
    const half* __restrict__ RB, // optional residual - size N
    half* __restrict__ Y, // output - size N
    unsigned N,
    unsigned K,
    float epsilon,
    float weights_offset,
    float scale
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  float var_x = 0.f;

  // can process ROWS_PER_BLOCK rows
  // shared state to do the reductions
  __shared__ float shared_accs[ROWS_PER_BLOCK];
  __shared__ float shared_var_x;

  // initialize the shared memory
  if (threadIdx.x < ROWS_PER_BLOCK) {
    shared_accs[threadIdx.x] = 0.f;
  }
  if (threadIdx.x == 0) {
    shared_var_x = epsilon;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  {
    int current_row = blockIdx.x * ROWS_PER_BLOCK + 0;
    if (current_row + 3 < N) {

      // compute the t-th element of Y. by doing the dot product with the
      // t-th row of W
      const half* W0 = &W[(current_row + 0) * K];
      const half* W1 = &W[(current_row + 1) * K];
      const half* W2 = &W[(current_row + 2) * K];
      const half* W3 = &W[(current_row + 3) * K];

      float acc0 = 0.f;
      float acc1 = 0.f;
      float acc2 = 0.f;
      float acc3 = 0.f;

      // do the dot product
      {
        // need to normalize the inputs
  
        unsigned k; // should be 2 * tidx ?
        //*
        for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
          // vectorized
          float8 x = __half82float8(*(const half8*)(addr(X, k)));
          float8 nw = __half82float8(*(const half8*)(addr(NW, k))) + weights_offset;

          float8 w0 = __half82float8(load_nontemporal_half8(addr(W0, k)));
          float8 w1 = __half82float8(load_nontemporal_half8(addr(W1, k)));
          float8 w2 = __half82float8(load_nontemporal_half8(addr(W2, k)));
          float8 w3 = __half82float8(load_nontemporal_half8(addr(W3, k)));

          // accumulate for the variance
          dot8(var_x, x, x);

          // multiply with normalization weights
          x.x = x.x * nw.x;
          x.y = x.y * nw.y;
          x.z = x.z * nw.z;
          x.w = x.w * nw.w;
          x.a = x.a * nw.a;
          x.b = x.b * nw.b;
          x.c = x.c * nw.c;
          x.d = x.d * nw.d;

          dot8(acc0, w0, x);
          dot8(acc1, w1, x);
          dot8(acc2, w2, x);
          dot8(acc3, w3, x);
        }
        if (k + 3 < K) {
          // vectorized
          float4 x = __half42float4(*(const half4*)(addr(X, k)));
          float4 nw = __half42float4(*(const half4*)(addr(NW, k))) + weights_offset;

          float4 w0 = __half42float4(load_nontemporal_half4(addr(W0, k)));
          float4 w1 = __half42float4(load_nontemporal_half4(addr(W1, k)));
          float4 w2 = __half42float4(load_nontemporal_half4(addr(W2, k)));
          float4 w3 = __half42float4(load_nontemporal_half4(addr(W3, k)));

          // accumulate for the variance
          dot4(var_x, x, x);

          // multiply with normalization weights
          x.x = x.x * nw.x;
          x.y = x.y * nw.y;
          x.z = x.z * nw.z;
          x.w = x.w * nw.w;

          dot4(acc0, w0, x);
          dot4(acc1, w1, x);
          dot4(acc2, w2, x);
          dot4(acc3, w3, x);

          k += 4;
        }
        if (k + 1 < K) {
          // remainder
          float2 x = __half22float2(*(const half2*)(addr(X, k)));
          float2 nw = __half22float2(*(const half2*)(addr(NW, k))) + weights_offset;

          float2 w0 = __half22float2(load_nontemporal_half2(addr(W0, k)));
          float2 w1 = __half22float2(load_nontemporal_half2(addr(W1, k)));
          float2 w2 = __half22float2(load_nontemporal_half2(addr(W2, k)));
          float2 w3 = __half22float2(load_nontemporal_half2(addr(W3, k)));

          // accumulate for the variance
          dot2(var_x, x, x);

          // multiply with normalization weights
          x.x = x.x * nw.x;
          x.y = x.y * nw.y;

          dot2(acc0, w0, x);
          dot2(acc1, w1, x);
          dot2(acc2, w2, x);
          dot2(acc3, w3, x);
        }
        if (k < K) {
          // remainder
          float x = __half2float(*addr(X,k));
          float nw = __half2float(*addr(NW,k)) + weights_offset;


          float w0 = __half2float(*addr(W0,k));
          float w1 = __half2float(*addr(W1,k));
          float w2 = __half2float(*addr(W2,k));
          float w3 = __half2float(*addr(W3,k));

          // accumulate for the variance
          var_x += x * x;

          // multiply with normalization weights
          x *= nw;

          acc0 += w0 * x;
          acc1 += w1 * x;
          acc2 += w2 * x;
          acc3 += w3 * x;
        }
      }

      // warp reduce
      var_x = warpReduce(var_x);
      acc0 = warpReduce(acc0);
      acc1 = warpReduce(acc1);
      acc2 = warpReduce(acc2);
      acc3 = warpReduce(acc3);

      // reduce accross warps
      if (laneId == 0) {
        atomicAdd(&shared_var_x, var_x);
        atomicAdd(&shared_accs[0], acc0);
        atomicAdd(&shared_accs[1], acc1);
        atomicAdd(&shared_accs[2], acc2);
        atomicAdd(&shared_accs[3], acc3);
      }
    } else {
      for (int i = 0; i < ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const half* W_ = &W[current_row * K];
      
        // do the dot product
        float acc = 0.f;
        if (i == 0) {
          // accumulate the variance
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float w = __half2float(W_[k]);

            float x = __half2float(X[k]);
            float nw = __half2float(NW[k]) + weights_offset;

            // accumuate the variance
            var_x += x * x;

            // multiply with normalization weights
            x *= nw;

            acc += w * x;
          }
        } else {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float w = __half2float(W_[k]);

            float x = __half2float(X[k]);
            float nw = __half2float(NW[k]) + weights_offset;

            // don't accumulate the variance (we already have done it with i == 0)

            // multiply with normalization weights
            x *= nw;

            acc += w * x;
          }
        }


        // warp reduce
        var_x = warpReduce(var_x);
        acc = warpReduce(acc);

        // reduce accross warps
        if (laneId == 0) {
          atomicAdd(&shared_var_x, var_x);
          atomicAdd(&shared_accs[i], acc);
        }
      }
    }
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  // write out the results
  {
    float rsqrt_var = rsqrtf(shared_var_x * scale);

    if (threadIdx.x >= ROWS_PER_BLOCK)
      return;

    int current_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x;

    if (current_row < N) {
      float acc = shared_accs[threadIdx.x] * rsqrt_var; // read the fully reduced value and scale

      if (activation == mui_activation::Silu) {
        // apply the activation if there is one
        acc = silu(acc);
      }

      if (MB != nullptr) { // apply the multipicative bias if there is one
        acc *= __half2float(MB[current_row]);
      }

      if (AB != nullptr) { // apply the additive bias if there is one
        acc += __half2float(AB[current_row]);
      }
      if (RB != nullptr) { // apply the residual if there is one
        acc += __half2float(RB[current_row]);
      }
      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}

void muillm_linear_activ_forward_fp16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  const half* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const half* weights,
  mui_activation activ,
  const half* mul_bias,
  const half* add_bias,
  const half* residual,
  const half* x,
  half* y,
  int simd_lanes
) {

  bool normalize = (norm_weights != nullptr);

  const int num_blocks = DIV_ROUND_UP(N, ROWS_PER_BLOCK);
  int threads_per_blocks = GEMV_THREADS_PER_BLOCK;

  // try to occupy enough to saturate memory bandwidth
  /*
  while ((num_blocks * threads_per_blocks < 8 * simd_lanes) && threads_per_blocks < 256) {
    threads_per_blocks *= 2;
  }
  */

  if (normalize) {
    float scale = 1.f / K;

    if (threads_per_blocks == 64) {
      muillm_gemv_norm_inputs_fp16_kernel<64><<<num_blocks, threads_per_blocks, 0, stream>>>(
        norm_weights,
        weights,
        x,
        activ,
        mul_bias,
        add_bias,
        residual,
        y,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    } else if (threads_per_blocks == 128) {
      muillm_gemv_norm_inputs_fp16_kernel<128><<<num_blocks, threads_per_blocks, 0, stream>>>(
        norm_weights,
        weights,
        x,
        activ,
        mul_bias,
        add_bias,
        residual,
        y,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    } else if (threads_per_blocks == 256) {
      muillm_gemv_norm_inputs_fp16_kernel<256><<<num_blocks, threads_per_blocks, 0, stream>>>(
        norm_weights,
        weights,
        x,
        activ,
        mul_bias,
        add_bias,
        residual,
        y,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    }
  } else {

    if (threads_per_blocks == 64) {
      muillm_gemv_fp16_kernel<64><<<num_blocks, threads_per_blocks, 0, stream>>>(
        weights,
        x,
        activ,
        mul_bias,
        add_bias,
        residual,
        y,
        N,
        K
      );
    } else if (threads_per_blocks == 128) {
      muillm_gemv_fp16_kernel<128><<<num_blocks, threads_per_blocks, 0, stream>>>(
        weights,
        x,
        activ,
        mul_bias,
        add_bias,
        residual,
        y,
        N,
        K
      );
    } else if (threads_per_blocks == 256) {
      muillm_gemv_fp16_kernel<256><<<num_blocks, threads_per_blocks, 0, stream>>>(
        weights,
        x,
        activ,
        mul_bias,
        add_bias,
        residual,
        y,
        N,
        K
      );
    }
  }
}