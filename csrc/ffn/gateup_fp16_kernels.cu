#include "gateupmlpactivation.h"
#include <hip/hip_fp16.h>

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

static inline float __device__ gelu_tanh(float x) {
  // in python:
  // 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))));

  return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x * (1.0f + 0.044715f * x * x))));
}

#define FUSED_ROWS_PER_BLOCK 2

template<int THREADS_PER_BLOCK>
__global__ void muillm_gateupmlp_gemv_fp16_kernel(
    const half* __restrict__ GW, // weight matrix - size N x K
    const half* __restrict__ UW, // weight matrix - size N x K
    const half* __restrict__ X, // input = size K
    half* __restrict__ Y, // output - size N
    unsigned N,
    unsigned K,
    MuiGateUpMLPActivation activation
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  __shared__ float shared_gaccs[FUSED_ROWS_PER_BLOCK];
  __shared__ float shared_uaccs[FUSED_ROWS_PER_BLOCK];

  // initialize the shared memory
  if (threadIdx.x < FUSED_ROWS_PER_BLOCK) {
    shared_gaccs[threadIdx.x] = 0.f;
    shared_uaccs[threadIdx.x] = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  {
    int current_row = blockIdx.x * FUSED_ROWS_PER_BLOCK + 0;

    // GW
    if (current_row + 1 < N) {
      // compute the t-th element of Y. by doing the dot product with the
      // t-th row of W
      const half* GW0 = &GW[(current_row + 0) * K];
      const half* GW1 = &GW[(current_row + 1) * K];

      float gacc0 = 0.f;
      float gacc1 = 0.f;

      const half* UW0 = &UW[(current_row + 0) * K];
      const half* UW1 = &UW[(current_row + 1) * K];

      float uacc0 = 0.f;
      float uacc1 = 0.f;
  
      // do the dot product
      {
        unsigned k;
        //*
        for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
          // vectorized
          float8 x = __half82float8(*(const half8*)(addr(X, k)));

          float8 gw0 = __half82float8(load_nontemporal_half8(addr(GW0, k)));
          float8 gw1 = __half82float8(load_nontemporal_half8(addr(GW1, k)));

          float8 uw0 = __half82float8(load_nontemporal_half8(addr(UW0, k)));
          float8 uw1 = __half82float8(load_nontemporal_half8(addr(UW1, k)));
      
          dot8(gacc0, gw0, x);
          dot8(gacc1, gw1, x);
          dot8(uacc0, uw0, x);
          dot8(uacc1, uw1, x);
        }
        if (k + 3 < K) {
          // vectorized
          float4 x = __half42float4(*(const half4*)(addr(X, k)));

          float4 gw0 = __half42float4(load_nontemporal_half4(addr(GW0, k)));
          float4 gw1 = __half42float4(load_nontemporal_half4(addr(GW1, k)));
          float4 uw0 = __half42float4(load_nontemporal_half4(addr(UW0, k)));
          float4 uw1 = __half42float4(load_nontemporal_half4(addr(UW1, k)));
      
          dot4(gacc0, gw0, x);
          dot4(gacc1, gw1, x);
          dot4(uacc0, uw0, x);
          dot4(uacc1, uw1, x);

          k += 4;
        }
        if (k + 1 < K) {
          // vectorized
          float2 x = __half22float2(*(const half2*)(addr(X, k)));

          float2 gw0 = __half22float2(load_nontemporal_half2(addr(GW0, k)));
          float2 gw1 = __half22float2(load_nontemporal_half2(addr(GW1, k)));
          float2 uw0 = __half22float2(load_nontemporal_half2(addr(UW0, k)));
          float2 uw1 = __half22float2(load_nontemporal_half2(addr(UW1, k)));
      
          dot2(gacc0, gw0, x);
          dot2(gacc1, gw1, x);
          dot2(uacc0, uw0, x);
          dot2(uacc1, uw1, x);

          k += 2;
        }

        if (k < K) {
          // remainder
          float x = __half2float(*addr(X,k));

          float gw0 = __half2float(*addr(GW0,k));
          float gw1 = __half2float(*addr(GW1,k));
          float uw0 = __half2float(*addr(UW0,k));
          float uw1 = __half2float(*addr(UW1,k));

          gacc0 += gw0 * x;
          gacc1 += gw1 * x;
          uacc0 += uw0 * x;
          uacc1 += uw1 * x;
        }
      }

      // warp reduce
      gacc0 = warpReduce(gacc0);
      gacc1 = warpReduce(gacc1);
      uacc0 = warpReduce(uacc0);
      uacc1 = warpReduce(uacc1);

      // reduce accross warps
      if (laneId == 0) {
        atomicAdd(&shared_gaccs[0], gacc0);
        atomicAdd(&shared_gaccs[1], gacc1);
        atomicAdd(&shared_uaccs[0], uacc0);
        atomicAdd(&shared_uaccs[1], uacc1);
      }
    } else {
      for (int i = 0; i < FUSED_ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * FUSED_ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const half* GW_ = &GW[current_row * K];
        const half* UW_ = &UW[current_row * K];
      
        // do the dot product
        float gacc = 0.f;
        float uacc = 0.f;

        for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
          float x =  __half2float(X[k]);
          float gw = __half2float(GW_[k]);
          float uw = __half2float(UW_[k]);
          gacc += gw * x;
          uacc += uw * x;
        }

        // warp reduce
        gacc = warpReduce(gacc);
        uacc = warpReduce(uacc);

        // reduce accross warps
        if (laneId == 0) {
          atomicAdd(&shared_gaccs[i], gacc);
          atomicAdd(&shared_uaccs[i], uacc);
        }
      }
    }
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  // write out the results
  {
    if (threadIdx.x >= FUSED_ROWS_PER_BLOCK)
      return;

    int current_row = blockIdx.x * FUSED_ROWS_PER_BLOCK + threadIdx.x;

    if (current_row < N) {
      float gacc = shared_gaccs[threadIdx.x]; // read the fully reduced value
      float uacc = shared_uaccs[threadIdx.x]; // read the fully reduced value
      float acc;
      
      if (activation == MuiGateUpMLPActivation::SILU) {
        acc = silu(gacc) * uacc;
      } else if (activation == MuiGateUpMLPActivation::GELU_TANH) {
        acc = gelu_tanh(gacc) * uacc;
      } else {
        // unsupported activation
        acc = 0.f;
      }

      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}

template<int THREADS_PER_BLOCK>
__global__ void muillm_gateupmlp_gemv_norm_inputs_fp16_kernel(
    const half* __restrict__ NW, // input normalization weights matrix - size K
    const half* __restrict__ GW, // weight matrix - size N x K
    const half* __restrict__ UW, // weight matrix - size N x K
    const half* __restrict__ X, // input = size K
    half* __restrict__ Y, // output - size N
    unsigned N,
    unsigned K,
    float epsilon,
    float weights_offset,
    float scale,
    MuiGateUpMLPActivation activation
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  float var_x = 0.f;

  __shared__ float shared_gaccs[FUSED_ROWS_PER_BLOCK];
  __shared__ float shared_uaccs[FUSED_ROWS_PER_BLOCK];
  __shared__ float shared_var_x;

  // initialize the shared memory
  if (threadIdx.x < FUSED_ROWS_PER_BLOCK) {
    shared_gaccs[threadIdx.x] = 0.f;
    shared_uaccs[threadIdx.x] = 0.f;
  }
  if (threadIdx.x == 0) {
    shared_var_x = epsilon;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  {
    int current_row = blockIdx.x * FUSED_ROWS_PER_BLOCK + 0;

    // GW
    if (current_row + 1 < N) {
      // compute the t-th element of Y. by doing the dot product with the
      // t-th row of W
      const half* GW0 = &GW[(current_row + 0) * K];
      const half* GW1 = &GW[(current_row + 1) * K];

      const half* UW0 = &UW[(current_row + 0) * K];
      const half* UW1 = &UW[(current_row + 1) * K];

      float gacc0 = 0.f;
      float gacc1 = 0.f;
      float uacc0 = 0.f;
      float uacc1 = 0.f;

      // do the dot product
      {
        unsigned k; // should be 2 * tidx ?
        //*
        for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
          // vectorized
          float8 x = __half82float8(*(const half8*)(addr(X, k)));
          float8 nw = __half82float8(*(const half8*)(addr(NW, k))) + weights_offset;

          float8 gw0 = __half82float8(load_nontemporal_half8(addr(GW0, k)));
          float8 gw1 = __half82float8(load_nontemporal_half8(addr(GW1, k)));
          float8 uw0 = __half82float8(load_nontemporal_half8(addr(UW0, k)));
          float8 uw1 = __half82float8(load_nontemporal_half8(addr(UW1, k)));

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

          dot8(gacc0, gw0, x);
          dot8(gacc1, gw1, x);
          dot8(uacc0, uw0, x);
          dot8(uacc1, uw1, x);
        }
        if (k + 3 < K) {
          // vectorized
          float4 x = __half42float4(*(const half4*)(addr(X, k)));
          float4 nw = __half42float4(*(const half4*)(addr(NW, k))) + weights_offset;

          float4 gw0 = __half42float4(load_nontemporal_half4(addr(GW0, k)));
          float4 gw1 = __half42float4(load_nontemporal_half4(addr(GW1, k)));
          float4 uw0 = __half42float4(load_nontemporal_half4(addr(UW0, k)));
          float4 uw1 = __half42float4(load_nontemporal_half4(addr(UW1, k)));

          // accumulate for the variance
          dot4(var_x, x, x);

          // multiply with normalization weights
          x.x = x.x * nw.x;
          x.y = x.y * nw.y;
          x.z = x.z * nw.z;
          x.w = x.w * nw.w;

          dot4(gacc0, gw0, x);
          dot4(gacc1, gw1, x);
          dot4(uacc0, uw0, x);
          dot4(uacc1, uw1, x);

          k += 4;
        }
        if (k + 1 < K) {
          // vectorized
          float2 x = __half22float2(*(const half2*)(addr(X, k)));
          float2 nw = __half22float2(*(const half2*)(addr(NW, k))) + weights_offset;

          float2 gw0 = __half22float2(load_nontemporal_half2(addr(GW0, k)));
          float2 gw1 = __half22float2(load_nontemporal_half2(addr(GW1, k)));
          float2 uw0 = __half22float2(load_nontemporal_half2(addr(UW0, k)));
          float2 uw1 = __half22float2(load_nontemporal_half2(addr(UW1, k)));
  
          // accumulate for the variance
          dot2(var_x, x, x);

          // multiply with normalization weights
          x.x = x.x * nw.x;
          x.y = x.y * nw.y;

          dot2(gacc0, gw0, x);
          dot2(gacc1, gw1, x);
          dot2(uacc0, uw0, x);
          dot2(uacc1, uw1, x);

          k += 2;
        }

        if (k < K) {
          // remainder
          float x = __half2float(*addr(X,k));
          float nw = __half2float(*addr(NW,k)) + weights_offset;

          float gw0 = __half2float(*addr(GW0,k));
          float gw1 = __half2float(*addr(GW1,k));
          float uw0 = __half2float(*addr(UW0,k));
          float uw1 = __half2float(*addr(UW1,k));

          // accumulate for the variance
          var_x += x * x;

          // multiply with normalization weights
          x *= nw;

          gacc0 += gw0 * x;
          gacc1 += gw1 * x;
          uacc0 += uw0 * x;
          uacc1 += uw1 * x;
        }
      }

      // warp reduce
      var_x = warpReduce(var_x);
      gacc0 = warpReduce(gacc0);
      gacc1 = warpReduce(gacc1);
      uacc0 = warpReduce(uacc0);
      uacc1 = warpReduce(uacc1);

      // reduce accross warps
      if (laneId == 0) {
        atomicAdd(&shared_var_x, var_x);
        atomicAdd(&shared_gaccs[0], gacc0);
        atomicAdd(&shared_gaccs[1], gacc1);
        atomicAdd(&shared_uaccs[0], uacc0);
        atomicAdd(&shared_uaccs[1], uacc1);
      }
    } else {
      for (int i = 0; i < FUSED_ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * FUSED_ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const half* GW_ = &GW[current_row * K];
        const half* UW_ = &UW[current_row * K];
      
        // do the dot product
        float gacc = 0.f;
        float uacc = 0.f;

        if (i == 0) {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float x =  __half2float(X[k]);
            float nw = __half2float(NW[k]) + weights_offset;

            // accumuate the variance
            var_x += x * x;

            // multiply with normalization weights
            x *= nw;

            float gw = __half2float(GW_[k]);
            float uw = __half2float(UW_[k]);
            gacc += gw * x;
            uacc += uw * x;
          }
        } else {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float x =  __half2float(X[k]);
            float nw = __half2float(NW[k]) + weights_offset;

            // don't accumulate the variance (we already have done it with i == 0)

            // multiply with normalization weights
            x *= nw;

            float gw = __half2float(GW_[k]);
            float uw = __half2float(UW_[k]);

            gacc += gw * x;
            uacc += uw * x;
          }
        }

        // warp reduce
        var_x = warpReduce(var_x);
        gacc = warpReduce(gacc);
        uacc = warpReduce(uacc);

        // reduce accross warps
        if (laneId == 0) {
          atomicAdd(&shared_var_x, var_x);
          atomicAdd(&shared_gaccs[i], gacc);
          atomicAdd(&shared_uaccs[i], uacc);
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

    if (threadIdx.x >= FUSED_ROWS_PER_BLOCK)
      return;

    int current_row = blockIdx.x * FUSED_ROWS_PER_BLOCK + threadIdx.x;

    if (current_row < N) {
      float gacc = shared_gaccs[threadIdx.x] * rsqrt_var; // read the fully reduced value and scale
      float uacc = shared_uaccs[threadIdx.x] * rsqrt_var; // read the fully reduced value and scale
      float acc;
      
      if (activation == MuiGateUpMLPActivation::SILU) {
        acc = silu(gacc) * uacc;
      } else if (activation == MuiGateUpMLPActivation::GELU_TANH) {
        acc = gelu_tanh(gacc) * uacc;
      } else {
        // unsupported activation
        acc = 0.f;
      }

      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}

#define SPLIT_ROWS_PER_BLOCK 4

template<int THREADS_PER_BLOCK>
__global__ void muillm_gateupmlp_gemv_norm_inputs_split_fp16_kernel(
    const half* __restrict__ NW, // input normalization weights matrix - size K
    const half* __restrict__ GW, // weight matrix - size N x K
    const half* __restrict__ UW, // weight matrix - size N x K
    const half* __restrict__ X, // input = size K
    half* __restrict__ GY, // output - size N
    half* __restrict__ UY, // output - size N
    unsigned N,
    unsigned K,
    float epsilon,
    float weights_offset,
    float scale
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;


  const half* __restrict__ W = blockIdx.y == 0 ? GW : UW; // weight matrix - size N x K
  half* __restrict__ Y = blockIdx.y == 0 ? GY : UY; // output - size N

  float var_x = 0.f;

  __shared__ float shared_accs[SPLIT_ROWS_PER_BLOCK];
  __shared__ float shared_var_x;

  // initialize the shared memory
  if (threadIdx.x < SPLIT_ROWS_PER_BLOCK) {
    shared_accs[threadIdx.x] = 0.f;
  }
  if (threadIdx.x == 0) {
    shared_var_x = epsilon;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  {
    int current_row = blockIdx.x * SPLIT_ROWS_PER_BLOCK + 0;

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
          // vectorized
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

          k += 2;
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
      for (int i = 0; i < SPLIT_ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * SPLIT_ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const half* W_ = &W[current_row * K];
      
        // do the dot product
        float acc = 0.f;
        if (i == 0) {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float x =  __half2float(X[k]);
            float nw = __half2float(NW[k]) + weights_offset;

            // accumuate the variance
            var_x += x * x;

            // multiply with normalization weights
            x *= nw;

            float w = __half2float(W_[k]);
            acc += w * x;
          }
        } else {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float x =  __half2float(X[k]);
            float nw = __half2float(NW[k]) + weights_offset;

            // don't accumulate the variance (we already have done it with i == 0)

            // multiply with normalization weights
            x *= nw;

            float w = __half2float(W_[k]);
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

    if (threadIdx.x >= SPLIT_ROWS_PER_BLOCK)
      return;

    int current_row = blockIdx.x * SPLIT_ROWS_PER_BLOCK + threadIdx.x;

    if (current_row < N) {
      float acc = shared_accs[threadIdx.x] * rsqrt_var; // read the fully reduced value and scale

      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}

void muillm_gateupmlp_forward_fp16(
  hipStream_t stream,
  MuiGateUpMLPActivation activation,
  unsigned N,
  unsigned K,
  const half* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const half* gate_weights,
  const half* up_weights,
  const half* x,
  half* y,
  int simd_lanes
) {

  bool normalize = norm_weights != nullptr;

  const int num_blocks = DIV_ROUND_UP(N, FUSED_ROWS_PER_BLOCK);
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
      muillm_gateupmlp_gemv_norm_inputs_fp16_kernel<64><<<num_blocks, threads_per_blocks, 0, stream>>>(
        norm_weights,
        gate_weights,
        up_weights,
        x,
        y,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale,
        activation
      );
    } else if (threads_per_blocks == 128) {
      muillm_gateupmlp_gemv_norm_inputs_fp16_kernel<128><<<num_blocks, threads_per_blocks, 0, stream>>>(
        norm_weights,
        gate_weights,
        up_weights,
        x,
        y,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale,
        activation
      );
    } else if (threads_per_blocks == 256) {
      muillm_gateupmlp_gemv_norm_inputs_fp16_kernel<256><<<num_blocks, threads_per_blocks, 0, stream>>>(
        norm_weights,
        gate_weights,
        up_weights,
        x,
        y,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale,
        activation
      );
    }
  } else {

    if (threads_per_blocks == 64) {
      muillm_gateupmlp_gemv_fp16_kernel<64><<<num_blocks, threads_per_blocks, 0, stream>>>(
        gate_weights,
        up_weights,
        x,
        y,
        N,
        K,
        activation
      );
    } else if (threads_per_blocks == 128) {
      muillm_gateupmlp_gemv_fp16_kernel<128><<<num_blocks, threads_per_blocks, 0, stream>>>(
        gate_weights,
        up_weights,
        x,
        y,
        N,
        K,
        activation
      );
    } else if (threads_per_blocks == 256) {
      muillm_gateupmlp_gemv_fp16_kernel<256><<<num_blocks, threads_per_blocks, 0, stream>>>(
        gate_weights,
        up_weights,
        x,
        y,
        N,
        K,
        activation
      );
    }
  }
}

template<int THREADS_PER_BLOCK>
__global__ void muillm_gateupmlp_gemv_split_fp16_kernel(
    const half* __restrict__ GW, // weight matrix - size N x K
    const half* __restrict__ UW, // weight matrix - size N x K
    const half* __restrict__ X, // input = size K
    half* __restrict__ GY, // output - size N
    half* __restrict__ UY, // output - size N
    unsigned N,
    unsigned K
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  const half* __restrict__ W = blockIdx.y == 0 ? GW : UW;
  half* __restrict__ Y = blockIdx.y == 0 ? GY : UY;

  __shared__ float shared_accs[SPLIT_ROWS_PER_BLOCK];

  // initialize the shared memory
  if (threadIdx.x < SPLIT_ROWS_PER_BLOCK) {
    shared_accs[threadIdx.x] = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  {
    int current_row = blockIdx.x * SPLIT_ROWS_PER_BLOCK + 0;

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
        unsigned k; // should be 2 * tidx ?
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
          // vectorized
          float2 x = __half22float2(*(const half2*)(addr(X, k)));

          float2 w0 = __half22float2(load_nontemporal_half2(addr(W0, k)));
          float2 w1 = __half22float2(load_nontemporal_half2(addr(W1, k)));
          float2 w2 = __half22float2(load_nontemporal_half2(addr(W2, k)));
          float2 w3 = __half22float2(load_nontemporal_half2(addr(W3, k)));

          dot2(acc0, w0, x);
          dot2(acc1, w1, x);
          dot2(acc2, w2, x);
          dot2(acc3, w3, x);

          k += 2;
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
      for (int i = 0; i < SPLIT_ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * SPLIT_ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const half* W_ = &W[current_row * K];
      
        // do the dot product
        float acc = 0.f;
        for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
          float x =  __half2float(X[k]);
          float w = __half2float(W_[k]);
          acc += w * x;
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

  // write out the results
  {

    if (threadIdx.x >= SPLIT_ROWS_PER_BLOCK)
      return;

    int current_row = blockIdx.x * SPLIT_ROWS_PER_BLOCK + threadIdx.x;

    if (current_row < N) {
      float acc = shared_accs[threadIdx.x]; // read the fully reduced value and scale

      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}

template<int THREADS_PER_BLOCK>
__global__ void muillm_gateupmlp_combine_fp16_kernel(
    const half* __restrict__ GY, // input - size N
    const half* __restrict__ UY, // input - size N
    half* __restrict__ Y, // output - size N
    unsigned N,
    MuiGateUpMLPActivation activation
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  int current_row = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  if (current_row < N) {
    float g = __half2float(GY[current_row]);
    float u = __half2float(UY[current_row]);
    float y;

    if (activation == MuiGateUpMLPActivation::SILU) {
      y = silu(g) * u;
    } else if (activation == MuiGateUpMLPActivation::GELU_TANH) {
      y = gelu_tanh(g) * u;
    } else {
      // unsupported activation
      y = 0.f;
    }

    // write the output value
    Y[current_row] = __float2half(y);
  }
}

void muillm_gateupmlp_split_forward_fp16(
  hipStream_t stream,
  MuiGateUpMLPActivation activation,
  unsigned N,
  unsigned K,
  const half* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const half* gate_weights,
  const half* up_weights,
  const half* x,
  half* gy,
  half* uy,
  half* y,
  int simd_lanes
) {

  bool normalize = norm_weights != nullptr;

  const int num_blocks = DIV_ROUND_UP(N, SPLIT_ROWS_PER_BLOCK);
  int threads_per_blocks = GEMV_THREADS_PER_BLOCK;

  // try to occupy enough to saturate memory bandwidth
  /*
  while ((num_blocks * threads_per_blocks < 8 * simd_lanes) && threads_per_blocks < 256) {
    threads_per_blocks *= 2;
  }
  */

  // Do GEMVs (some blocks the gate proj, some the up proj)
  if (normalize) {
    float scale = 1.f / K;

    if (threads_per_blocks == 64) {
      muillm_gateupmlp_gemv_norm_inputs_split_fp16_kernel<64><<<dim3(num_blocks, 2), threads_per_blocks, 0, stream>>>(
        norm_weights,
        gate_weights,
        up_weights,
        x,
        gy,
        uy,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    } else if (threads_per_blocks == 128) {
      muillm_gateupmlp_gemv_norm_inputs_split_fp16_kernel<128><<<dim3(num_blocks, 2), threads_per_blocks, 0, stream>>>(
        norm_weights,
        gate_weights,
        up_weights,
        x,
        gy,
        uy,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    } else if (threads_per_blocks == 256) {
      muillm_gateupmlp_gemv_norm_inputs_split_fp16_kernel<256><<<dim3(num_blocks, 2), threads_per_blocks, 0, stream>>>(
        norm_weights,
        gate_weights,
        up_weights,
        x,
        gy,
        uy,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    }
  } else {

    if (threads_per_blocks == 64) {
      muillm_gateupmlp_gemv_split_fp16_kernel<64><<<dim3(num_blocks, 2), threads_per_blocks, 0, stream>>>(
        gate_weights,
        up_weights,
        x,
        gy,
        uy,
        N,
        K
      );
    } else if (threads_per_blocks == 128) {
      muillm_gateupmlp_gemv_split_fp16_kernel<128><<<dim3(num_blocks, 2), threads_per_blocks, 0, stream>>>(
        gate_weights,
        up_weights,
        x,
        gy,
        uy,
        N,
        K
      );
    } else if (threads_per_blocks == 256) {
      muillm_gateupmlp_gemv_split_fp16_kernel<256><<<dim3(num_blocks, 2), threads_per_blocks, 0, stream>>>(
        gate_weights,
        up_weights,
        x,
        gy,
        uy,
        N,
        K
      );
    }
  }

  // do final reduction
  const int num_blocks_combine = DIV_ROUND_UP(N, threads_per_blocks);
  if (threads_per_blocks == 64) {
    muillm_gateupmlp_combine_fp16_kernel<64><<<num_blocks_combine, threads_per_blocks, 0, stream>>>(
      gy,
      uy,
      y,
      N,
      activation
    );
  } else if (threads_per_blocks == 128) {
    muillm_gateupmlp_combine_fp16_kernel<128><<<num_blocks_combine, threads_per_blocks, 0, stream>>>(
      gy,
      uy,
      y,
      N,
      activation
    );
  } else if (threads_per_blocks == 256) {
    muillm_gateupmlp_combine_fp16_kernel<256><<<num_blocks_combine, threads_per_blocks, 0, stream>>>(
      gy,
      uy,
      y,
      N,
      activation
    );
  }
}