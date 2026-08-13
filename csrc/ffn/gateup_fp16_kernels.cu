#include "gateupmlpactivation.h"
#include <hip/hip_fp16.h>
#include <stdexcept>

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

template<int THREADS_PER_BLOCK, int BATCH_SIZE, int FUSED_ROWS_PER_BLOCK>
__device__ void muillm_gateupmlp_gemv_fp16_func(
    const half* __restrict__ GW, // weight matrix - size N x K
    const half* __restrict__ UW, // weight matrix - size N x K
    const half* __restrict__ X, // input = size B x K
    half* __restrict__ Y, // output - size B x N
    unsigned N,
    unsigned K,
    MuiGateUpMLPActivation activation
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  __shared__ float shared_gaccs[BATCH_SIZE][FUSED_ROWS_PER_BLOCK];
  __shared__ float shared_uaccs[BATCH_SIZE][FUSED_ROWS_PER_BLOCK];

  // initialize the shared memory
  if (threadIdx.x < FUSED_ROWS_PER_BLOCK) {
    for (int b = 0; b < BATCH_SIZE; b++) {
      shared_gaccs[b][threadIdx.x] = 0.f;
      shared_uaccs[b][threadIdx.x] = 0.f;
    }
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
      const half* GWs[FUSED_ROWS_PER_BLOCK];
      for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
        GWs[r] = &GW[(current_row + r) * K];
      }

      const half* UWs[FUSED_ROWS_PER_BLOCK];
      for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
        UWs[r] = &UW[(current_row + r) * K];
      }

      const half* Xps[BATCH_SIZE];
      for (int b = 0; b < BATCH_SIZE; b++) {
        Xps[b] = &X[b * K];
      }
  
      float gaccs[BATCH_SIZE][FUSED_ROWS_PER_BLOCK];
      float uaccs[BATCH_SIZE][FUSED_ROWS_PER_BLOCK];
      for (int b = 0; b < BATCH_SIZE; b++) {
        for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
          gaccs[b][r] = 0.f;
          uaccs[b][r] = 0.f;
        }
      }

      // do the dot product
      {
        unsigned k;
        //*
        for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
          // vectorized
          float8 gws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            gws[r] = __half82float8(load_nontemporal_half8(addr(GWs[r], k)));
          }

          float8 uws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            uws[r] = __half82float8(load_nontemporal_half8(addr(UWs[r], k)));
          }
      
          for (int b = 0; b < BATCH_SIZE; b++) {
            float8 x = __half82float8(*(const half8*)(addr(Xps[b], k)));
            for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
              dot8(gaccs[b][r], gws[r], x);
              dot8(uaccs[b][r], uws[r], x);
            }
          }
        }

        if (k + 3 < K) {
          // vectorized
          float4 gws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            gws[r] = __half42float4(load_nontemporal_half4(addr(GWs[r], k)));
          }
          float4 uws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            uws[r] = __half42float4(load_nontemporal_half4(addr(UWs[r], k)));
          }
      
          for (int b = 0; b < BATCH_SIZE; b++) {
            float4 x = __half42float4(*(const half4*)(addr(Xps[b], k)));
            for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
              dot4(gaccs[b][r], gws[r], x);
              dot4(uaccs[b][r], uws[r], x);
            }
          }

          k += 4;
        }

        if (k + 1 < K) {
          // vectorized
          float2 gws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            gws[r] = __half22float2(load_nontemporal_half2(addr(GWs[r], k)));
          }
          float2 uws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            uws[r] = __half22float2(load_nontemporal_half2(addr(UWs[r], k)));
          }
      
          for (int b = 0; b < BATCH_SIZE; b++) {
            float2 x = __half22float2(*(const half2*)(addr(Xps[b], k)));
            for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
              dot2(gaccs[b][r], gws[r], x);
              dot2(uaccs[b][r], uws[r], x);
            }
          }

          k += 2;
        }

        if (k < K) {
          // remainder
          float gws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            gws[r] = __half2float(*addr(GWs[r], k));
          }

          float uws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            uws[r] = __half2float(*addr(UWs[r], k));
          }

          for (int b = 0; b < BATCH_SIZE; b++) {
            float x = __half2float(*addr(Xps[b], k));
            for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
              gaccs[b][r] += gws[r] * x;
              uaccs[b][r] += uws[r] * x;
            }
          }
        }
      }

      // warp reduce
      for (int b = 0; b < BATCH_SIZE; b++) {
        for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
          gaccs[b][r] = warpReduce(gaccs[b][r]);
          uaccs[b][r] = warpReduce(uaccs[b][r]);
        }
      }

      // reduce accross warps
      if (laneId == 0) {
        for (int b = 0; b < BATCH_SIZE; b++) {
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            atomicAdd(&shared_gaccs[b][r], gaccs[b][r]);
            atomicAdd(&shared_uaccs[b][r], uaccs[b][r]);
          }
        }
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
        float gaccs[BATCH_SIZE];
        float uaccs[BATCH_SIZE];
        for (int b = 0; b < BATCH_SIZE; b++) {
          gaccs[b] = 0.f;
          uaccs[b] = 0.f;
        }

        const half* Xps[BATCH_SIZE];
        for (int b = 0; b < BATCH_SIZE; b++) {
          Xps[b] = &X[b * K];
        }

        for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
          float gw = __half2float(GW_[k]);
          float uw = __half2float(UW_[k]);

          for (int b = 0; b < BATCH_SIZE; b++) {
            float x = __half2float(*addr(Xps[b], k));
            gaccs[b] += gw * x;
            uaccs[b] += uw * x;
          }
        }

        // warp reduce
        for (int b = 0; b < BATCH_SIZE; b++) {
          gaccs[b] = warpReduce(gaccs[b]);
          uaccs[b] = warpReduce(uaccs[b]);
        }

        // reduce accross warps
        if (laneId == 0) {
          for (int b = 0; b < BATCH_SIZE; b++) {
            atomicAdd(&shared_gaccs[b][i], gaccs[b]);
            atomicAdd(&shared_uaccs[b][i], uaccs[b]);
          }
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
      for (int b = 0; b < BATCH_SIZE; b++) {
        float gacc = shared_gaccs[b][threadIdx.x]; // read the fully reduced value
        float uacc = shared_uaccs[b][threadIdx.x]; // read the fully reduced value
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
        Y[(b * N) + current_row] = __float2half(acc);
      }
    }
  }
}

template<int BATCH_SIZE, int FUSED_ROWS_PER_BLOCK>
__global__ void muillm_gateupmlp_gemv_fp16_kernel(
    const half* __restrict__ GW, // weight matrix - size N x K
    const half* __restrict__ UW, // weight matrix - size N x K
    const half* __restrict__ X, // input = size B x K
    half* __restrict__ Y, // output - size B x N
    unsigned N,
    unsigned K,
    MuiGateUpMLPActivation activation
) {
  if (warpSize == 32) {
    constexpr int THREADS_PER_BLOCK = 32 * 4;
    muillm_gateupmlp_gemv_fp16_func<THREADS_PER_BLOCK, BATCH_SIZE, FUSED_ROWS_PER_BLOCK>(
        GW, UW, X, Y, N, K, activation);
  } else if (warpSize == 64) {
    constexpr int THREADS_PER_BLOCK = 64 * 4;
    muillm_gateupmlp_gemv_fp16_func<THREADS_PER_BLOCK, BATCH_SIZE, FUSED_ROWS_PER_BLOCK>(
        GW, UW, X, Y, N, K, activation);
  } else {
    // unsupported warp size
  }
}

template<int THREADS_PER_BLOCK, int BATCH_SIZE, int FUSED_ROWS_PER_BLOCK>
__device__ void muillm_gateupmlp_gemv_norm_inputs_fp16_func(
    const half* __restrict__ NW, // input normalization weights matrix - size K
    const half* __restrict__ GW, // weight matrix - size N x K
    const half* __restrict__ UW, // weight matrix - size N x K
    const half* __restrict__ X, // input = size B x K
    half* __restrict__ Y, // output - size B x N
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

  float var_xs[BATCH_SIZE];
  for (int b = 0; b < BATCH_SIZE; b++) {
    var_xs[b] = 0.f;
  }

  __shared__ float shared_gaccs[BATCH_SIZE][FUSED_ROWS_PER_BLOCK];
  __shared__ float shared_uaccs[BATCH_SIZE][FUSED_ROWS_PER_BLOCK];
  __shared__ float shared_var_x[BATCH_SIZE];

  // initialize the shared memory
  if (threadIdx.x < FUSED_ROWS_PER_BLOCK) {
    for (int b = 0; b < BATCH_SIZE; b++) {
      shared_gaccs[b][threadIdx.x] = 0.f;
      shared_uaccs[b][threadIdx.x] = 0.f;
    }
  }
  if (threadIdx.x == 0) {
    for (int b = 0; b < BATCH_SIZE; b++) {
      shared_var_x[b] = epsilon;
    }
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
      const half* GWs[FUSED_ROWS_PER_BLOCK];
      for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
        GWs[r] = &GW[(current_row + r) * K];
      }

      const half* UWs[FUSED_ROWS_PER_BLOCK];
      for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
        UWs[r] = &UW[(current_row + r) * K];
      }

      const half* Xps[BATCH_SIZE];
      for (int b = 0; b < BATCH_SIZE; b++) {
        Xps[b] = &X[b * K];
      }
  
      float gaccs[BATCH_SIZE][FUSED_ROWS_PER_BLOCK];
      float uaccs[BATCH_SIZE][FUSED_ROWS_PER_BLOCK];
      for (int b = 0; b < BATCH_SIZE; b++) {
        for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
          gaccs[b][r] = 0.f;
          uaccs[b][r] = 0.f;
        }
      }

      // do the dot product
      {
        unsigned k; // should be 2 * tidx ?
        //*
        for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
          // vectorized
          float8 gws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            gws[r] = __half82float8(load_nontemporal_half8(addr(GWs[r], k)));
          }
          float8 uws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            uws[r] = __half82float8(load_nontemporal_half8(addr(UWs[r], k)));
          }

          float8 nw = __half82float8(*(const half8*)(addr(NW, k))) + weights_offset;

          for (int b = 0; b < BATCH_SIZE; b++) {
            float8 x = __half82float8(*(const half8*)(addr(Xps[b], k)));

            // accumulate for the variance
            dot8(var_xs[b], x, x);

            // multiply with normalization weights
            x.x = x.x * nw.x;
            x.y = x.y * nw.y;
            x.z = x.z * nw.z;
            x.w = x.w * nw.w;
            x.a = x.a * nw.a;
            x.b = x.b * nw.b;
            x.c = x.c * nw.c;
            x.d = x.d * nw.d;

            for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
              dot8(gaccs[b][r], gws[r], x);
              dot8(uaccs[b][r], uws[r], x);
            }
          }
        }

        if (k + 3 < K) {
          // vectorized
          float4 gws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            gws[r] = __half42float4(load_nontemporal_half4(addr(GWs[r], k)));
          }
          float4 uws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            uws[r] = __half42float4(load_nontemporal_half4(addr(UWs[r], k)));
          }

          float4 nw = __half42float4(*(const half4*)(addr(NW, k))) + weights_offset;

          for (int b = 0; b < BATCH_SIZE; b++) {
            float4 x = __half42float4(*(const half4*)(addr(Xps[b], k)));

            // accumulate for the variance
            dot4(var_xs[b], x, x);

            // multiply with normalization weights
            x.x = x.x * nw.x;
            x.y = x.y * nw.y;
            x.z = x.z * nw.z;
            x.w = x.w * nw.w;

            for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
              dot4(gaccs[b][r], gws[r], x);
              dot4(uaccs[b][r], uws[r], x);
            }
          }

          k += 4;
        }

        if (k + 1 < K) {
          // vectorized
          float2 gws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            gws[r] = __half22float2(load_nontemporal_half2(addr(GWs[r], k)));
          }
          float2 uws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            uws[r] = __half22float2(load_nontemporal_half2(addr(UWs[r], k)));
          }
  
          float2 nw = __half22float2(*(const half2*)(addr(NW, k))) + weights_offset;

          for (int b = 0; b < BATCH_SIZE; b++) {
            float2 x = __half22float2(*(const half2*)(addr(Xps[b], k)));

            // accumulate for the variance
            dot2(var_xs[b], x, x);

            // multiply with normalization weights
            x.x = x.x * nw.x;
            x.y = x.y * nw.y;

            for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
              dot2(gaccs[b][r], gws[r], x);
              dot2(uaccs[b][r], uws[r], x);
            }
          }

          k += 2;
        }

        if (k < K) {
          // remainder
          float gws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            gws[r] = __half2float(*addr(GWs[r], k));
          }
          float uws[FUSED_ROWS_PER_BLOCK];
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            uws[r] = __half2float(*addr(UWs[r], k));
          }

          float nw = __half2float(*addr(NW,k)) + weights_offset;

          for (int b = 0; b < BATCH_SIZE; b++) {
            float x = __half2float(*addr(Xps[b], k));

            // accumulate for the variance
            var_xs[b] += x * x;

            // multiply with normalization weights
            x *= nw;

            for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
              gaccs[b][r] += gws[r] * x;
              uaccs[b][r] += uws[r] * x;
            }
          }
        }
      }

      // warp reduce
      for (int b = 0; b < BATCH_SIZE; b++) {
        var_xs[b] = warpReduce(var_xs[b]);
        for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
          gaccs[b][r] = warpReduce(gaccs[b][r]);
          uaccs[b][r] = warpReduce(uaccs[b][r]);
        }
      }

      // reduce accross warps
      if (laneId == 0) {
        for (int b = 0; b < BATCH_SIZE; b++) {
          atomicAdd(&shared_var_x[b], var_xs[b]);
          for (int r = 0; r < FUSED_ROWS_PER_BLOCK; r++) {
            atomicAdd(&shared_gaccs[b][r], gaccs[b][r]);
            atomicAdd(&shared_uaccs[b][r], uaccs[b][r]);
          }
        }
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

        const half* Xps[BATCH_SIZE];
        for (int b = 0; b < BATCH_SIZE; b++) {
          Xps[b] = &X[b * K];
        }

        float gaccs[BATCH_SIZE];
        float uaccs[BATCH_SIZE];
        for (int b = 0; b < BATCH_SIZE; b++) {
          gaccs[b] = 0.f;
          uaccs[b] = 0.f;
        }

        // do the dot product
        if (i == 0) {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float gw = __half2float(GW_[k]);
            float uw = __half2float(UW_[k]);
            float nw = __half2float(NW[k]) + weights_offset;

            for (int b = 0; b < BATCH_SIZE; b++) {
              float x = __half2float(*addr(Xps[b], k));

              // accumuate the variance
              var_xs[b] += x * x;

              // multiply with normalization weights
              x *= nw;

              gaccs[b] += gw * x;
              uaccs[b] += uw * x;
            }
          }
        } else {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float gw = __half2float(GW_[k]);
            float uw = __half2float(UW_[k]);
            float nw = __half2float(NW[k]) + weights_offset;

            for (int b = 0; b < BATCH_SIZE; b++) {
              float x = __half2float(*addr(Xps[b], k));

              // don't accumulate the variance (we already have done it with i == 0)

              // multiply with normalization weights
              x *= nw;

              gaccs[b] += gw * x;
              uaccs[b] += uw * x;
            }
          }
        }

        // warp reduce
        if (i == 0) {
          for (int b = 0; b < BATCH_SIZE; b++) {
            var_xs[b] = warpReduce(var_xs[b]);
          }
        }
        for (int b = 0; b < BATCH_SIZE; b++) {
          gaccs[b] = warpReduce(gaccs[b]);
          uaccs[b] = warpReduce(uaccs[b]);
        }

        // reduce accross warps
        if (laneId == 0) {
          if (i == 0) {
            for (int b = 0; b < BATCH_SIZE; b++) {
              atomicAdd(&shared_var_x[b], var_xs[b]);
            }
          }
          for (int b = 0; b < BATCH_SIZE; b++) {
            atomicAdd(&shared_gaccs[b][i], gaccs[b]);
            atomicAdd(&shared_uaccs[b][i], uaccs[b]);
          }
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
      for (int b = 0; b < BATCH_SIZE; b++) {
        float rsqrt_var = rsqrtf(shared_var_x[b] * scale);
        float gacc = shared_gaccs[b][threadIdx.x] * rsqrt_var; // read the fully reduced value and scale
        float uacc = shared_uaccs[b][threadIdx.x] * rsqrt_var; // read the fully reduced value and scale
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
        Y[(b * N) + current_row] = __float2half(acc);
      }
    }
  }
}

template<int BATCH_SIZE, int FUSED_ROWS_PER_BLOCK>
__global__ void muillm_gateupmlp_gemv_norm_inputs_fp16_kernel(
    const half* __restrict__ NW, // input normalization weights matrix - size K
    const half* __restrict__ GW, // weight matrix - size N x K
    const half* __restrict__ UW, // weight matrix - size N x K
    const half* __restrict__ X, // input = size B x K
    half* __restrict__ Y, // output - size B x N
    unsigned N,
    unsigned K,
    float epsilon,
    float weights_offset,
    float scale,
    MuiGateUpMLPActivation activation
) {
  if (warpSize == 32) {
    constexpr int THREADS_PER_BLOCK = 32 * 4;
    muillm_gateupmlp_gemv_norm_inputs_fp16_func<THREADS_PER_BLOCK, BATCH_SIZE, FUSED_ROWS_PER_BLOCK>(
        NW, GW, UW, X, Y, N, K, epsilon, weights_offset, scale, activation);
  } else if (warpSize == 64) {
    constexpr int THREADS_PER_BLOCK = 64 * 4;
    muillm_gateupmlp_gemv_norm_inputs_fp16_func<THREADS_PER_BLOCK, BATCH_SIZE, FUSED_ROWS_PER_BLOCK>(
        NW, GW, UW, X, Y, N, K, epsilon, weights_offset, scale, activation);
  } else {
    // unsupported warp size
  }
}

template<int BATCH_SIZE_OFFSET, int FUSED_ROWS_PER_BLOCK>
static inline void call_gateup_gemv_norm_kernel(
    int num_blocks,
    int threads_per_blocks,
    hipStream_t stream,
    const half* norm_weights,
    const half* gate_weights,
    const half* up_weights,
    const half* x,
    half* y,
    unsigned B,
    unsigned N,
    unsigned K,
    float epsilon,
    float norm_weights_offset,
    float scale,
    MuiGateUpMLPActivation activation
) {
  if (B == (BATCH_SIZE_OFFSET + 1)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 1;
    muillm_gateupmlp_gemv_norm_inputs_fp16_kernel<BATCH_SIZE, FUSED_ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
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
  } else if (B == (BATCH_SIZE_OFFSET + 2)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 2;
    muillm_gateupmlp_gemv_norm_inputs_fp16_kernel<BATCH_SIZE, FUSED_ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
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
  } else if (B == (BATCH_SIZE_OFFSET + 3)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 3;
    muillm_gateupmlp_gemv_norm_inputs_fp16_kernel<BATCH_SIZE, FUSED_ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
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
  } else if (B == (BATCH_SIZE_OFFSET + 4)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 4;
    muillm_gateupmlp_gemv_norm_inputs_fp16_kernel<BATCH_SIZE, FUSED_ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
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
  } else {
    throw std::runtime_error("Unsupported batch size for muillm_gateupmlp_gemv_norm_inputs_fp16_kernel");
  }
}

template<int BATCH_SIZE_OFFSET, int FUSED_ROWS_PER_BLOCK>
static inline void call_gateup_gemv_kernel(
    int num_blocks,
    int threads_per_blocks,
    hipStream_t stream,
    const half* gate_weights,
    const half* up_weights,
    const half* x,
    half* y,
    unsigned B,
    unsigned N,
    unsigned K,
    MuiGateUpMLPActivation activation
) {
  if (B == (BATCH_SIZE_OFFSET + 1)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 1;
    muillm_gateupmlp_gemv_fp16_kernel<BATCH_SIZE, FUSED_ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      gate_weights,
      up_weights,
      x,
      y,
      N,
      K,
      activation
    );
  } else if (B == (BATCH_SIZE_OFFSET + 2)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 2;
    muillm_gateupmlp_gemv_fp16_kernel<BATCH_SIZE, FUSED_ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      gate_weights,
      up_weights,
      x,
      y,
      N,
      K,
      activation
    );
  } else if (B == (BATCH_SIZE_OFFSET + 3)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 3;
    muillm_gateupmlp_gemv_fp16_kernel<BATCH_SIZE, FUSED_ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      gate_weights,
      up_weights,
      x,
      y,
      N,
      K,
      activation
    );
  } else if (B == (BATCH_SIZE_OFFSET + 4)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 4;
    muillm_gateupmlp_gemv_fp16_kernel<BATCH_SIZE, FUSED_ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      gate_weights,
      up_weights,
      x,
      y,
      N,
      K,
      activation
    );
  } else {
    throw std::runtime_error("Unsupported batch size for muillm_gateupmlp_gemv_fp16_kernel");
  }
}

template<int FUSED_ROWS_PER_BLOCK>
static inline void call_gateup_gemv_norm_kernel_batch_mux(
    int num_blocks,
    int threads_per_blocks,
    hipStream_t stream,
    const half* norm_weights,
    const half* gate_weights,
    const half* up_weights,
    const half* x,
    half* y,
    unsigned B,
    unsigned N,
    unsigned K,
    float epsilon,
    float norm_weights_offset,
    float scale,
    MuiGateUpMLPActivation activation
) {
  if (B <= 4) {
    constexpr int BATCH_SIZE_OFFSET = 0;
    call_gateup_gemv_norm_kernel<BATCH_SIZE_OFFSET, FUSED_ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      gate_weights,
      up_weights,
      x,
      y,
      B,
      N,
      K,
      epsilon,
      norm_weights_offset,
      scale,
      activation
    );
  } else if (B <= 8) {
    constexpr int BATCH_SIZE_OFFSET = 4;
    call_gateup_gemv_norm_kernel<BATCH_SIZE_OFFSET, FUSED_ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      gate_weights,
      up_weights,
      x,
      y,
      B,
      N,
      K,
      epsilon,
      norm_weights_offset,
      scale,
      activation
    );
  } else if (B <= 12) {
    constexpr int BATCH_SIZE_OFFSET = 8;
    call_gateup_gemv_norm_kernel<BATCH_SIZE_OFFSET, FUSED_ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      gate_weights,
      up_weights,
      x,
      y,
      B,
      N,
      K,
      epsilon,
      norm_weights_offset,
      scale,
      activation
    );
  } else if (B <= 16) {
    constexpr int BATCH_SIZE_OFFSET = 12;
    call_gateup_gemv_norm_kernel<BATCH_SIZE_OFFSET, FUSED_ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      gate_weights,
      up_weights,
      x,
      y,
      B,
      N,
      K,
      epsilon,
      norm_weights_offset,
      scale,
      activation
    );
  } else {
    throw std::runtime_error("Unsupported batch size for muillm_gateupmlp_gemv_norm_inputs_fp16_kernel");
  }
}

template<int FUSED_ROWS_PER_BLOCK>
static inline void call_gateup_gemv_kernel_batch_mux(
    int num_blocks,
    int threads_per_blocks,
    hipStream_t stream,
    const half* gate_weights,
    const half* up_weights,
    const half* x,
    half* y,
    unsigned B,
    unsigned N,
    unsigned K,
    MuiGateUpMLPActivation activation
) {
  if (B <= 4) {
    constexpr int BATCH_SIZE_OFFSET = 0;
    call_gateup_gemv_kernel<BATCH_SIZE_OFFSET, FUSED_ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      gate_weights,
      up_weights,
      x,
      y,
      B,
      N,
      K,
      activation
    );
  } else if (B <= 8) {
    constexpr int BATCH_SIZE_OFFSET = 4;
    call_gateup_gemv_kernel<BATCH_SIZE_OFFSET, FUSED_ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      gate_weights,
      up_weights,
      x,
      y,
      B,
      N,
      K,
      activation
    );
  } else if (B <= 12) {
    constexpr int BATCH_SIZE_OFFSET = 8;
    call_gateup_gemv_kernel<BATCH_SIZE_OFFSET, FUSED_ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      gate_weights,
      up_weights,
      x,
      y,
      B,
      N,
      K,
      activation
    );
  } else if (B <= 16) {
    constexpr int BATCH_SIZE_OFFSET = 12;
    call_gateup_gemv_kernel<BATCH_SIZE_OFFSET, FUSED_ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      gate_weights,
      up_weights,
      x,
      y,
      B,
      N,
      K,
      activation
    );
  } else {
    throw std::runtime_error("Unsupported batch size for muillm_gateupmlp_gemv_fp16_kernel");
  }
}

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
) {

  bool normalize = norm_weights != nullptr;

  constexpr int FUSED_ROWS_PER_BLOCK = 2;

  const int num_blocks = DIV_ROUND_UP(N, FUSED_ROWS_PER_BLOCK);
  int threads_per_blocks = 4 * warp_size;


  // try to occupy enough to saturate memory bandwidth
  /*
  while ((num_blocks * threads_per_blocks < 8 * simd_lanes) && threads_per_blocks < 256) {
    threads_per_blocks *= 2;
  }
  */

  if (normalize) {
    float scale = 1.f / K;
    call_gateup_gemv_norm_kernel_batch_mux<FUSED_ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      gate_weights,
      up_weights,
      x,
      y,
      B,
      N,
      K,
      epsilon,
      norm_weights_offset,
      scale,
      activation
    );
  } else {
    call_gateup_gemv_kernel_batch_mux<FUSED_ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      gate_weights,
      up_weights,
      x,
      y,
      B,
      N,
      K,
      activation
    );
  }
}