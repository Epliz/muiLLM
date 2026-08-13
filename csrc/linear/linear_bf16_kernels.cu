#include "activation.h"

#include <hip/hip_bf16.h>

//
// actual module
//

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

struct __align__(8) bfloat164 {
  __hip_bfloat16 x;
  __hip_bfloat16 y;
  __hip_bfloat16 z;
  __hip_bfloat16 w;
};

struct __align__(8) bfloat168 {
  __hip_bfloat16 x;
  __hip_bfloat16 y;
  __hip_bfloat16 z;
  __hip_bfloat16 w;
  __hip_bfloat16 a;
  __hip_bfloat16 b;
  __hip_bfloat16 c;
  __hip_bfloat16 d;
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

static inline float4 __device__ __bfloat1642float4(const bfloat164& v) {
  float4 f;
  f.x = __bfloat162float(v.x);
  f.y = __bfloat162float(v.y);
  f.z = __bfloat162float(v.z);
  f.w = __bfloat162float(v.w);

  return f;
}

static inline float8 __device__ __bfloat1682float8(const bfloat168& v) {
  float8 f;
  f.x = __bfloat162float(v.x);
  f.y = __bfloat162float(v.y);
  f.z = __bfloat162float(v.z);
  f.w = __bfloat162float(v.w);
  f.a = __bfloat162float(v.a);
  f.b = __bfloat162float(v.b);
  f.c = __bfloat162float(v.c);
  f.d = __bfloat162float(v.d);

  return f;
}

__device__ __hip_bfloat162 load_nontemporal_bfloat162(const __hip_bfloat16* p) {
  float _v = __builtin_nontemporal_load((const float*)p);
  return *((__hip_bfloat162*)&_v);
}

__device__ bfloat164 load_nontemporal_bfloat164(const __hip_bfloat16* p) {
  float _v0 = __builtin_nontemporal_load(((const float*)p));
  float _v1 = __builtin_nontemporal_load(((const float*)p) + 1);

  __hip_bfloat162 _hv0 = *((__hip_bfloat162*)&_v0);
  __hip_bfloat162 _hv1 = *((__hip_bfloat162*)&_v1);

  bfloat164 v;
  v.x = _hv0.x;
  v.y = _hv0.y;
  v.z = _hv1.x;
  v.w = _hv1.y;

  return v;
}

__device__ bfloat168 load_nontemporal_bfloat168(const __hip_bfloat16* p) {
  float _v0 = __builtin_nontemporal_load(((const float*)p));
  float _v1 = __builtin_nontemporal_load(((const float*)p) + 1);
  float _v2 = __builtin_nontemporal_load(((const float*)p) + 2);
  float _v3 = __builtin_nontemporal_load(((const float*)p) + 3);

  __hip_bfloat162 _hv0 = *((__hip_bfloat162*)&_v0);
  __hip_bfloat162 _hv1 = *((__hip_bfloat162*)&_v1);
  __hip_bfloat162 _hv2 = *((__hip_bfloat162*)&_v2);
  __hip_bfloat162 _hv3 = *((__hip_bfloat162*)&_v3);

  bfloat168 v;
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

template<int THREADS_PER_BLOCK, int BATCH_SIZE, int ROWS_PER_BLOCK>
__device__ void muillm_gemv_bf16_func(
    const __hip_bfloat16* __restrict__ W, // weight matrix - size N x K
    const __hip_bfloat16* __restrict__ X, // input = size B x K
    mui_activation activation, // activation function 
    const __hip_bfloat16* __restrict__ AB, // optional additive bias - size N
    const __hip_bfloat16* __restrict__ MRB, // optional multiplicative residual bias - size BxN (applied before additive residual)
    const __hip_bfloat16* __restrict__ RB, // optional residual - size B x N
    __hip_bfloat16* __restrict__ Y, // output - size B x N
    unsigned N,
    unsigned K,
    unsigned xK // stride of the input X (in case it is not contiguous)
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  // can process ROWS_PER_BLOCK rows
  // shared state to do the reductions

  // TODO: avoid bank conflicts by having per warp shared memory
  __shared__ float shared_accs[BATCH_SIZE][ROWS_PER_BLOCK];

  // initialize the shared memory
  if (threadIdx.x < ROWS_PER_BLOCK) {
    for (int b = 0; b < BATCH_SIZE; b++) {
      shared_accs[b][threadIdx.x] = 0.f;
    }
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  {
    int current_row = blockIdx.x * ROWS_PER_BLOCK + 0;
    if (current_row + 3 < N) {

      // compute the t-th element of Y. by doing the dot product with the
      // t-th row of W
      const __hip_bfloat16* Wps[ROWS_PER_BLOCK];
      float accs[BATCH_SIZE][ROWS_PER_BLOCK];

      for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        Wps[r] = &W[(current_row + r) * K];
        for (int b = 0; b < BATCH_SIZE; b++) {
          accs[b][r] = 0.f;
        }
      }

      const __hip_bfloat16* Xps[BATCH_SIZE];
      for (int b = 0; b < BATCH_SIZE; b++) {
        Xps[b] = &X[b * xK];
      }

      // do the dot product
      {
        unsigned k;
        //*
        for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
          // vectorized
          float8 ws[ROWS_PER_BLOCK];
          for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            ws[r] = __bfloat1682float8(load_nontemporal_bfloat168(addr(Wps[r], k)));
          }
          
          for (int b = 0; b < BATCH_SIZE; b++) {
            float8 xs = __bfloat1682float8(*(const bfloat168*)(addr(Xps[b], k)));

            for (int r = 0; r < ROWS_PER_BLOCK; r++) {
              dot8(accs[b][r], ws[r], xs);
            }
          }
        }
        if (k + 3 < K) {
          // vectorized
          float4 ws[ROWS_PER_BLOCK];
          for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            ws[r] = __bfloat1642float4(load_nontemporal_bfloat164(addr(Wps[r], k)));
          }

          for (int b = 0; b < BATCH_SIZE; b++) {
            float4 xs = __bfloat1642float4(*(const bfloat164*)(addr(Xps[b], k)));
            for (int r = 0; r < ROWS_PER_BLOCK; r++) {
              dot4(accs[b][r], ws[r], xs);
            }
          }

          k += 4;
        }
        if (k + 1 < K) {
          // remainder
          float2 ws[ROWS_PER_BLOCK];
          for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            ws[r] = __bfloat1622float2(load_nontemporal_bfloat162(addr(Wps[r], k)));
          }

          for (int b = 0; b < BATCH_SIZE; b++) {
            float2 xs = __bfloat1622float2(*(const __hip_bfloat162*)(addr(Xps[b], k)));

            for (int r = 0; r < ROWS_PER_BLOCK; r++) {
              dot2(accs[b][r], ws[r], xs);
            }
          }

          k+= 2;
        }
        if (k < K) {
          // remainder
          float ws[ROWS_PER_BLOCK];
          for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            ws[r] = __bfloat162float(*addr(Wps[r], k));
          }

          for (int b = 0; b < BATCH_SIZE; b++) {
            float xs = __bfloat162float(*addr(Xps[b], k));

            for (int r = 0; r < ROWS_PER_BLOCK; r++) {
              accs[b][r] += ws[r] * xs;
            }
          }
        }
      }

      // warp reduce
      for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
          accs[b][r] = warpReduce(accs[b][r]);
        }
      }

      // reduce accross warps
      if (laneId == 0) {
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
          for (int b = 0; b < BATCH_SIZE; b++) {
            atomicAdd(&shared_accs[b][r], accs[b][r]);
          }
        }
      }
    } else {
      for (int i = 0; i < ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const __hip_bfloat16* W_ = &W[current_row * K];
      
        // do the dot product
        float accs[BATCH_SIZE];
        for (int b = 0; b < BATCH_SIZE; b++) {
          accs[b] = 0.f;
        }
        {
          const __hip_bfloat16* Xps[BATCH_SIZE];
          for (int b = 0; b < BATCH_SIZE; b++) {
            Xps[b] = &X[b * xK];
          }

          int k = threadIdx.x  * 2;
          for (; k + 1 < K; k += THREADS_PER_BLOCK * 2) {
            float2 w = __bfloat1622float2(*(const __hip_bfloat162*)&W_[k]);

            for (int b = 0; b < BATCH_SIZE; b++) {
              float2 x = __bfloat1622float2(*(const __hip_bfloat162*)(addr(Xps[b], k)));
              dot2(accs[b], w, x);
            }
          }
          if (k < K) {
            float w = __bfloat162float(W_[k]);

            for (int b = 0; b < BATCH_SIZE; b++) {
              float x = __bfloat162float(*addr(Xps[b], k));
              accs[b] += w * x;
            }
          }
        }


        // warp reduce
        for (int b = 0; b < BATCH_SIZE; b++) {
          accs[b] = warpReduce(accs[b]);
        }

        // reduce accross warps
        if (laneId == 0) {
          for (int b = 0; b < BATCH_SIZE; b++) {
            atomicAdd(&shared_accs[b][i], accs[b]);
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
    if (threadIdx.x >= ROWS_PER_BLOCK)
      return;

    int current_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x;

    if (current_row < N) {
      for (int b = 0; b < BATCH_SIZE; b++) {
        float acc = shared_accs[b][threadIdx.x]; // read the fully reduced value

        if (activation == mui_activation::Silu) {
          // apply the activation if there is one
          acc = silu(acc);
        } else if (activation == mui_activation::Gelu_Tanh) {
          acc = gelu_tanh(acc);
        }

        if (AB != nullptr) { // apply the additive bias if there is one
          acc += __bfloat162float(AB[current_row]);
        }

        if (MRB != nullptr) { // apply the multipicative residual if there is one
          acc *= __bfloat162float(MRB[b * N +current_row]);
        }
        if (RB != nullptr) { // apply the residual if there is one
          acc += __bfloat162float(RB[b * N + current_row]);
        }
        // write the output value
        Y[(b * N) + current_row] = __float2bfloat16(acc);
      }
    }
  }
}

template<int BATCH_SIZE, int ROWS_PER_BLOCK>
__global__ void muillm_gemv_bf16_kernel(
    const __hip_bfloat16* __restrict__ W, // weight matrix - size N x K
    const __hip_bfloat16* __restrict__ X, // input = size B x K
    mui_activation activation, // activation function 
    const __hip_bfloat16* __restrict__ AB, // optional additive bias - size N
    const __hip_bfloat16* __restrict__ MRB, // optional multiplicative residual bias - size BxN (applied before additive residual)
    const __hip_bfloat16* __restrict__ RB, // optional residual - size B x N
    __hip_bfloat16* __restrict__ Y, // output - size B x N
    unsigned N,
    unsigned K,
    unsigned xK // stride of the input X (in case it is not contiguous)
) {
  if (warpSize == 32) {
    constexpr int THREADS_PER_BLOCK = 4 * 32;
    muillm_gemv_bf16_func<THREADS_PER_BLOCK, BATCH_SIZE, ROWS_PER_BLOCK>(
        W, X, activation, AB, MRB, RB, Y, N, K, xK);
  } else if (warpSize == 64) {
    constexpr int THREADS_PER_BLOCK = 4 * 64;
    muillm_gemv_bf16_func<THREADS_PER_BLOCK, BATCH_SIZE, ROWS_PER_BLOCK>(
        W, X, activation, AB, MRB, RB, Y, N, K, xK);
  }
}

template<int THREADS_PER_BLOCK, int BATCH_SIZE, int ROWS_PER_BLOCK>
__device__ void muillm_gemv_norm_inputs_bf16_func(
    const __hip_bfloat16* __restrict__ NW, // input normalization weights matrix - size K
    const __hip_bfloat16* __restrict__ W, // weight matrix - size N x K
    const __hip_bfloat16* __restrict__ X, // input = size B x K
    mui_activation activation, // activation function
    const __hip_bfloat16* __restrict__ AB, // optional additive bias - size N
    const __hip_bfloat16* __restrict__ MRB, // optional multiplicative residual bias - size BxN (applied before additive residual)
    const __hip_bfloat16* __restrict__ RB, // optional residual - size B x N
    __hip_bfloat16* __restrict__ Y, // output - size B x N
    unsigned N,
    unsigned K,
    unsigned xK, // stride of the input X (in case it is not contiguous)
    float epsilon,
    float weights_offset,
    float scale
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  float var_xs[BATCH_SIZE];
  for (int b = 0; b < BATCH_SIZE; b++) {
    var_xs[b] = 0.f;
  }

  // can process ROWS_PER_BLOCK rows
  // shared state to do the reductions
  __shared__ float shared_accs[BATCH_SIZE][ROWS_PER_BLOCK];
  __shared__ float shared_var_x[BATCH_SIZE];

  // initialize the shared memory
  if (threadIdx.x < ROWS_PER_BLOCK) {
    for (int b = 0; b < BATCH_SIZE; b++) {
      shared_accs[b][threadIdx.x] = 0.f;
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
    int current_row = blockIdx.x * ROWS_PER_BLOCK + 0;
    if (current_row + 3 < N) {

      // compute the t-th element of Y. by doing the dot product with the
      // t-th row of W
      const __hip_bfloat16* Wps[ROWS_PER_BLOCK];
      float accs[BATCH_SIZE][ROWS_PER_BLOCK];

      for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        Wps[r] = &W[(current_row + r) * K];
        for (int b = 0; b < BATCH_SIZE; b++) {
          accs[b][r] = 0.f;
        }
      }

      const __hip_bfloat16* Xps[BATCH_SIZE];
      for (int b = 0; b < BATCH_SIZE; b++) {
        Xps[b] = &X[b * xK];
      }

      // do the dot product
      {
        // need to normalize the inputs
  
        unsigned k; // should be 2 * tidx ?
        //*
        for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
          float8 nw = __bfloat1682float8(*(const bfloat168*)(addr(NW, k))) + weights_offset;

          float8 ws[ROWS_PER_BLOCK];
          for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            ws[r] = __bfloat1682float8(load_nontemporal_bfloat168(addr(Wps[r], k)));
          }

          for (int b = 0; b < BATCH_SIZE; b++) {
            float8 x = __bfloat1682float8(*(const bfloat168*)(addr(Xps[b], k)));
            
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
            
            for (int r = 0; r < ROWS_PER_BLOCK; r++) {
              // vectorized
              dot8(accs[b][r], ws[r], x);
            }
          }
        }
        if (k + 3 < K) {
          float4 nw = __bfloat1642float4(*(const bfloat164*)(addr(NW, k))) + weights_offset;

          float4 ws[ROWS_PER_BLOCK];
          for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            ws[r] = __bfloat1642float4(load_nontemporal_bfloat164(addr(Wps[r], k)));
          }

          for (int b = 0; b < BATCH_SIZE; b++) {
            float4 x = __bfloat1642float4(*(const bfloat164*)(addr(Xps[b], k)));

            // accumulate for the variance
            dot4(var_xs[b], x, x);

            // multiply with normalization weights
            x.x = x.x * nw.x;
            x.y = x.y * nw.y;
            x.z = x.z * nw.z;
            x.w = x.w * nw.w;

            for (int r = 0; r < ROWS_PER_BLOCK; r++) {
              // vectorized
              dot4(accs[b][r], ws[r], x);
            }
          }

          k += 4;
        }
        if (k + 1 < K) {
          float2 nw = __bfloat1622float2(*(const __hip_bfloat162*)(addr(NW, k))) + weights_offset;

          float2 ws[ROWS_PER_BLOCK];
          for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            ws[r] = __bfloat1622float2(load_nontemporal_bfloat162(addr(Wps[r], k)));
          }

          for (int b = 0; b < BATCH_SIZE; b++) {
            // remainder
            float2 x = __bfloat1622float2(*(const __hip_bfloat162*)(addr(Xps[b], k)));

            // accumulate for the variance
            dot2(var_xs[b], x, x);

            // multiply with normalization weights
            x.x = x.x * nw.x;
            x.y = x.y * nw.y;

            for (int r = 0; r < ROWS_PER_BLOCK; r++) {
              dot2(accs[b][r], ws[r], x);
            }
          }

          k += 2;
        }
        if (k < K) {
          float nw = __bfloat162float(*addr(NW,k)) + weights_offset;

          float ws[ROWS_PER_BLOCK];
          for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            ws[r] = __bfloat162float(*addr(Wps[r],k));
          }

          for (int b = 0; b < BATCH_SIZE; b++) {
            float x = __bfloat162float(*addr(Xps[b],k));

            // accumulate for the variance
            var_xs[b] += x * x;

            // multiply with normalization weights
            x *= nw;

            for (int r = 0; r < ROWS_PER_BLOCK; r++) {
              // remainder
              accs[b][r] += ws[r] * x;
            }
          }
        }
      }

      // warp reduce
      for (int b = 0; b < BATCH_SIZE; b++) {
        var_xs[b] = warpReduce(var_xs[b]);
      }

      for (int b = 0; b < BATCH_SIZE; b++) {
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
          accs[b][r] = warpReduce(accs[b][r]);
        }
      }

      // reduce accross warps
      if (laneId == 0) {
        for (int b = 0; b < BATCH_SIZE; b++) {
          atomicAdd(&shared_var_x[b], var_xs[b]);
        }

        for (int b = 0; b < BATCH_SIZE; b++) {
          for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            atomicAdd(&shared_accs[b][r], accs[b][r]);
          }
        }
      }
    } else {
      for (int i = 0; i < ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const __hip_bfloat16* W_ = &W[current_row * K];
      
        // do the dot product
        float accs[BATCH_SIZE];
        for (int b = 0; b < BATCH_SIZE; b++) {
          accs[b] = 0.f;
        }
        const __hip_bfloat16* Xps[BATCH_SIZE];
        for (int b = 0; b < BATCH_SIZE; b++) {
          Xps[b] = &X[b * xK];
        }

        if (i == 0) {
          // accumulate the variance
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float w = __bfloat162float(W_[k]);
            float nw = __bfloat162float(NW[k]) + weights_offset;

            for (int b = 0; b < BATCH_SIZE; b++) {
              float x = __bfloat162float(*addr(Xps[b], k));

              // accumuate the variance
              var_xs[b] += x * x;

              // multiply with normalization weights
              x *= nw;

              accs[b] += w * x;
            }
          }
        } else {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float w = __bfloat162float(W_[k]);
            float nw = __bfloat162float(NW[k]) + weights_offset;

            for (int b = 0; b < BATCH_SIZE; b++) {
              float x = __bfloat162float(*addr(Xps[b], k));

              // don't accumulate the variance (we already have done it with i == 0)

              // multiply with normalization weights
              x *= nw;

              accs[b] += w * x;
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
          accs[b] = warpReduce(accs[b]);
        }

        // reduce accross warps
        if (laneId == 0) {
          if (i == 0) {
            for (int b = 0; b < BATCH_SIZE; b++) {
              atomicAdd(&shared_var_x[b], var_xs[b]);
            }
          }
          for (int b = 0; b < BATCH_SIZE; b++) {
            atomicAdd(&shared_accs[b][i], accs[b]);
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
    if (threadIdx.x >= ROWS_PER_BLOCK)
      return;

    int current_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x;

    if (current_row < N) {
      for (int b = 0; b < BATCH_SIZE; b++) {
        float rsqrt_var = rsqrtf(shared_var_x[b] * scale);
        float acc = shared_accs[b][threadIdx.x] * rsqrt_var; // read the fully reduced value and scale

        if (activation == mui_activation::Silu) {
          // apply the activation if there is one
          acc = silu(acc);
        } else if (activation == mui_activation::Gelu_Tanh) {
          acc = gelu_tanh(acc);
        }

        if (AB != nullptr) { // apply the additive bias if there is one
          acc += __bfloat162float(AB[current_row]);
        }

        if (MRB != nullptr) { // apply the multipicative residual bias if there is one
          acc *= __bfloat162float(MRB[b * N + current_row]);
        }
        if (RB != nullptr) { // apply the residual if there is one
          acc += __bfloat162float(RB[b * N + current_row]);
        }
        // write the output value
        Y[(b * N) + current_row] = __float2bfloat16(acc);
      }
    }
  }
}

template<int BATCH_SIZE, int ROWS_PER_BLOCK>
__global__ void muillm_gemv_norm_inputs_bf16_kernel(
    const __hip_bfloat16* __restrict__ NW, // input normalization weights matrix - size K
    const __hip_bfloat16* __restrict__ W, // weight matrix - size N x K
    const __hip_bfloat16* __restrict__ X, // input = size B x K
    mui_activation activation, // activation function
    const __hip_bfloat16* __restrict__ AB, // optional additive bias - size N
    const __hip_bfloat16* __restrict__ MRB, // optional multiplicative residual bias - size BxN (applied before additive residual)
    const __hip_bfloat16* __restrict__ RB, // optional residual - size B x N
    __hip_bfloat16* __restrict__ Y, // output - size B x N
    unsigned N,
    unsigned K,
    unsigned xK, // stride of the input X (in case it is not contiguous)
    float epsilon,
    float weights_offset,
    float scale
) {
  if (warpSize == 32) {
    constexpr int THREADS_PER_BLOCK = 4 * 32;
    muillm_gemv_norm_inputs_bf16_func<THREADS_PER_BLOCK, BATCH_SIZE, ROWS_PER_BLOCK>(
        NW, W, X, activation, AB, MRB, RB, Y, N, K, xK, epsilon, weights_offset, scale);
  } else if (warpSize == 64) {
    constexpr int THREADS_PER_BLOCK = 4 * 64;
    muillm_gemv_norm_inputs_bf16_func<THREADS_PER_BLOCK, BATCH_SIZE, ROWS_PER_BLOCK>(
        NW, W, X, activation, AB, MRB, RB, Y, N, K, xK, epsilon, weights_offset, scale);
  }
}

template<int BATCH_SIZE_OFFSET, int ROWS_PER_BLOCK>
static inline void call_gemv_norm_kernel(
    int num_blocks,
    int threads_per_blocks,
    hipStream_t stream,
    const __hip_bfloat16* norm_weights,
    const __hip_bfloat16* weights,
    const __hip_bfloat16* x,
    mui_activation activ,
    const __hip_bfloat16* add_bias,
    const __hip_bfloat16* mul_residual,
    const __hip_bfloat16* residual,
    __hip_bfloat16* y,
    unsigned B,
    unsigned N,
    unsigned K,
    unsigned xK, // stride of the input X (in case it is not contiguous)
    float epsilon,
    float norm_weights_offset
) {
  float scale = 1.f / K;

  if (B == (BATCH_SIZE_OFFSET + 1)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 1;
    muillm_gemv_norm_inputs_bf16_kernel<BATCH_SIZE, ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      N,
      K,
      xK,
      epsilon,
      norm_weights_offset,
      scale
    );
  } else if (B == (BATCH_SIZE_OFFSET + 2)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 2;
    muillm_gemv_norm_inputs_bf16_kernel<BATCH_SIZE, ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      N,
      K,
      xK,
      epsilon,
      norm_weights_offset,
      scale
    );
  } else if (B == (BATCH_SIZE_OFFSET + 3)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 3;
    muillm_gemv_norm_inputs_bf16_kernel<BATCH_SIZE, ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      N,
      K,
      xK,
      epsilon,
      norm_weights_offset,
      scale
    );
  } else if (B == (BATCH_SIZE_OFFSET + 4)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 4;
    muillm_gemv_norm_inputs_bf16_kernel<BATCH_SIZE, ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      N,
      K,
      xK,
      epsilon,
      norm_weights_offset,
      scale
    );
  } else {
    throw std::runtime_error("Unsupported batch size for muillm_gemv_norm_inputs_bf16_kernel");
  }
}

template<int BATCH_SIZE_OFFSET, int ROWS_PER_BLOCK>
static inline void call_gemv_kernel(
    int num_blocks,
    int threads_per_blocks,
    hipStream_t stream,
    const __hip_bfloat16* weights,
    const __hip_bfloat16* x,
    mui_activation activ,
    const __hip_bfloat16* add_bias,
    const __hip_bfloat16* mul_residual,
    const __hip_bfloat16* residual,
    __hip_bfloat16* y,
    unsigned B,
    unsigned N,
    unsigned K,
    unsigned xK // stride of the input X (in case it is not contiguous)
) {
  if (B == (BATCH_SIZE_OFFSET + 1)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 1;
    muillm_gemv_bf16_kernel<BATCH_SIZE, ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      N,
      K,
      xK
    );
  } else if (B == (BATCH_SIZE_OFFSET + 2)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 2;
    muillm_gemv_bf16_kernel<BATCH_SIZE, ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      N,
      K,
      xK
    );
  } else if (B == (BATCH_SIZE_OFFSET + 3)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 3;
    muillm_gemv_bf16_kernel<BATCH_SIZE, ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      N,
      K,
      xK
    );
  } else if (B == (BATCH_SIZE_OFFSET + 4)) {
    constexpr int BATCH_SIZE = BATCH_SIZE_OFFSET + 4;
    muillm_gemv_bf16_kernel<BATCH_SIZE, ROWS_PER_BLOCK><<<num_blocks, threads_per_blocks, 0, stream>>>(
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      N,
      K,
      xK
    );
  } else {
    throw std::runtime_error("Unsupported batch size for muillm_gemv_bf16_kernel");
  }
}

template<int ROWS_PER_BLOCK>
static inline void call_gemv_norm_kernel_batch_mux(
    int num_blocks,
    int threads_per_blocks,
    hipStream_t stream,
    const __hip_bfloat16* norm_weights,
    const __hip_bfloat16* weights,
    const __hip_bfloat16* x,
    mui_activation activ,
    const __hip_bfloat16* add_bias,
    const __hip_bfloat16* mul_residual,
    const __hip_bfloat16* residual,
    __hip_bfloat16* y,
    unsigned B,
    unsigned N,
    unsigned K,
    unsigned xK, // stride of the input X (in case it is not contiguous)
    float epsilon,
    float norm_weights_offset
) {
  if (B <= 4) {
    constexpr int BATCH_SIZE_OFFSET = 0;
    call_gemv_norm_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      xK,
      epsilon,
      norm_weights_offset
    );
  } else if (B <= 8) {
    constexpr int BATCH_SIZE_OFFSET = 4;
    call_gemv_norm_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      xK,
      epsilon,
      norm_weights_offset
    );
  } else if (B <= 12) {
    constexpr int BATCH_SIZE_OFFSET = 8;
    call_gemv_norm_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      xK,
      epsilon,
      norm_weights_offset
    );
  } else if (B <= 16) {
    constexpr int BATCH_SIZE_OFFSET = 12;
    call_gemv_norm_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      xK,
      epsilon,
      norm_weights_offset
    );
  } else {
    throw std::runtime_error("Unsupported batch size for muillm_gemv_norm_inputs_bf16_kernel");
  }
}

template<int ROWS_PER_BLOCK>
static inline void call_gemv_kernel_batch_mux(
    int num_blocks,
    int threads_per_blocks,
    hipStream_t stream,
    const __hip_bfloat16* weights,
    const __hip_bfloat16* x,
    mui_activation activ,
    const __hip_bfloat16* add_bias,
    const __hip_bfloat16* mul_residual,
    const __hip_bfloat16* residual,
    __hip_bfloat16* y,
    unsigned B,
    unsigned N,
    unsigned K,
    unsigned xK // stride of the input X (in case it is not contiguous)
) {
  if (B <= 4) {
    constexpr int BATCH_SIZE_OFFSET = 0;
    call_gemv_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      xK
    );
  } else if (B <= 8) {
    constexpr int BATCH_SIZE_OFFSET = 4;
    call_gemv_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      xK
    );
  } else if (B <= 12) {
    constexpr int BATCH_SIZE_OFFSET = 8;
    call_gemv_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      xK
    );
  } else if (B <= 16) {
    constexpr int BATCH_SIZE_OFFSET = 12;
    call_gemv_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      xK
    );
  } else {
    throw std::runtime_error("Unsupported batch size for muillm_gemv_bf16_kernel");
  }
}

template<int ROWS_PER_BLOCK>
static inline void call_gemv_norm_kernel_batch_mux(
    int num_blocks,
    int threads_per_blocks,
    hipStream_t stream,
    const __hip_bfloat16* norm_weights,
    const __hip_bfloat16* weights,
    const __hip_bfloat16* x,
    mui_activation activ,
    const __hip_bfloat16* add_bias,
    const __hip_bfloat16* mul_residual,
    const __hip_bfloat16* residual,
    __hip_bfloat16* y,
    unsigned B,
    unsigned N,
    unsigned K,
    float epsilon,
    float norm_weights_offset
) {
  if (B <= 4) {
    constexpr int BATCH_SIZE_OFFSET = 0;
    call_gemv_norm_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      epsilon,
      norm_weights_offset
    );
  } else if (B <= 8) {
    constexpr int BATCH_SIZE_OFFSET = 4;
    call_gemv_norm_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      epsilon,
      norm_weights_offset
    );
  } else if (B <= 12) {
    constexpr int BATCH_SIZE_OFFSET = 8;
    call_gemv_norm_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      epsilon,
      norm_weights_offset
    );
  } else if (B <= 16) {
    constexpr int BATCH_SIZE_OFFSET = 12;
    call_gemv_norm_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      epsilon,
      norm_weights_offset
    );
  } else {
    throw std::runtime_error("Unsupported batch size for muillm_gemv_norm_inputs_bf16_kernel");
  }
}

template<int ROWS_PER_BLOCK>
static inline void call_gemv_kernel_batch_mux(
    int num_blocks,
    int threads_per_blocks,
    hipStream_t stream,
    const __hip_bfloat16* weights,
    const __hip_bfloat16* x,
    mui_activation activ,
    const __hip_bfloat16* add_bias,
    const __hip_bfloat16* mul_residual,
    const __hip_bfloat16* residual,
    __hip_bfloat16* y,
    unsigned B,
    unsigned N,
    unsigned K
) {
  if (B <= 4) {
    constexpr int BATCH_SIZE_OFFSET = 0;
    call_gemv_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K
    );
  } else if (B <= 8) {
    constexpr int BATCH_SIZE_OFFSET = 4;
    call_gemv_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K
    );
  } else if (B <= 12) {
    constexpr int BATCH_SIZE_OFFSET = 8;
    call_gemv_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K
    );
  } else if (B <= 16) {
    constexpr int BATCH_SIZE_OFFSET = 12;
    call_gemv_kernel<BATCH_SIZE_OFFSET, ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K
    );
  } else {
    throw std::runtime_error("Unsupported batch size for muillm_gemv_bf16_kernel");
  }
}

void muillm_linear_activ_forward_bf16(
  hipStream_t stream,
  unsigned B,
  unsigned N,
  unsigned K,
  unsigned xK, // stride of the input X (in case it is not contiguous)
  const __hip_bfloat16* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const __hip_bfloat16* weights,
  mui_activation activ,
  const __hip_bfloat16* add_bias,
  const __hip_bfloat16* mul_residual,
  const __hip_bfloat16* residual,
  const __hip_bfloat16* x,
  __hip_bfloat16* y,
  int warp_size
) {

  bool normalize = (norm_weights != nullptr);

  constexpr int ROWS_PER_BLOCK = 4;
  constexpr int BATCH_SIZE = 1;

  const int num_blocks = DIV_ROUND_UP(N, ROWS_PER_BLOCK);

  // either 128 on RDNA/CDNA5+ or 256 on CDNA < 5
  int threads_per_blocks = 4 * warp_size;

  // try to occupy enough to saturate memory bandwidth
  /*
  while ((num_blocks * threads_per_blocks < 8 * simd_lanes) && threads_per_blocks < 256) {
    threads_per_blocks *= 2;
  }
  */

  if (normalize) {
    call_gemv_norm_kernel_batch_mux<ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      norm_weights,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      xK,
      epsilon,
      norm_weights_offset
    );
  } else {
    call_gemv_kernel_batch_mux<ROWS_PER_BLOCK>(
      num_blocks,
      threads_per_blocks,
      stream,
      weights,
      x,
      activ,
      add_bias,
      mul_residual,
      residual,
      y,
      B,
      N,
      K,
      xK
    );
  }
}