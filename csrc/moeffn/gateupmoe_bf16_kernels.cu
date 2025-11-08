#include <hip/hip_bf16.h>

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


#define FUSED_ROWS_PER_BLOCK 2

template<int THREADS_PER_BLOCK>
__global__ void muillm_gateupmlpmoe_gemv_bf16_kernel(
    const __hip_bfloat16* __restrict__ GW, // weight matrix - size (num_shared_experts + num_dynamic_experts) x N x K
    const __hip_bfloat16* __restrict__ UW, // weight matrix - size (num_shared_experts + num_dynamic_experts) x N x K
    const __hip_bfloat16* __restrict__ X, // input = size K
    const __hip_bfloat16* __restrict__ router_scores, // size num_routed_experts
    const int64_t* __restrict__ router_indices, // size num_routed_experts
    __hip_bfloat16* __restrict__ Y, // output - size (num_shared_experts + num_routed_experts) x N
    unsigned num_shared_experts,
    unsigned N,
    unsigned K
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  // blockIdx.x is to divide the weight matrices rows
  // blockIdx.y is for each expert to compute
  int exp_tok = blockIdx.y;
  int exp_idx;
  float exp_score;
  if (blockIdx.y < num_shared_experts) {
    // shared expert
    exp_idx = exp_tok;
    exp_score = 1.0f;
  } else {
    // routed expert
    int routed_exp_tok = exp_tok - num_shared_experts;
    exp_idx = ((int) router_indices[routed_exp_tok]) + num_shared_experts;
    exp_score = __bfloat162float(router_scores[routed_exp_tok]);
  }

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
      const __hip_bfloat16* GW0 = &GW[(exp_idx * N + current_row + 0) * K];
      const __hip_bfloat16* GW1 = &GW[(exp_idx * N + current_row + 1) * K];

      float gacc0 = 0.f;
      float gacc1 = 0.f;

      const __hip_bfloat16* UW0 = &UW[(exp_idx * N + current_row + 0) * K];
      const __hip_bfloat16* UW1 = &UW[(exp_idx * N + current_row + 1) * K];

      float uacc0 = 0.f;
      float uacc1 = 0.f;
  
      // do the dot product
      {
        unsigned k;
        //*
        for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
          // vectorized
          float8 x = __bfloat1682float8(*(const bfloat168*)(addr(X, k)));

          float8 gw0 = __bfloat1682float8(load_nontemporal_bfloat168(addr(GW0, k)));
          float8 gw1 = __bfloat1682float8(load_nontemporal_bfloat168(addr(GW1, k)));

          float8 uw0 = __bfloat1682float8(load_nontemporal_bfloat168(addr(UW0, k)));
          float8 uw1 = __bfloat1682float8(load_nontemporal_bfloat168(addr(UW1, k)));
      
          dot8(gacc0, gw0, x);
          dot8(gacc1, gw1, x);
          dot8(uacc0, uw0, x);
          dot8(uacc1, uw1, x);
        }
        if (k + 3 < K) {
          // vectorized
          float4 x = __bfloat1642float4(*(const bfloat164*)(addr(X, k)));

          float4 gw0 = __bfloat1642float4(load_nontemporal_bfloat164(addr(GW0, k)));
          float4 gw1 = __bfloat1642float4(load_nontemporal_bfloat164(addr(GW1, k)));
          float4 uw0 = __bfloat1642float4(load_nontemporal_bfloat164(addr(UW0, k)));
          float4 uw1 = __bfloat1642float4(load_nontemporal_bfloat164(addr(UW1, k)));
      
          dot4(gacc0, gw0, x);
          dot4(gacc1, gw1, x);
          dot4(uacc0, uw0, x);
          dot4(uacc1, uw1, x);

          k += 4;
        }
        if (k + 1 < K) {
          // vectorized
          float2 x = __bfloat1622float2(*(const __hip_bfloat162*)(addr(X, k)));

          float2 gw0 = __bfloat1622float2(load_nontemporal_bfloat162(addr(GW0, k)));
          float2 gw1 = __bfloat1622float2(load_nontemporal_bfloat162(addr(GW1, k)));
          float2 uw0 = __bfloat1622float2(load_nontemporal_bfloat162(addr(UW0, k)));
          float2 uw1 = __bfloat1622float2(load_nontemporal_bfloat162(addr(UW1, k)));
      
          dot2(gacc0, gw0, x);
          dot2(gacc1, gw1, x);
          dot2(uacc0, uw0, x);
          dot2(uacc1, uw1, x);

          k += 2;
        }

        if (k < K) {
          // remainder
          float x = __bfloat162float(*addr(X,k));

          float gw0 = __bfloat162float(*addr(GW0,k));
          float gw1 = __bfloat162float(*addr(GW1,k));
          float uw0 = __bfloat162float(*addr(UW0,k));
          float uw1 = __bfloat162float(*addr(UW1,k));

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
        int current_row = (blockIdx.x * FUSED_ROWS_PER_BLOCK + i);

        if (current_row >= N)
          break;

        const __hip_bfloat16* GW_ = &GW[(exp_idx * N + current_row) * K];
        const __hip_bfloat16* UW_ = &UW[(exp_idx * N + current_row) * K];
      
        // do the dot product
        float gacc = 0.f;
        float uacc = 0.f;

        for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
          float x =  __bfloat162float(X[k]);
          float gw = __bfloat162float(GW_[k]);
          float uw = __bfloat162float(UW_[k]);
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

    int current_row = (blockIdx.x * FUSED_ROWS_PER_BLOCK + threadIdx.x);

    if (current_row < N) {
      float gacc = shared_gaccs[threadIdx.x]; // read the fully reduced value
      float uacc = shared_uaccs[threadIdx.x]; // read the fully reduced value

      // apply the router scores
      gacc = exp_score * gacc;
      uacc = exp_score * uacc;

      float acc= silu(gacc) * uacc;

      // write the output value
      Y[(exp_tok * N) + current_row] = __float2bfloat16(acc);
    }
  }
}

template<int THREADS_PER_BLOCK>
__global__ void muillm_gateupmlpmoe_gemv_norm_inputs_bf16_kernel(
    const __hip_bfloat16* __restrict__ NW, // input normalization weights matrix - size K
    const __hip_bfloat16* __restrict__ GW, // weight matrix - size (num_shared_experts + num_dynamic_experts) x N x K
    const __hip_bfloat16* __restrict__ UW, // weight matrix - size (num_shared_experts + num_dynamic_experts) x N x K
    const __hip_bfloat16* __restrict__ X, // input = size K
    const __hip_bfloat16* __restrict__ router_scores, // size num_routed_experts
    const int64_t* __restrict__ router_indices, // size num_routed_experts
    __hip_bfloat16* __restrict__ Y, // output - size (num_shared_experts + num_routed_experts) x N
    unsigned num_shared_experts,
    unsigned N,
    unsigned K,
    float epsilon,
    float weights_offset,
    float scale
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  // blockIdx.x is to divide the weight matrices rows
  // blockIdx.y is for each expert to compute
  int exp_tok = blockIdx.y;
  int exp_idx;
  float exp_score;
  if (blockIdx.y < num_shared_experts) {
    // shared expert
    exp_idx = exp_tok;
    exp_score = 1.0f;
  } else {
    // routed expert
    int routed_exp_tok = exp_tok - num_shared_experts;
    exp_idx = ((int) router_indices[routed_exp_tok]) + num_shared_experts;
    exp_score = __bfloat162float(router_scores[routed_exp_tok]);
  }

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
      const __hip_bfloat16* GW0 = &GW[(exp_idx * N + current_row + 0) * K];
      const __hip_bfloat16* GW1 = &GW[(exp_idx * N + current_row + 1) * K];

      const __hip_bfloat16* UW0 = &UW[(exp_idx * N + current_row + 0) * K];
      const __hip_bfloat16* UW1 = &UW[(exp_idx * N + current_row + 1) * K];

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
          float8 x = __bfloat1682float8(*(const bfloat168*)(addr(X, k)));
          float8 nw = __bfloat1682float8(*(const bfloat168*)(addr(NW, k))) + weights_offset;

          float8 gw0 = __bfloat1682float8(load_nontemporal_bfloat168(addr(GW0, k)));
          float8 gw1 = __bfloat1682float8(load_nontemporal_bfloat168(addr(GW1, k)));
          float8 uw0 = __bfloat1682float8(load_nontemporal_bfloat168(addr(UW0, k)));
          float8 uw1 = __bfloat1682float8(load_nontemporal_bfloat168(addr(UW1, k)));

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
          float4 x = __bfloat1642float4(*(const bfloat164*)(addr(X, k)));
          float4 nw = __bfloat1642float4(*(const bfloat164*)(addr(NW, k))) + weights_offset;

          float4 gw0 = __bfloat1642float4(load_nontemporal_bfloat164(addr(GW0, k)));
          float4 gw1 = __bfloat1642float4(load_nontemporal_bfloat164(addr(GW1, k)));
          float4 uw0 = __bfloat1642float4(load_nontemporal_bfloat164(addr(UW0, k)));
          float4 uw1 = __bfloat1642float4(load_nontemporal_bfloat164(addr(UW1, k)));

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
          float2 x = __bfloat1622float2(*(const __hip_bfloat162*)(addr(X, k)));
          float2 nw = __bfloat1622float2(*(const __hip_bfloat162*)(addr(NW, k))) + weights_offset;

          float2 gw0 = __bfloat1622float2(load_nontemporal_bfloat162(addr(GW0, k)));
          float2 gw1 = __bfloat1622float2(load_nontemporal_bfloat162(addr(GW1, k)));
          float2 uw0 = __bfloat1622float2(load_nontemporal_bfloat162(addr(UW0, k)));
          float2 uw1 = __bfloat1622float2(load_nontemporal_bfloat162(addr(UW1, k)));
  
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
          float x = __bfloat162float(*addr(X,k));
          float nw = __bfloat162float(*addr(NW,k)) + weights_offset;

          float gw0 = __bfloat162float(*addr(GW0,k));
          float gw1 = __bfloat162float(*addr(GW1,k));
          float uw0 = __bfloat162float(*addr(UW0,k));
          float uw1 = __bfloat162float(*addr(UW1,k));

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

        const __hip_bfloat16* GW_ = &GW[(exp_idx * N + current_row) * K];
        const __hip_bfloat16* UW_ = &UW[(exp_idx * N + current_row) * K];
      
        // do the dot product
        float gacc = 0.f;
        float uacc = 0.f;

        if (i == 0) {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float x =  __bfloat162float(X[k]);
            float nw = __bfloat162float(NW[k]) + weights_offset;

            // accumuate the variance
            var_x += x * x;

            // multiply with normalization weights
            x *= nw;

            float gw = __bfloat162float(GW_[k]);
            float uw = __bfloat162float(UW_[k]);
            gacc += gw * x;
            uacc += uw * x;
          }
        } else {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float x =  __bfloat162float(X[k]);
            float nw = __bfloat162float(NW[k]) + weights_offset;

            // don't accumulate the variance (we already have done it with i == 0)

            // multiply with normalization weights
            x *= nw;

            float gw = __bfloat162float(GW_[k]);
            float uw = __bfloat162float(UW_[k]);

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

    int current_row = (blockIdx.x * FUSED_ROWS_PER_BLOCK + threadIdx.x);

    if (current_row < N) {
      float gacc = shared_gaccs[threadIdx.x] * rsqrt_var; // read the fully reduced value and scale
      float uacc = shared_uaccs[threadIdx.x] * rsqrt_var; // read the fully reduced value and scale

      // apply the router scores
      gacc = exp_score * gacc;
      uacc = exp_score * uacc;

      float acc= silu(gacc) * uacc;

      // write the output value
      Y[(exp_tok * N) + current_row] = __float2bfloat16(acc);
    }
  }
}

void muillm_gateupmlpmoe_forward_bf16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  unsigned num_shared_experts,
  unsigned num_computed_experts,
  const __hip_bfloat16* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const __hip_bfloat16* gate_weights,
  const __hip_bfloat16* up_weights,
  const __hip_bfloat16* x,
  const __hip_bfloat16* router_scores,
  const int64_t* router_indices,
  __hip_bfloat16* y,
  int simd_lanes
) {
  bool normalize = (norm_weights != nullptr);

  const int num_blocks_x = DIV_ROUND_UP(N, FUSED_ROWS_PER_BLOCK);
  const int num_blocks_y = num_computed_experts; // one y block per expert to compute
  const dim3 num_blocks = dim3(num_blocks_x, num_blocks_y);

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
      muillm_gateupmlpmoe_gemv_norm_inputs_bf16_kernel<64><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)norm_weights,
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)y,
        num_shared_experts,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    } else if (threads_per_blocks == 128) {
      muillm_gateupmlpmoe_gemv_norm_inputs_bf16_kernel<128><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)norm_weights,
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)y,
        num_shared_experts,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    } else if (threads_per_blocks == 256) {
      muillm_gateupmlpmoe_gemv_norm_inputs_bf16_kernel<256><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)norm_weights,
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)y,
        num_shared_experts,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    }
  } else {

    if (threads_per_blocks == 64) {
      muillm_gateupmlpmoe_gemv_bf16_kernel<64><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)y,
        num_shared_experts,
        N,
        K
      );
    } else if (threads_per_blocks == 128) {
      muillm_gateupmlpmoe_gemv_bf16_kernel<128><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)y,
        num_shared_experts,
        N,
        K
      );
    } else if (threads_per_blocks == 256) {
      muillm_gateupmlpmoe_gemv_bf16_kernel<256><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)y,
        num_shared_experts,
        N,
        K
      );
    }
  }
}

#define SPLIT_ROWS_PER_BLOCK 4

template<int THREADS_PER_BLOCK>
__global__ void muillm_gateupmlpmoe_gemv_norm_inputs_split_bf16_kernel(
    const __hip_bfloat16* __restrict__ NW, // input normalization weights matrix - size K
    const __hip_bfloat16* __restrict__ GW, // weight matrix - size ((num_shared_experts + num_dynamic_experts)*N) x K
    const __hip_bfloat16* __restrict__ UW, // weight matrix - size ((num_shared_experts + num_dynamic_experts)*N) x K
    const __hip_bfloat16* __restrict__ X, // input = size K
    const __hip_bfloat16* __restrict__ router_scores, // size num_routed_experts
    const int64_t* __restrict__ router_indices, // size num_routed_experts
    __hip_bfloat16* __restrict__ GY, // output - size (num_shared_experts + num_routed_experts) x N
    __hip_bfloat16* __restrict__ UY, // output - size (num_shared_experts + num_routed_experts) x N
    unsigned num_shared_experts,
    unsigned N,
    unsigned K,
    float epsilon,
    float weights_offset,
    float scale
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  // blockIdx.x is to divide the weight matrices rows
  // blockIdx.y is for each expert to compute
  int exp_tok = blockIdx.y;
  int exp_idx;
  float exp_score;
  if (blockIdx.y < num_shared_experts) {
    // shared expert
    exp_idx = exp_tok;
    exp_score = 1.0f;
  } else {
    // routed expert
    int routed_exp_tok = exp_tok - num_shared_experts;
    exp_idx = ((int) router_indices[routed_exp_tok]) + num_shared_experts;
    exp_score = __bfloat162float(router_scores[routed_exp_tok]);
  }

  const __hip_bfloat16* __restrict__ W = blockIdx.z == 0 ? GW : UW; // weight matrix - size N x K
  __hip_bfloat16* __restrict__ Y = blockIdx.z == 0 ? GY : UY; // output - size N

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
      const __hip_bfloat16* W0 = &W[(exp_idx * N + current_row + 0) * K];
      const __hip_bfloat16* W1 = &W[(exp_idx * N + current_row + 1) * K];
      const __hip_bfloat16* W2 = &W[(exp_idx * N + current_row + 2) * K];
      const __hip_bfloat16* W3 = &W[(exp_idx * N + current_row + 3) * K];

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
          float8 x = __bfloat1682float8(*(const bfloat168*)(addr(X, k)));
          float8 nw = __bfloat1682float8(*(const bfloat168*)(addr(NW, k))) + weights_offset;

          float8 w0 = __bfloat1682float8(load_nontemporal_bfloat168(addr(W0, k)));
          float8 w1 = __bfloat1682float8(load_nontemporal_bfloat168(addr(W1, k)));
          float8 w2 = __bfloat1682float8(load_nontemporal_bfloat168(addr(W2, k)));
          float8 w3 = __bfloat1682float8(load_nontemporal_bfloat168(addr(W3, k)));

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
          float4 x = __bfloat1642float4(*(const bfloat164*)(addr(X, k)));
          float4 nw = __bfloat1642float4(*(const bfloat164*)(addr(NW, k))) + weights_offset;

          float4 w0 = __bfloat1642float4(load_nontemporal_bfloat164(addr(W0, k)));
          float4 w1 = __bfloat1642float4(load_nontemporal_bfloat164(addr(W1, k)));
          float4 w2 = __bfloat1642float4(load_nontemporal_bfloat164(addr(W2, k)));
          float4 w3 = __bfloat1642float4(load_nontemporal_bfloat164(addr(W3, k)));

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
          float2 x = __bfloat1622float2(*(const __hip_bfloat162*)(addr(X, k)));
          float2 nw = __bfloat1622float2(*(const __hip_bfloat162*)(addr(NW, k))) + weights_offset;

          float2 w0 = __bfloat1622float2(load_nontemporal_bfloat162(addr(W0, k)));
          float2 w1 = __bfloat1622float2(load_nontemporal_bfloat162(addr(W1, k)));
          float2 w2 = __bfloat1622float2(load_nontemporal_bfloat162(addr(W2, k)));
          float2 w3 = __bfloat1622float2(load_nontemporal_bfloat162(addr(W3, k)));

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
          float x = __bfloat162float(*addr(X,k));
          float nw = __bfloat162float(*addr(NW,k)) + weights_offset;

          float w0 = __bfloat162float(*addr(W0,k));
          float w1 = __bfloat162float(*addr(W1,k));
          float w2 = __bfloat162float(*addr(W2,k));
          float w3 = __bfloat162float(*addr(W3,k));
          
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

        const __hip_bfloat16* W_ = &W[(exp_idx * N + current_row) * K];
      
        // do the dot product
        float acc = 0.f;
        if (i == 0) {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float x =  __bfloat162float(X[k]);
            float nw = __bfloat162float(NW[k]) + weights_offset;

            // accumuate the variance
            var_x += x * x;

            // multiply with normalization weights
            x *= nw;

            float w = __bfloat162float(W_[k]);
            acc += w * x;
          }
        } else {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float x =  __bfloat162float(X[k]);
            float nw = __bfloat162float(NW[k]) + weights_offset;

            // don't accumulate the variance (we already have done it with i == 0)

            // multiply with normalization weights
            x *= nw;

            float w = __bfloat162float(W_[k]);
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

      // apply the router scores
      acc = exp_score * acc;

      // write the output value
      Y[(exp_tok * N) + current_row] = __float2bfloat16(acc);
    }
  }
}

template<int THREADS_PER_BLOCK>
__global__ void muillm_gateupmlpmoe_gemv_split_bf16_kernel(
    const __hip_bfloat16* __restrict__ GW, // weight matrix - size ((num_shared_experts + num_dynamic_experts)*N) x K
    const __hip_bfloat16* __restrict__ UW, // weight matrix - size ((num_shared_experts + num_dynamic_experts)*N) x K
    const __hip_bfloat16* __restrict__ X, // input = size K
    const __hip_bfloat16* __restrict__ router_scores, // size num_routed_experts
    const int64_t* __restrict__ router_indices, // size num_routed_experts
    __hip_bfloat16* __restrict__ GY, // output - size (num_shared_experts + num_routed_experts) x N
    __hip_bfloat16* __restrict__ UY, // output - size (num_shared_experts + num_routed_experts) x N
    unsigned num_shared_experts,
    unsigned N,
    unsigned K
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  // blockIdx.x is to divide the weight matrices rows
  // blockIdx.y is for each expert to compute
  int exp_tok = blockIdx.y;
  int exp_idx;
  float exp_score;
  if (blockIdx.y < num_shared_experts) {
    // shared expert
    exp_idx = exp_tok;
    exp_score = 1.0f;
  } else {
    // routed expert
    int routed_exp_tok = exp_tok - num_shared_experts;
    exp_idx = ((int) router_indices[routed_exp_tok]) + num_shared_experts;
    exp_score = __bfloat162float(router_scores[routed_exp_tok]);
  }

  const __hip_bfloat16* __restrict__ W = blockIdx.z == 0 ? GW : UW;
  __hip_bfloat16* __restrict__ Y = blockIdx.z == 0 ? GY : UY;

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
      const __hip_bfloat16* W0 = &W[(exp_idx * N + current_row + 0) * K];
      const __hip_bfloat16* W1 = &W[(exp_idx * N + current_row + 1) * K];
      const __hip_bfloat16* W2 = &W[(exp_idx * N + current_row + 2) * K];
      const __hip_bfloat16* W3 = &W[(exp_idx * N + current_row + 3) * K];

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
          float8 x = __bfloat1682float8(*(const bfloat168*)(addr(X, k)));

          float8 w0 = __bfloat1682float8(load_nontemporal_bfloat168(addr(W0, k)));
          float8 w1 = __bfloat1682float8(load_nontemporal_bfloat168(addr(W1, k)));
          float8 w2 = __bfloat1682float8(load_nontemporal_bfloat168(addr(W2, k)));
          float8 w3 = __bfloat1682float8(load_nontemporal_bfloat168(addr(W3, k)));

          dot8(acc0, w0, x);
          dot8(acc1, w1, x);
          dot8(acc2, w2, x);
          dot8(acc3, w3, x);
        }
        if (k + 3 < K) {
          // vectorized
          float4 x = __bfloat1642float4(*(const bfloat164*)(addr(X, k)));

          float4 w0 = __bfloat1642float4(load_nontemporal_bfloat164(addr(W0, k)));
          float4 w1 = __bfloat1642float4(load_nontemporal_bfloat164(addr(W1, k)));
          float4 w2 = __bfloat1642float4(load_nontemporal_bfloat164(addr(W2, k)));
          float4 w3 = __bfloat1642float4(load_nontemporal_bfloat164(addr(W3, k)));

          dot4(acc0, w0, x);
          dot4(acc1, w1, x);
          dot4(acc2, w2, x);
          dot4(acc3, w3, x);

          k += 4;
        }
        if (k + 1 < K) {
          // vectorized
          float2 x = __bfloat1622float2(*(const __hip_bfloat162*)(addr(X, k)));

          float2 w0 = __bfloat1622float2(load_nontemporal_bfloat162(addr(W0, k)));
          float2 w1 = __bfloat1622float2(load_nontemporal_bfloat162(addr(W1, k)));
          float2 w2 = __bfloat1622float2(load_nontemporal_bfloat162(addr(W2, k)));
          float2 w3 = __bfloat1622float2(load_nontemporal_bfloat162(addr(W3, k)));

          dot2(acc0, w0, x);
          dot2(acc1, w1, x);
          dot2(acc2, w2, x);
          dot2(acc3, w3, x);

          k += 2;
        }

        if (k < K) {
          // remainder
          float x = __bfloat162float(*addr(X,k));

          float w0 = __bfloat162float(*addr(W0,k));
          float w1 = __bfloat162float(*addr(W1,k));
          float w2 = __bfloat162float(*addr(W2,k));
          float w3 = __bfloat162float(*addr(W3,k));
          
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

        const __hip_bfloat16* W_ = &W[(exp_idx * N + current_row) * K];
      
        // do the dot product
        float acc = 0.f;
        for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
          float x =  __bfloat162float(X[k]);
          float w = __bfloat162float(W_[k]);
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

      // apply the router scores
      acc = exp_score * acc;

      // write the output value
      Y[(exp_tok * N) + current_row] = __float2bfloat16(acc);
    }
  }
}

template<int THREADS_PER_BLOCK>
__global__ void muillm_gateupmlpmoe_combine_bf16_kernel(
    const __hip_bfloat16* __restrict__ GY, // input - size (num_shared_experts + num_routed_experts) * N
    const __hip_bfloat16* __restrict__ UY, // input - size (num_shared_experts + num_routed_experts) * N
    __hip_bfloat16* __restrict__ Y, // output - size (num_shared_experts + num_routed_experts) * N
    unsigned S // S = (num_shared_experts + num_routed_experts) * N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  int current_row = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  if (current_row < S) {
    float g = __bfloat162float(GY[current_row]);
    float u = __bfloat162float(UY[current_row]);
    float y = silu(g) * u;

    // write the output value
    Y[current_row] = __float2bfloat16(y);
  }
}

void muillm_gateupmlpmoe_split_forward_bf16(
  hipStream_t stream,
  unsigned N,
  unsigned K,
  unsigned num_shared_experts,
  unsigned num_computed_experts,
  const __hip_bfloat16* norm_weights,
  float epsilon,
  float norm_weights_offset,
  const __hip_bfloat16* gate_weights,
  const __hip_bfloat16* up_weights,
  const __hip_bfloat16* x,
  const __hip_bfloat16* router_scores,
  const int64_t* router_indices,
  __hip_bfloat16* gy,
  __hip_bfloat16* uy,
  __hip_bfloat16* y,
  int simd_lanes
) {

  bool normalize = norm_weights != nullptr;

  const int num_blocks_x = DIV_ROUND_UP(N, SPLIT_ROWS_PER_BLOCK);
  const int num_blocks_y = num_computed_experts; // one y block per expert to compute
  const int num_blocks_z = 2;
  const dim3 num_blocks = dim3(num_blocks_x, num_blocks_y);
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
      muillm_gateupmlpmoe_gemv_norm_inputs_split_bf16_kernel<64><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)norm_weights,
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)gy,
        (__hip_bfloat16*)uy,
        num_shared_experts,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    } else if (threads_per_blocks == 128) {
      muillm_gateupmlpmoe_gemv_norm_inputs_split_bf16_kernel<128><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)norm_weights,
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)gy,
        (__hip_bfloat16*)uy,
        num_shared_experts,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    } else if (threads_per_blocks == 256) {
      muillm_gateupmlpmoe_gemv_norm_inputs_split_bf16_kernel<256><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)norm_weights,
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)gy,
        (__hip_bfloat16*)uy,
        num_shared_experts,
        N,
        K,
        epsilon,
        norm_weights_offset,
        scale
      );
    }
  } else {

    if (threads_per_blocks == 64) {
      muillm_gateupmlpmoe_gemv_split_bf16_kernel<64><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)gy,
        (__hip_bfloat16*)uy,
        num_shared_experts,
        N,
        K
      );
    } else if (threads_per_blocks == 128) {
      muillm_gateupmlpmoe_gemv_split_bf16_kernel<128><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)gy,
        (__hip_bfloat16*)uy,
        num_shared_experts,
        N,
        K
      );
    } else if (threads_per_blocks == 256) {
      muillm_gateupmlpmoe_gemv_split_bf16_kernel<256><<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const __hip_bfloat16*)gate_weights,
        (const __hip_bfloat16*)up_weights,
        (const __hip_bfloat16*)x,
        (const __hip_bfloat16*)router_scores,
        (const int64_t*)router_indices,
        (__hip_bfloat16*)gy,
        (__hip_bfloat16*)uy,
        num_shared_experts,
        N,
        K
      );
    }
  }

  // do final reduction
  const int S = num_computed_experts * N;
  const int num_blocks_combine = DIV_ROUND_UP(S, threads_per_blocks);
  if (threads_per_blocks == 64) {
    muillm_gateupmlpmoe_combine_bf16_kernel<64><<<num_blocks_combine, threads_per_blocks, 0, stream>>>(
      (const __hip_bfloat16*)gy,
      (const __hip_bfloat16*)uy,
      (__hip_bfloat16*)y,
      S
    );
  } else if (threads_per_blocks == 128) {
    muillm_gateupmlpmoe_combine_bf16_kernel<128><<<num_blocks_combine, threads_per_blocks, 0, stream>>>(
      (const __hip_bfloat16*)gy,
      (const __hip_bfloat16*)uy,
      (__hip_bfloat16*)y,
      S
    );
  } else if (threads_per_blocks == 256) {
    muillm_gateupmlpmoe_combine_bf16_kernel<256><<<num_blocks_combine, threads_per_blocks, 0, stream>>>(
      (const __hip_bfloat16*)gy,
      (const __hip_bfloat16*)uy,
      (__hip_bfloat16*)y,
      S
    );
  }
}