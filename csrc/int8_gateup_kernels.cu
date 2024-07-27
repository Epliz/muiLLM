#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#include "karray.cuh"

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256

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

using khalf2 = karray<half, 2>;
using khalf2x2 = karray<khalf2, 2>;
using kfloat2 = karray<float, 2>;

using khalf4 = karray<half, 4>;
using kfloat4 = karray<float, 4>;

using khalf8 = karray<half, 8>;
using kfloat8 = karray<float, 8>;

using kuint8x2 = karray<uint8_t, 2>;
using kuint8x4 = karray<uint8_t, 4>;
using kuint8x8 = karray<uint8_t, 8>;

using kuint8x2x8 = karray<karray<uint8_t, 2>, 8>;
using kuint8x2x4 = karray<karray<uint8_t, 2>, 4>;
using kuint8x2x2 = karray<karray<uint8_t, 2>, 2>;

using kuint8x8x2 = karray<karray<uint8_t, 8>, 2>;
using kuint8x4x2 = karray<karray<uint8_t, 4>, 2>;

template<typename T, unsigned N, unsigned M>
static inline karray<karray<T, M>, N> __device__ transpose(const karray<karray<T, N>, M>& v) {
  karray<karray<T, M>, N> ret;
  for (unsigned i = 0; i < M; i++) {
    for (unsigned j = 0; j < N; j++) {
      ret[j][i] = v[i][j];
    }
  }
  return ret;
}

struct half2x4 {
    half2 x;
    half2 y;
    half2 z;
    half2 w;
};

template<unsigned N>
static inline karray<float, N> __device__ __halfNtofloatN(const karray<half, N>& v) {
  karray<float, N> r;
  for (unsigned i = 0; i < N; i++) {
    r.data[i] = __half2float(v[i]);
  }
  return r;
}

static inline kfloat2 __device__ __half22float2(const khalf2& v) {
  return __halfNtofloatN<2>(v);
}

static inline kfloat4 __device__ __half42float4(const khalf4& v) {
  return __halfNtofloatN<4>(v);
}

static inline kfloat8 __device__ __half82float8(const khalf8& v) {
  return __halfNtofloatN<8>(v);
}

template<unsigned N>
__device__ void dotN(float& acc, const karray<float, N>& a, const karray<float, N>& b) {
  for (unsigned i = 0; i < N; i++) {
    acc += a[i] * b[i];
  }
}

static inline void __device__ dot2(float& acc, const kfloat2& a, const kfloat2& b) {
 dotN<2>(acc, a, b);
}

static inline void __device__ dot4(float& acc, const kfloat4& a, const kfloat4& b) {
  dotN<4>(acc, a, b);
}

static inline void __device__ dot8(float& acc, const kfloat8& a, const kfloat8& b) {
  dotN<8>(acc, a, b);
}

template <unsigned N>
static inline half __device__ hsumN(const karray<half, N>& x) {
  half r = __float2half(0.f);
  // TODO: specialize to have less long dependency chain?
  for (unsigned i = 0; i < N; i++) {
    r = __hadd(r, x[i]);
  }
  return r;
}


static inline half __device__ hsum4(const khalf4& x) {
  half r0 = __float2half(0.f);
  half r1 = __float2half(0.f);
  // TODO: specialize to have less long dependency chain?
  for (unsigned i = 0; i < 4; i+=2) {
    r0 = __hadd(r0, x[i + 0]);
    r1 = __hadd(r1, x[i + 1]);
  }
  return __hadd(r0, r1);
}

static inline half __device__ hsum8(const khalf8& x) {
  half r0 = __float2half(0.f);
  half r1 = __float2half(0.f);
  // TODO: specialize to have less long dependency chain?
  for (unsigned i = 0; i < 8; i+=2) {
    r0 = __hadd(r0, x[i + 0]);
    r1 = __hadd(r1, x[i + 1]);
  }
  return __hadd(r0, r1);
}

template <typename T>
static inline const T* __device__ addr(const T* p, unsigned index) {
  // helps the AMDGPU compiler understand it can use the sgrp pair + single vgpr addressing mode
  unsigned byte_offset = sizeof(T) * index;
  const uint8_t* p8 = (const uint8_t*)p;
  return (const T*) (p8 + byte_offset);
}


template <typename T>
static inline T* __device__ waddr(T* p, unsigned index) {
  // helps the AMDGPU compiler understand it can use the sgrp pair + single vgpr addressing mode
  unsigned byte_offset = sizeof(T) * index;
  uint8_t* p8 = (uint8_t*)p;
  return (T*) (p8 + byte_offset);
}

static inline float __device__ silu(float x) {
  return x / (1.0f + expf(-x));
}

static inline float __device__ qint8tofloat(uint8_t q, half2 scale_min_val) {
  half scale = scale_min_val.x;
  half min_val = scale_min_val.y;
  return q * __half2float(scale) + __half2float(min_val);
}

template<unsigned N>
static inline karray<float, N> __device__ qint8xNtofloatN(const karray<uint8_t, N>& qs, half2 scale_min_val) {
  float scale = __half2float(scale_min_val.x);
  float min_val = __half2float(scale_min_val.y);
  karray<float, N> r;
  for (unsigned i = 0; i < N; i++) {
    r[i] = qs[i] * scale + min_val;
  }
  return r;
}

static inline kfloat2 __device__ qint8x2tofloat2(const kuint8x2& qs, half2 scale_min_val) {
  return qint8xNtofloatN<2>(qs, scale_min_val);
}

static inline kfloat4 __device__ qint8x4tofloat4(const kuint8x4& qs, half2 scale_min_val) {
  return qint8xNtofloatN<4>(qs, scale_min_val);
}

static inline kfloat8 __device__ qint8x8tofloat8(const kuint8x8& qs, half2 scale_min_val) {
  return qint8xNtofloatN<8>(qs, scale_min_val);
}

static inline void __device__ qdot(float& acc, uint8_t qw, half2 scale_min_val, float x) {
  float w = qint8tofloat(qw, scale_min_val);
  acc += w * x;
}

static inline void __device__ qdot2(float& acc, const kuint8x2& qw, half2 scale_min_val, const kfloat2& x) {
  kfloat2 w = qint8x2tofloat2(qw, scale_min_val);
  dot2(acc, w, x);
}


template<unsigned N>
static inline float __device__ ifdotN(const karray<uint8_t, N>& qw, const karray<float, N>& x) {
  // first op is a mul and not fma, causes conversion to f32 of x
  // TODO: use half precision
  float r = 0.f;
  for (unsigned i = 0; i < N; i++) {
    // TODO: reduce length of dependency chain?
    r += qw[i] * x[i];
  }
  return r;
}

template<unsigned N>
static inline void __device__ ifdotN_scale(float& acc, const karray<uint8_t, N>& qw, half scale, const karray<float, N>& x) {
  acc += __half2float(scale) * ifdotN<N>(qw, x);
}

// TODO: lots of cvt_f32_f16 in the assembly, need to figure it out
template<unsigned N>
static inline void __device__ qdotN(float& acc, const karray<uint8_t, N>& qw, half2 scale_min_val, const karray<half, N>& x) {
  // we compute
  // (1) acc += sum(w_i * x_i)
  // but w_i = qw_i * s_g(i) + min_val_g(i) where g(i) is the index of the group
  // so we are computing acc += sum((qw_i * s_g(i) + min_val_g(i)) * x_i)
  // which is (2) acc += sum(qw_i * x_i) * s_g(i) + min_val_g(i) * sum(x_i)
  // as g(i) is constant within this call
  // the formula (2) reduces FMAs a lot, especially as sum(x_i) can be hoisted across rows
  ifdotN_scale<N>(acc, qw, scale_min_val.x, __halfNtofloatN<N>(x));
  // TODO: use a half sum
  acc += __half2float(scale_min_val.y) * __half2float(hsumN(x));
}

static inline void __device__ qdot4(float& acc, const kuint8x4& qw, half2 scale_min_val, const khalf4& x) {
  ifdotN_scale<4>(acc, qw, scale_min_val.x, __halfNtofloatN<4>(x));
  acc += __half2float(scale_min_val.y) * __half2float(hsum4(x));
}

// TODO: lots of cvt_f32_f16 in the assembly, need to figure it out
static inline void __device__ qdot8(float& acc, const kuint8x8& qw, half2 scale_min_val, const khalf8& x) {
  ifdotN_scale<8>(acc, qw, scale_min_val.x, __halfNtofloatN<8>(x));
  acc += __half2float(scale_min_val.y) * __half2float(hsum8(x));
}

static_assert(sizeof(karray<half2, 2>) == 8);

__global__ void muillm_int8_gateupsilu_gemv_kernel(
    const uint8_t* __restrict__ GUW, // weight matrix - size N x K x 2
    const half* __restrict__ GUQSMV, // quantization scales and minimum values matrix - size N x G x 2 x 2
    const half* __restrict__ X, // input = size K
    half* __restrict__ Y, // output - size N
    unsigned N,
    unsigned K,
    unsigned G,
    unsigned group_size_shift
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  __shared__ float shared_gaccs[ROWS_PER_BLOCK];
  __shared__ float shared_uaccs[ROWS_PER_BLOCK];

  // initialize the shared memory
  if (threadIdx.x < ROWS_PER_BLOCK) {
    shared_gaccs[threadIdx.x] = 0.f;
    shared_uaccs[threadIdx.x] = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  {
    int current_row = blockIdx.x * ROWS_PER_BLOCK + 0;

    // GW
    if (current_row + 3 < N) {
      // compute the t-th element of Y. by doing the dot product with the
      // t-th row of W
      const uint8_t* GUW0 = &GUW[(current_row + 0) * K * 2];
      const uint8_t* GUW1 = &GUW[(current_row + 1) * K * 2];
      const uint8_t* GUW2 = &GUW[(current_row + 2) * K * 2];
      const uint8_t* GUW3 = &GUW[(current_row + 3) * K * 2];

      const half* GUQSMV0 = &GUQSMV[(current_row + 0) * G * 2 * 2];
      const half* GUQSMV1 = &GUQSMV[(current_row + 1) * G * 2 * 2];
      const half* GUQSMV2 = &GUQSMV[(current_row + 2) * G * 2 * 2];
      const half* GUQSMV3 = &GUQSMV[(current_row + 3) * G * 2 * 2];

      float gacc0 = 0.f;
      float gacc1 = 0.f;
      float gacc2 = 0.f;
      float gacc3 = 0.f;

      float uacc0 = 0.f;
      float uacc1 = 0.f;
      float uacc2 = 0.f;
      float uacc3 = 0.f;

      // do the dot product
      {
        unsigned k; // should be 2 * tidx ?
        //*
        for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
          // vectorized
          khalf8 x = khalf8::load(addr(X, k));

          // the quantized group index is the same for all rows
          unsigned qgidx = k >> group_size_shift;

          kuint8x2x8 guw0 = kuint8x2x8::load((const kuint8x2*)addr(GUW0, 2 * k));
          kuint8x8x2 tguw0 = transpose(guw0);
          karray<half2, 2> scales_mins0 = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
          qdot8(gacc0, tguw0[0], (scales_mins0[0]), x);
          qdot8(uacc0, tguw0[1], (scales_mins0[1]), x);

          kuint8x2x8 guw1 = kuint8x2x8::load((const kuint8x2*)addr(GUW1, 2 * k));
          kuint8x8x2 tguw1 = transpose(guw1);
          karray<half2, 2> scales_mins1 = karray<half2, 2>::load((const half2*)addr(GUQSMV1, 4 * qgidx));
          qdot8(gacc1, tguw1[0], (scales_mins1[0]), x);
          qdot8(uacc1, tguw1[1], (scales_mins1[1]), x);

          kuint8x2x8 guw2 = kuint8x2x8::load((const kuint8x2*)addr(GUW2, 2 * k));
          kuint8x8x2 tguw2 = transpose(guw2);
          karray<half2, 2> scales_mins2 = karray<half2, 2>::load((const half2*)addr(GUQSMV2, 4 * qgidx));
          qdot8(gacc2, tguw2[0], (scales_mins2[1]), x);
          qdot8(uacc2, tguw2[1], (scales_mins2[0]), x);

          kuint8x2x8 guw3 = kuint8x2x8::load((const kuint8x2*)addr(GUW3, 2 * k));
          kuint8x8x2 tguw3 = transpose(guw3);
          karray<half2, 2> scales_mins3 = karray<half2, 2>::load((const half2*)addr(GUQSMV3, 4 * qgidx));
          qdot8(gacc3, tguw3[0], (scales_mins3[0]), x);
          qdot8(uacc3, tguw3[1], (scales_mins3[1]), x);
        }
        if (k + 3 < K) {
          // vectorized
          khalf4 x = khalf4::load(addr(X, k));

          // the quantized group index is the same for all rows
          unsigned qgidx = k >> group_size_shift;

          kuint8x2x4 guw0 = kuint8x2x4::load((const kuint8x2*)addr(GUW0, 2 * k));
          kuint8x4x2 tguw0 = transpose(guw0);
          karray<half2, 2> scales_mins0 = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
          qdot4(gacc0, tguw0[0], (scales_mins0[0]), x);
          qdot4(uacc0, tguw0[1], (scales_mins0[1]), x);

          kuint8x2x4 guw1 = kuint8x2x4::load((const kuint8x2*)addr(GUW1, 2 * k));
          kuint8x4x2 tguw1 = transpose(guw1);
          karray<half2, 2> scales_mins1 = karray<half2, 2>::load((const half2*)addr(GUQSMV1, 4 * qgidx));
          qdot4(gacc1, tguw1[0], (scales_mins1[0]), x);
          qdot4(uacc1, tguw1[1], (scales_mins1[1]), x);

          kuint8x2x4 guw2 = kuint8x2x4::load((const kuint8x2*)addr(GUW2, 2 * k));
          kuint8x4x2 tguw2 = transpose(guw2);
          karray<half2, 2> scales_mins2 = karray<half2, 2>::load((const half2*)addr(GUQSMV2, 4 * qgidx));
          qdot4(gacc2, tguw2[0], (scales_mins2[0]), x);
          qdot4(uacc2, tguw2[1], (scales_mins2[1]), x);

          kuint8x2x4 guw3 = kuint8x2x4::load((const kuint8x2*)addr(GUW3, 2 * k));
          kuint8x4x2 tguw3 = transpose(guw3);
          karray<half2, 2> scales_mins3 = karray<half2, 2>::load((const half2*)addr(GUQSMV3, 4 * qgidx));
          qdot4(gacc3, tguw3[0], (scales_mins3[0]), x);
          qdot4(uacc3, tguw3[1], (scales_mins3[1]), x);

          k += 4;
        }
        if (k + 1 < K) {
          // vectorized
          kfloat2 x = __half22float2(khalf2::load(addr(X, k)));

          // the quantized group index is the same for all rows
          unsigned qgidx = k >> group_size_shift;

          kuint8x2x2 guw0 = kuint8x2x2::load((const kuint8x2*)addr(GUW0, 2 * k));
          kuint8x2x2 tguw0 = transpose(guw0);
          karray<half2, 2> scales_mins0 = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
          qdot2(gacc0, tguw0[0], (scales_mins0[0]), x);
          qdot2(uacc0, tguw0[1], (scales_mins0[1]), x);

          kuint8x2x2 guw1 = kuint8x2x2::load((const kuint8x2*)addr(GUW1, 2 * k));
          kuint8x2x2 tguw1 = transpose(guw1);
          karray<half2, 2> scales_mins1 = karray<half2, 2>::load((const half2*)addr(GUQSMV1, 4 * qgidx));
          qdot2(gacc1, tguw1[0], (scales_mins1[0]), x);
          qdot2(uacc1, tguw1[1], (scales_mins1[1]), x);

          kuint8x2x2 guw2 = kuint8x2x2::load((const kuint8x2*)addr(GUW2, 2 * k));
          kuint8x2x2 tguw2 = transpose(guw2);
          karray<half2, 2> scales_mins2 = karray<half2, 2>::load((const half2*)addr(GUQSMV2, 4 * qgidx));
          qdot2(gacc2, tguw2[0], (scales_mins2[0]), x);
          qdot2(uacc2, tguw2[1], (scales_mins2[1]), x);
        
          kuint8x2x2 guw3 = kuint8x2x2::load((const kuint8x2*)addr(GUW3, 2 * k));
          kuint8x2x2 tguw3 = transpose(guw3);
          karray<half2, 2> scales_mins3 = karray<half2, 2>::load((const half2*)addr(GUQSMV3, 4 * qgidx));
          qdot2(gacc3, tguw3[0], (scales_mins3[0]), x);
          qdot2(uacc3, tguw3[1], (scales_mins3[1]), x);

          k += 2;
        }

        if (k < K) {
          // remainder
          float x = __half2float(*addr(X,k));

          // the quantized group index is the same for all rows
          unsigned qgidx = k >> group_size_shift;

          kuint8x2 guw0 = kuint8x2::load((const uint8_t*)addr(GUW0, 2 * k + 0));
          karray<half2, 2> scales_mins0 = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
          qdot(gacc0, guw0[0], (scales_mins0[0]), x);
          qdot(uacc0, guw0[1], (scales_mins0[1]), x);

          kuint8x2 guw1 = kuint8x2::load((const uint8_t*)addr(GUW1, 2 * k + 0));
          karray<half2, 2> scales_mins1 = karray<half2, 2>::load((const half2*)addr(GUQSMV1, 4 * qgidx));
          qdot(gacc1, guw1[0], (scales_mins1[0]), x);
          qdot(uacc1, guw1[1], (scales_mins1[1]), x);

          kuint8x2 guw2 = kuint8x2::load((const uint8_t*)addr(GUW2, 2 * k + 0));
          karray<half2, 2> scales_mins2 = karray<half2, 2>::load((const half2*)addr(GUQSMV2, 4 * qgidx));
          qdot(gacc2, guw2[0], (scales_mins2[0]), x);
          qdot(uacc2, guw2[1], (scales_mins2[1]), x);

          kuint8x2 guw3 = kuint8x2::load((const uint8_t*)addr(GUW3, 2 * k + 0));
          karray<half2, 2> scales_mins3 = karray<half2, 2>::load((const half2*)addr(GUQSMV3, 4 * qgidx));
          qdot(gacc3, guw3[0], (scales_mins3[0]), x);
          qdot(uacc3, guw3[1], (scales_mins3[1]), x);
        }
      }

      // warp reduce
      gacc0 = warpReduce(gacc0);
      gacc1 = warpReduce(gacc1);
      gacc2 = warpReduce(gacc2);
      gacc3 = warpReduce(gacc3);

      uacc0 = warpReduce(uacc0);
      uacc1 = warpReduce(uacc1);
      uacc2 = warpReduce(uacc2);
      uacc3 = warpReduce(uacc3);

      // reduce accross warps
      if (laneId == 0) {
        atomicAdd(&shared_gaccs[0], gacc0);
        atomicAdd(&shared_gaccs[1], gacc1);
        atomicAdd(&shared_gaccs[2], gacc2);
        atomicAdd(&shared_gaccs[3], gacc3);

        atomicAdd(&shared_uaccs[0], uacc0);
        atomicAdd(&shared_uaccs[1], uacc1);
        atomicAdd(&shared_uaccs[2], uacc2);
        atomicAdd(&shared_uaccs[3], uacc3);
      }
    } else {
      for (int i = 0; i < ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const uint8_t* GUW_ = &GUW[current_row * K * 2];
        const half* GUQSMV_ = &GUQSMV[current_row * G * 2 * 2];
      
        // do the dot product
        float gacc = 0.f;
        float uacc = 0.f;
        for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
          unsigned qgidx = k >> group_size_shift;
          karray<half2, 2> scales_mins = karray<half2, 2>::load((const half2*)addr(GUQSMV_, 4 * qgidx));
          float gw = qint8tofloat(*addr(GUW_,2 * k + 0), (scales_mins[0]));
          float uw = qint8tofloat(*addr(GUW_,2 * k + 1), (scales_mins[1]));
          float x =  __half2float(X[k]);
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
    if (threadIdx.x >= ROWS_PER_BLOCK)
      return;

    int current_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x;

    if (current_row < N) {
      float gacc = shared_gaccs[threadIdx.x]; // read the fully reduced value
      float uacc = shared_uaccs[threadIdx.x]; // read the fully reduced value
      float acc= silu(gacc) * uacc;

      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 8) muillm_int8_gateupsilu_gemv_norm_inputs_kernel(
    const half* __restrict__ NW, // input normalization weights matrix - size K
    const uint8_t* __restrict__ GUW, // weight matrix - size N x K x 2
    const half* __restrict__ GUQSMV, // quantization scales matrix - size N x G x 2 x 2
    const half* __restrict__ X, // input = size K
    half* __restrict__ Y, // output - size N
    unsigned N,
    unsigned K,
    unsigned G,
    float epsilon,
    float scale,
    unsigned group_size_shift
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  float var_x = 0.f;

  __shared__ float shared_gaccs[ROWS_PER_BLOCK];
  __shared__ float shared_uaccs[ROWS_PER_BLOCK];
  __shared__ float shared_var_x;

  // initialize the shared memory
  if (threadIdx.x < ROWS_PER_BLOCK) {
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
    int current_row = blockIdx.x * ROWS_PER_BLOCK + 0;

    // GW
    if (current_row + 3 < N) {
      // compute the t-th element of Y. by doing the dot product with the
      // t-th row of W
      const uint8_t* GUW0 = &GUW[(current_row + 0) * K * 2];
      const uint8_t* GUW1 = &GUW[(current_row + 1) * K * 2];
      const uint8_t* GUW2 = &GUW[(current_row + 2) * K * 2];
      const uint8_t* GUW3 = &GUW[(current_row + 3) * K * 2];

      const half* GUQSMV0 = &GUQSMV[(current_row + 0) * G * 2 * 2];
      const half* GUQSMV1 = &GUQSMV[(current_row + 1) * G * 2 * 2];
      const half* GUQSMV2 = &GUQSMV[(current_row + 2) * G * 2 * 2];
      const half* GUQSMV3 = &GUQSMV[(current_row + 3) * G * 2 * 2];

      float gacc0 = 0.f;
      float gacc1 = 0.f;
      float gacc2 = 0.f;
      float gacc3 = 0.f;

      float uacc0 = 0.f;
      float uacc1 = 0.f;
      float uacc2 = 0.f;
      float uacc3 = 0.f;

      // do the dot product
      {
        unsigned k; // should be 2 * tidx ?
        //*
        for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
          // vectorized
          khalf8 x = khalf8::load(addr(X, k));
          khalf8 nw = khalf8::load(addr(NW, k));

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;

          // accumulate for the variance
          dot8(var_x, __half82float8(x), __half82float8(x));

          // multiply with normalization weights
          x *= nw;

          kuint8x2x8 guw0 = kuint8x2x8::load((const kuint8x2*)addr(GUW0, 2 * k));
          kuint8x8x2 tguw0 = transpose(guw0);
          karray<half2, 2> scales_mins0 = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
          qdot8(gacc0, tguw0[0], (scales_mins0[0]), x);
          qdot8(uacc0, tguw0[1], (scales_mins0[1]), x);

          kuint8x2x8 guw1 = kuint8x2x8::load((const kuint8x2*)addr(GUW1, 2 * k));
          kuint8x8x2 tguw1 = transpose(guw1);
          karray<half2, 2> scales_mins1 = karray<half2, 2>::load((const half2*)addr(GUQSMV1, 4 * qgidx));
          qdot8(gacc1, tguw1[0], (scales_mins1[0]), x);
          qdot8(uacc1, tguw1[1], (scales_mins1[1]), x);

          kuint8x2x8 guw2 = kuint8x2x8::load((const kuint8x2*)addr(GUW2, 2 * k));
          kuint8x8x2 tguw2 = transpose(guw2);
          karray<half2, 2> scales_mins2 = karray<half2, 2>::load((const half2*)addr(GUQSMV2, 4 * qgidx));
          qdot8(gacc2, tguw2[0], (scales_mins2[1]), x);
          qdot8(uacc2, tguw2[1], (scales_mins2[0]), x);

          kuint8x2x8 guw3 = kuint8x2x8::load((const kuint8x2*)addr(GUW3, 2 * k));
          kuint8x8x2 tguw3 = transpose(guw3);
          karray<half2, 2> scales_mins3 = karray<half2, 2>::load((const half2*)addr(GUQSMV3, 4 * qgidx));
          qdot8(gacc3, tguw3[0], (scales_mins3[0]), x);
          qdot8(uacc3, tguw3[1], (scales_mins3[1]), x);
        }
        if (k + 3 < K) {
          // vectorized
          khalf4 x = khalf4::load(addr(X, k));
          khalf4 nw = khalf4::load(addr(NW, k));

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;

          // accumulate for the variance
          dot4(var_x, __half42float4(x), __half42float4(x));

          // TODO: retry fusing normalization weights???
          // multiply with normalization weights
          x *= nw;

          kuint8x2x4 guw0 = kuint8x2x4::load((const kuint8x2*)addr(GUW0, 2 * k));
          kuint8x4x2 tguw0 = transpose(guw0);
          karray<half2, 2> scales_mins0 = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
          qdot4(gacc0, tguw0[0], (scales_mins0[0]), x);
          qdot4(uacc0, tguw0[1], (scales_mins0[1]), x);

          kuint8x2x4 guw1 = kuint8x2x4::load((const kuint8x2*)addr(GUW1, 2 * k));
          kuint8x4x2 tguw1 = transpose(guw1);
          karray<half2, 2> scales_mins1 = karray<half2, 2>::load((const half2*)addr(GUQSMV1, 4 * qgidx));
          qdot4(gacc1, tguw1[0], (scales_mins1[0]), x);
          qdot4(uacc1, tguw1[1], (scales_mins1[1]), x);

          kuint8x2x4 guw2 = kuint8x2x4::load((const kuint8x2*)addr(GUW2, 2 * k));
          kuint8x4x2 tguw2 = transpose(guw2);
          karray<half2, 2> scales_mins2 = karray<half2, 2>::load((const half2*)addr(GUQSMV2, 4 * qgidx));
          qdot4(gacc2, tguw2[0], (scales_mins2[0]), x);
          qdot4(uacc2, tguw2[1], (scales_mins2[1]), x);

          kuint8x2x4 guw3 = kuint8x2x4::load((const kuint8x2*)addr(GUW3, 2 * k));
          kuint8x4x2 tguw3 = transpose(guw3);
          karray<half2, 2> scales_mins3 = karray<half2, 2>::load((const half2*)addr(GUQSMV3, 4 * qgidx));
          qdot4(gacc3, tguw3[0], (scales_mins3[0]), x);
          qdot4(uacc3, tguw3[1], (scales_mins3[1]), x);

          k += 4;
        }
        if (k + 1 < K) {
          // vectorized
          kfloat2 x = __half22float2(khalf2::load(addr(X, k)));
          kfloat2 nw = __half22float2(khalf2::load(addr(NW, k)));

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;

          // accumulate for the variance
          dot2(var_x, x, x);

          // multiply with normalization weights
          x *= nw;

          kuint8x2x2 guw0 = kuint8x2x2::load((const kuint8x2*)addr(GUW0, 2 * k));
          kuint8x2x2 tguw0 = transpose(guw0);
          karray<half2, 2> scales_mins0 = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
          qdot2(gacc0, tguw0[0], (scales_mins0[0]), x);
          qdot2(uacc0, tguw0[1], (scales_mins0[1]), x);

          kuint8x2x2 guw1 = kuint8x2x2::load((const kuint8x2*)addr(GUW1, 2 * k));
          kuint8x2x2 tguw1 = transpose(guw1);
          karray<half2, 2> scales_mins1 = karray<half2, 2>::load((const half2*)addr(GUQSMV1, 4 * qgidx));
          qdot2(gacc1, tguw1[0], (scales_mins1[0]), x);
          qdot2(uacc1, tguw1[1], (scales_mins1[1]), x);

          kuint8x2x2 guw2 = kuint8x2x2::load((const kuint8x2*)addr(GUW2, 2 * k));
          kuint8x2x2 tguw2 = transpose(guw2);
          karray<half2, 2> scales_mins2 = karray<half2, 2>::load((const half2*)addr(GUQSMV2, 4 * qgidx));
          qdot2(gacc2, tguw2[0], (scales_mins2[0]), x);
          qdot2(uacc2, tguw2[1], (scales_mins2[1]), x);
        
          kuint8x2x2 guw3 = kuint8x2x2::load((const kuint8x2*)addr(GUW3, 2 * k));
          kuint8x2x2 tguw3 = transpose(guw3);
          karray<half2, 2> scales_mins3 = karray<half2, 2>::load((const half2*)addr(GUQSMV3, 4 * qgidx));
          qdot2(gacc3, tguw3[0], (scales_mins3[0]), x);
          qdot2(uacc3, tguw3[1], (scales_mins3[1]), x);

          k += 2;
        }

        if (k < K) {
          // remainder
          float x = __half2float(*addr(X,k));
          float nw = __half2float(*addr(NW,k));

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;
          
          // accumulate for the variance
          var_x += x * x;

          // multiply with normalization weights
          x *= nw;

          kuint8x2 guw0 = kuint8x2::load((const uint8_t*)addr(GUW0, 2 * k + 0));
          karray<half2, 2> scales_mins0 = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
          qdot(gacc0, guw0[0], (scales_mins0[0]), x);
          qdot(uacc0, guw0[1], (scales_mins0[1]), x);

          kuint8x2 guw1 = kuint8x2::load((const uint8_t*)addr(GUW1, 2 * k + 0));
          karray<half2, 2> scales_mins1 = karray<half2, 2>::load((const half2*)addr(GUQSMV1, 4 * qgidx));
          qdot(gacc1, guw1[0], (scales_mins1[0]), x);
          qdot(uacc1, guw1[1], (scales_mins1[1]), x);

          kuint8x2 guw2 = kuint8x2::load((const uint8_t*)addr(GUW2, 2 * k + 0));
          karray<half2, 2> scales_mins2 = karray<half2, 2>::load((const half2*)addr(GUQSMV2, 4 * qgidx));
          qdot(gacc2, guw2[0], (scales_mins2[0]), x);
          qdot(uacc2, guw2[1], (scales_mins2[1]), x);

          kuint8x2 guw3 = kuint8x2::load((const uint8_t*)addr(GUW3, 2 * k + 0));
          karray<half2, 2> scales_mins3 = karray<half2, 2>::load((const half2*)addr(GUQSMV3, 4 * qgidx));
          qdot(gacc3, guw3[0], (scales_mins3[0]), x);
          qdot(uacc3, guw3[1], (scales_mins3[1]), x);
        }
      }

      // warp reduce
      var_x = warpReduce(var_x);
      gacc0 = warpReduce(gacc0);
      gacc1 = warpReduce(gacc1);
      gacc2 = warpReduce(gacc2);
      gacc3 = warpReduce(gacc3);

      uacc0 = warpReduce(uacc0);
      uacc1 = warpReduce(uacc1);
      uacc2 = warpReduce(uacc2);
      uacc3 = warpReduce(uacc3);

      // reduce accross warps
      if (laneId == 0) {
        atomicAdd(&shared_var_x, var_x);
        atomicAdd(&shared_gaccs[0], gacc0);
        atomicAdd(&shared_gaccs[1], gacc1);
        atomicAdd(&shared_gaccs[2], gacc2);
        atomicAdd(&shared_gaccs[3], gacc3);

        atomicAdd(&shared_uaccs[0], uacc0);
        atomicAdd(&shared_uaccs[1], uacc1);
        atomicAdd(&shared_uaccs[2], uacc2);
        atomicAdd(&shared_uaccs[3], uacc3);
      }
    } else {
      for (int i = 0; i < ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const uint8_t* GUW_ = &GUW[current_row * K * 2];
        const half* GUQSMV_ = &GUQSMV[current_row * G * 2 * 2];
      
        // do the dot product
        float gacc = 0.f;
        float uacc = 0.f;
        if (i == 0) {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            unsigned qgidx = k >> group_size_shift;
            karray<half2, 2> scales_mins = karray<half2, 2>::load((const half2*)addr(GUQSMV_, 4 * qgidx));
            float gw = qint8tofloat(*addr(GUW_,2 * k + 0), (scales_mins[0]));
            float uw = qint8tofloat(*addr(GUW_,2 * k + 1), (scales_mins[1]));

            float x =  __half2float(X[k]);
            float nw = __half2float(NW[k]);

            // accumuate the variance
            var_x += x * x;

            // multiply with normalization weights
            x *= nw;

            gacc += gw * x;
            uacc += uw * x;
          }
        } else {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            unsigned qgidx = k >> group_size_shift;
            karray<half2, 2> scales_mins = karray<half2, 2>::load((const half2*)addr(GUQSMV_, 4 * qgidx));
            float gw = qint8tofloat(*addr(GUW_,2 * k + 0), (scales_mins[0]));
            float uw = qint8tofloat(*addr(GUW_,2 * k + 1), (scales_mins[1]));

            float x =  __half2float(X[k]);
            float nw = __half2float(NW[k]);

            // don't accumulate the variance (we already have done it with i == 0)

            // multiply with normalization weights
            x *= nw;

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

    if (threadIdx.x >= ROWS_PER_BLOCK)
      return;

    int current_row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x;

    if (current_row < N) {
      float gacc = shared_gaccs[threadIdx.x] * rsqrt_var; // read the fully reduced value and scale
      float uacc = shared_uaccs[threadIdx.x] * rsqrt_var; // read the fully reduced value and scale
      float acc= silu(gacc) * uacc;

      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor muillm_int8_gateupsilu_forward(
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor gate_up_weights,
    torch::Tensor gate_up_scales_min_vals,
    int group_size_shift,
    torch::Tensor x) {
  bool normalize = norm_weights.defined();
  if (normalize) {
    CHECK_INPUT(norm_weights);
  }
  CHECK_INPUT(gate_up_weights);
  CHECK_INPUT(gate_up_scales_min_vals);
  CHECK_INPUT(x);


  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto N = gate_up_weights.size(0);
  const auto K = gate_up_weights.size(1);
  const auto G = K >> group_size_shift;

  auto dtype = torch::kFloat16;
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(at::kCUDA)
                            .requires_grad(false);

  // y has the same dimensions as x, except the last dim that is given by
  // the out_features of weights
  auto output_sizes = x.sizes().vec();
  output_sizes[output_sizes.size() - 1] = N;

  auto y = torch::empty(output_sizes, output_options);

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(N, ROWS_PER_BLOCK);

  if (normalize) {
    float scale = 1.f / K;

    muillm_int8_gateupsilu_gemv_norm_inputs_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      norm_weights.defined() ? (const half*)norm_weights.data_ptr() : nullptr,
      (const uint8_t*)gate_up_weights.data_ptr(),
      (const half*)gate_up_scales_min_vals.data_ptr(),
      (const half*)x.data_ptr(),
      (half*)y.data_ptr(),
      N,
      K,
      G,
      epsilon,
      scale,
      group_size_shift
    );
  } else {

    muillm_int8_gateupsilu_gemv_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      (const uint8_t*)gate_up_weights.data_ptr(),
      (const half*)gate_up_scales_min_vals.data_ptr(),
      (const half*)x.data_ptr(),
      (half*)y.data_ptr(),
      N,
      K,
      G,
      group_size_shift
    );
  }

  return y;
}