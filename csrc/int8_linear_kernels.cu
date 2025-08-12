#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cstdint>

#include "karray.cuh"

#include "int8_linear_kernels.cuh"

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
using kfloat2 = karray<float, 2>;

using khalf4 = karray<half, 4>;
using kfloat4 = karray<float, 4>;

using khalf8 = karray<half, 8>;
using kfloat8 = karray<float, 8>;

using kuint8x2 = karray<uint8_t, 2>;
using kuint8x4 = karray<uint8_t, 4>;
using kuint8x8 = karray<uint8_t, 8>;

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
    r.data[i] = __half2float(v.data[i]);
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
    acc += a.data[i] * b.data[i];
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
    r = __hadd(r, x.data[i]);
  }
  return r;
}


static inline half __device__ hsum4(const khalf4& x) {
  half r0 = __float2half(0.f);
  half r1 = __float2half(0.f);
  // TODO: specialize to have less long dependency chain?
  for (unsigned i = 0; i < 4; i+=2) {
    r0 = __hadd(r0, x.data[i + 0]);
    r1 = __hadd(r1, x.data[i + 1]);
  }
  return __hadd(r0, r1);
}

static inline half __device__ hsum8(const khalf8& x) {
  half r0 = __float2half(0.f);
  half r1 = __float2half(0.f);
  // TODO: specialize to have less long dependency chain?
  for (unsigned i = 0; i < 8; i+=2) {
    r0 = __hadd(r0, x.data[i + 0]);
    r1 = __hadd(r1, x.data[i + 1]);
  }
  return __hadd(r0, r1);
}

template <unsigned N>
static inline float __device__ sumN(const karray<float, N>& x) {
  float r = 0.f;
  // TODO: specialize to have less long dependency chain?
  for (unsigned i = 0; i < N; i++) {
    r += x.data[i];
  }
  return r;
}


static inline float __device__ sum4(const kfloat4& x) {
  float r0 = 0.f;
  float r1 = 0.f;
  // TODO: specialize to have less long dependency chain?
  for (unsigned i = 0; i < 4; i+=2) {
    r0 += x.data[i + 0];
    r1 += x.data[i + 1];
  }
  return r0 + r1;
}

static inline float __device__ sum8(const kfloat8& x) {
  float r0 = 0.f;
  float r1 = 0.f;
  // TODO: specialize to have less long dependency chain?
  for (unsigned i = 0; i < 8; i+=2) {
    r0 += x.data[i + 0];
    r1 += x.data[i + 1];
  }
  return r0 + r1;
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
    r.data[i] = qs.data[i] * scale + min_val;
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
    r += qw.data[i] * x.data[i];
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

static inline void __device__ qdot4(float& acc, const kuint8x4& qw, half2 scale_min_val, const kfloat4& x) {
  ifdotN_scale<4>(acc, qw, scale_min_val.x, x);
  acc += __half2float(scale_min_val.y) * sum4(x);
}

// TODO: lots of cvt_f32_f16 in the assembly, need to figure it out
static inline void __device__ qdot8(float& acc, const kuint8x8& qw, half2 scale_min_val, const kfloat8& x) {
  ifdotN_scale<8>(acc, qw, scale_min_val.x, x);
  acc += __half2float(scale_min_val.y) * sum8(x);
}

__global__ void muillm_int8_gemv_kernel(
    const uint8_t* __restrict__ W, // weight matrix - size N x K
    const half* __restrict__ QSMV, // quantization scales and minimum values matrix - size N x G x 2
    const half* __restrict__ X, // input = size K
    mui_activation activation, // activation function 
    const half* __restrict__ MB, // optional multiplicative bias - size N (applied before additive bias)
    const half* __restrict__ AB, // optional additive bias - size N
    half* __restrict__ Y, // output - size N
    unsigned N,
    unsigned K,
    unsigned G,
    unsigned group_size_shift
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  // can process ROWS_PER_BLOCK rows
  // shared state to do the reductions
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
      const uint8_t* W0 = &W[(current_row + 0) * K];
      const uint8_t* W1 = &W[(current_row + 1) * K];
      const uint8_t* W2 = &W[(current_row + 2) * K];
      const uint8_t* W3 = &W[(current_row + 3) * K];

      const half* QSMV0 = &QSMV[(current_row + 0) * G * 2];
      const half* QSMV1 = &QSMV[(current_row + 1) * G * 2];
      const half* QSMV2 = &QSMV[(current_row + 2) * G * 2];
      const half* QSMV3 = &QSMV[(current_row + 3) * G * 2];

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
          kfloat8 x = __half82float8(khalf8::load(addr(X, k)));

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;

          qdot8(acc0, kuint8x8::load(addr(W0, k)), *(const half2*)addr(QSMV0, 2 * qgidx), x);
          qdot8(acc1, kuint8x8::load(addr(W1, k)), *(const half2*)addr(QSMV1, 2 * qgidx), x);
          qdot8(acc2, kuint8x8::load(addr(W2, k)), *(const half2*)addr(QSMV2, 2 * qgidx), x);
          qdot8(acc3, kuint8x8::load(addr(W3, k)), *(const half2*)addr(QSMV3, 2 * qgidx), x);
        }
        if (k + 3 < K) {
          // vectorized
          kfloat4 x = __half42float4(khalf4::load(addr(X, k)));

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;

          qdot4(acc0, kuint8x4::load(addr(W0, k)), *(const half2*)addr(QSMV0, 2 * qgidx), x);
          qdot4(acc1, kuint8x4::load(addr(W1, k)), *(const half2*)addr(QSMV1, 2 * qgidx), x);
          qdot4(acc2, kuint8x4::load(addr(W2, k)), *(const half2*)addr(QSMV2, 2 * qgidx), x);
          qdot4(acc3, kuint8x4::load(addr(W3, k)), *(const half2*)addr(QSMV3, 2 * qgidx), x);

          k += 4;
        }
        if (k + 1 < K) {
          // vectorized
          kfloat2 x = __half22float2(khalf2::load(addr(X, k)));

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;

          qdot2(acc0, kuint8x2::load(addr(W0, k)), *(const half2*)addr(QSMV0, 2 * qgidx), x);
          qdot2(acc1, kuint8x2::load(addr(W1, k)), *(const half2*)addr(QSMV1, 2 * qgidx), x);
          qdot2(acc2, kuint8x2::load(addr(W2, k)), *(const half2*)addr(QSMV2, 2 * qgidx), x);
          qdot2(acc3, kuint8x2::load(addr(W3, k)), *(const half2*)addr(QSMV3, 2 * qgidx), x);


          k += 2;
        }
        if (k < K) {

          // remainder
          float x = __half2float(*addr(X,k));

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;

          qdot(acc0, *((const uint8_t*)addr(W0, k)), *(const half2*)addr(QSMV0, 2 * qgidx), x);
          qdot(acc1, *((const uint8_t*)addr(W1, k)), *(const half2*)addr(QSMV1, 2 * qgidx), x);
          qdot(acc2, *((const uint8_t*)addr(W2, k)), *(const half2*)addr(QSMV2, 2 * qgidx), x);
          qdot(acc3, *((const uint8_t*)addr(W3, k)), *(const half2*)addr(QSMV3, 2 * qgidx), x);
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

        const uint8_t* W_ = &W[current_row * K];
        const half* QSMV_ = &QSMV[current_row * G * 2];
      
        // do the dot product
        float acc = 0.f;
        {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            unsigned qgidx = k >> group_size_shift;
            float w = qint8tofloat(*addr(W_,k), *(const half2*)addr(QSMV_, 2 * qgidx));
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

      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}


__global__ void muillm_int8_gemv_norm_inputs_kernel(
    const half* __restrict__ NW, // input normalization weights matrix - size K
    const uint8_t* __restrict__ W, // weight matrix - size N x K
    const half* __restrict__ QSMV, // quantization scales and minimum values matrix - size N x G x 2
    const half* __restrict__ X, // input = size K
    mui_activation activation, // activation function 
    const half* __restrict__ MB, // optional multiplicative bias - size N (applied before additive bias)
    const half* __restrict__ AB, // optional additive bias - size N
    half* __restrict__ Y, // output - size N
    unsigned N,
    unsigned K,
    unsigned G,
    float epsilon,
    float weights_offset,
    float scale,
    unsigned group_size_shift
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
      const uint8_t* W0 = &W[(current_row + 0) * K];
      const uint8_t* W1 = &W[(current_row + 1) * K];
      const uint8_t* W2 = &W[(current_row + 2) * K];
      const uint8_t* W3 = &W[(current_row + 3) * K];

      const half* QSMV0 = &QSMV[(current_row + 0) * G * 2];
      const half* QSMV1 = &QSMV[(current_row + 1) * G * 2];
      const half* QSMV2 = &QSMV[(current_row + 2) * G * 2];
      const half* QSMV3 = &QSMV[(current_row + 3) * G * 2];

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
          kfloat8 x = __half82float8(khalf8::load(addr(X, k)));
          kfloat8 nw = __half82float8(khalf8::load(addr(NW, k))) + weights_offset;

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;

          // accumulate for the variance
          dot8(var_x, x, x);

          // multiply with normalization weights
          x *= nw;

          qdot8(acc0, kuint8x8::load(addr(W0, k)), *(const half2*)addr(QSMV0, 2 * qgidx), x);
          qdot8(acc1, kuint8x8::load(addr(W1, k)), *(const half2*)addr(QSMV1, 2 * qgidx), x);
          qdot8(acc2, kuint8x8::load(addr(W2, k)), *(const half2*)addr(QSMV2, 2 * qgidx), x);
          qdot8(acc3, kuint8x8::load(addr(W3, k)), *(const half2*)addr(QSMV3, 2 * qgidx), x);
        }
        if (k + 3 < K) {

          // vectorized
          kfloat4 x = __half42float4(khalf4::load(addr(X, k)));
          kfloat4 nw = __half42float4(khalf4::load(addr(NW, k))) + weights_offset;

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;

          // accumulate for the variance
          dot4(var_x, x, x);

          // multiply with normalization weights
          x *= nw;

          qdot4(acc0, kuint8x4::load(addr(W0, k)), *(const half2*)addr(QSMV0, 2 * qgidx), x);
          qdot4(acc1, kuint8x4::load(addr(W1, k)), *(const half2*)addr(QSMV1, 2 * qgidx), x);
          qdot4(acc2, kuint8x4::load(addr(W2, k)), *(const half2*)addr(QSMV2, 2 * qgidx), x);
          qdot4(acc3, kuint8x4::load(addr(W3, k)), *(const half2*)addr(QSMV3, 2 * qgidx), x);

          k += 4;
        }
        if (k + 1 < K) {
          // vectorized
          kfloat2 x = __half22float2(khalf2::load(addr(X, k)));
          kfloat2 nw = __half22float2(khalf2::load(addr(NW, k))) + weights_offset;

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;

          // accumulate for the variance
          dot2(var_x, x, x);

          // multiply with normalization weights
          x *= nw;

          qdot2(acc0, kuint8x2::load(addr(W0, k)), *(const half2*)addr(QSMV0, 2 * qgidx), x);
          qdot2(acc1, kuint8x2::load(addr(W1, k)), *(const half2*)addr(QSMV1, 2 * qgidx), x);
          qdot2(acc2, kuint8x2::load(addr(W2, k)), *(const half2*)addr(QSMV2, 2 * qgidx), x);
          qdot2(acc3, kuint8x2::load(addr(W3, k)), *(const half2*)addr(QSMV3, 2 * qgidx), x);


          k += 2;
        }
        if (k < K) {
          // remainder
          float x = __half2float(*addr(X,k));
          float nw = __half2float(*addr(NW,k)) + weights_offset;

          // the quantized group index is the same for all rows
          // TODO: remove division and replace with shift
          unsigned qgidx = k >> group_size_shift;
          
          // accumulate for the variance
          var_x += x * x;

          // multiply with normalization weights
          x *= nw;

          qdot(acc0, *((const uint8_t*)addr(W0, k)), *(const half2*)addr(QSMV0, 2 * qgidx), x);
          qdot(acc1, *((const uint8_t*)addr(W1, k)), *(const half2*)addr(QSMV1, 2 * qgidx), x);
          qdot(acc2, *((const uint8_t*)addr(W2, k)), *(const half2*)addr(QSMV2, 2 * qgidx), x);
          qdot(acc3, *((const uint8_t*)addr(W3, k)), *(const half2*)addr(QSMV3, 2 * qgidx), x);
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

        const uint8_t* W_ = &W[current_row * K];
        const half* QSMV_ = &QSMV[current_row * G * 2];
      
        // do the dot product
        float acc = 0.f;
        if (i == 0) {
          // accumulate the variance
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            unsigned qgidx = k >> group_size_shift;
            float w = qint8tofloat(*addr(W_,k), *(const half2*)addr(QSMV_, 2 * qgidx));

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
            unsigned qgidx = k >> group_size_shift;
            float w = qint8tofloat(*addr(W_,k), *(const half2*)addr(QSMV_, 2 * qgidx));

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

      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor muillm_int8_linear_activ_forward(
    torch::Tensor norm_weights,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor weights,
    torch::Tensor scales_min_vals,
    int group_size_shift,
    mui_activation activ,
    torch::Tensor mul_bias,
    torch::Tensor add_bias,
    torch::Tensor x) {
  bool normalize = norm_weights.defined();
  if (normalize) {
    CHECK_INPUT(norm_weights);
  }
  CHECK_INPUT(weights);
  if (mul_bias.defined()) {
    CHECK_INPUT(mul_bias);
  }
  if (add_bias.defined()) {
    CHECK_INPUT(add_bias);
  }
  CHECK_INPUT(x);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto N = weights.size(0);
  const auto K = weights.size(1);
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

    muillm_int8_gemv_norm_inputs_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      norm_weights.defined() ? (const half*)norm_weights.data_ptr() : nullptr,
      (const uint8_t*)weights.data_ptr(),
      (const half*)scales_min_vals.data_ptr(),
      (const half*)x.data_ptr(),
      activ,
      mul_bias.defined() ? (const half*)mul_bias.data_ptr() : nullptr,
      add_bias.defined() ? (const half*)add_bias.data_ptr() : nullptr,
      (half*)y.data_ptr(),
      N,
      K,
      G,
      epsilon,
      norm_weights_offset,
      scale,
      group_size_shift
    );
  } else {
    muillm_int8_gemv_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      (const uint8_t*)weights.data_ptr(),
      (const half*)scales_min_vals.data_ptr(),
      (const half*)x.data_ptr(),
      activ,
      mul_bias.defined() ? (const half*)mul_bias.data_ptr() : nullptr,
      add_bias.defined() ? (const half*)add_bias.data_ptr() : nullptr,
      (half*)y.data_ptr(),
      N,
      K,
      G,
      group_size_shift
    );
  }

  return y;
}