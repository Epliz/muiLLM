#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#include "karray.cuh"

#include <tuple>

#define THREADS_PER_BLOCK 64

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

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

using khalf2 = karray<half, 2>;
using kfloat2 = karray<float, 2>;

using khalf4 = karray<half, 4>;
using kfloat4 = karray<float, 4>;

using khalf8 = karray<half, 8>;
using kfloat8 = karray<float, 8>;

using kuint8x2 = karray<uint8_t, 2>;
using kuint8x4 = karray<uint8_t, 4>;
using kuint8x8 = karray<uint8_t, 8>;

static inline half __device__ qint8tohalf(uint8_t q, half2 scale_min_val) {
  half scale = scale_min_val.x;
  half min_val = scale_min_val.y;
  return __float2half(q * __half2float(scale) + __half2float(min_val));
}

template<unsigned N>
static inline karray<half, N> __device__ qint8xNtohalfN(const karray<uint8_t, N>& qs, half2 scale_min_val) {
  float scale = __half2float(scale_min_val.x);
  float min_val = __half2float(scale_min_val.y);
  karray<half, N> r;
  for (unsigned i = 0; i < N; i++) {
    r[i] = __float2half(qs[i] * scale + min_val);
  }
  return r;
}

static inline khalf2 __device__ qint8x2tohalf2(const kuint8x2& qs, half2 scale_min_val) {
  return qint8xNtohalfN<2>(qs, scale_min_val);
}

static inline khalf4 __device__ qint8x4tohalf4(const kuint8x4& qs, half2 scale_min_val) {
  return qint8xNtohalfN<4>(qs, scale_min_val);
}

static inline khalf8 __device__ qint8x8tohalf8(const kuint8x8& qs, half2 scale_min_val) {
  return qint8xNtohalfN<8>(qs, scale_min_val);
}


__global__ void muillm_int8_dequantize_kernel(
    const uint8_t* __restrict__ W, // weight matrix - size N x K
    const half* __restrict__ QSMV, // quantization scales and minimum values matrix - size N x G
    half* __restrict__ DEQUANT_W, // output - size N x K
    unsigned N,
    unsigned K,
    unsigned G,
    unsigned group_size_shift
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  int current_row = blockIdx.x;

  // compute the t-th element of Y. by doing the dot product with the
  // t-th row of W
  const uint8_t* W0 = &W[(current_row + 0) * K];

  const half* QSMV0 = &QSMV[(current_row + 0) * G * 2];


  half* DQW0 = &DEQUANT_W[(current_row + 0) * K];

  unsigned k; // should be 2 * tidx ?
  //*
  for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
    // vectorized
    // the quantized group index is the same for all rows
    unsigned qgidx = k >> group_size_shift;

    kuint8x8 guw0 = kuint8x8::load((const uint8_t*)addr(W0, k));

    khalf8 dqw = qint8x8tohalf8(guw0, *(const half2*)addr(QSMV0, 2 * qgidx));

    khalf8::store(waddr(DQW0, k), dqw);
  }
  if (k + 3 < K) {
    // vectorized

    // the quantized group index is the same for all rows
    unsigned qgidx = k >> group_size_shift;

    kuint8x4 guw0 = kuint8x4::load((const uint8_t*)addr(W0, k));

    khalf4 dqw = qint8x4tohalf4(guw0, *(const half2*)addr(QSMV0, 2 * qgidx));

    khalf4::store(waddr(DQW0, k), dqw);

    k += 4;
  }
  if (k + 1 < K) {
    // vectorized

    // the quantized group index is the same for all rows
    unsigned qgidx = k >> group_size_shift;

    kuint8x2 guw0 = kuint8x2::load((const uint8_t*)addr(W0, k));

    khalf2 dqw = qint8x2tohalf2(guw0, *(const half2*)addr(QSMV0, 2 * qgidx));

    khalf2::store(waddr(DQW0, k), dqw);

    k += 2;
  }

  if (k < K) {
    // remainder

    // the quantized group index is the same for all rows
    unsigned qgidx = k >> group_size_shift;

    half dqw = qint8tohalf(*((const uint8_t*)addr(W0, k)), *(const half2*)addr(QSMV0, 2 * qgidx));

    DQW0[k] = dqw;
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor muillm_int8_dequantize_forward(
    torch::Tensor weights,
    torch::Tensor scales_min_vals,
    int group_size_shift) {
  CHECK_INPUT(weights);
  CHECK_INPUT(scales_min_vals);


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

  auto dequantized_weights = torch::empty({N, K}, output_options);

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = N;

  muillm_int8_dequantize_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint8_t*)weights.data_ptr(),
    (const half*)scales_min_vals.data_ptr(),
    (half*)dequantized_weights.data_ptr(),
    N,
    K,
    G,
    group_size_shift
  );

  return dequantized_weights;
}