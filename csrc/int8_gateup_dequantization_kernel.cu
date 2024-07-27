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

static_assert(sizeof(karray<half2, 2>) == 8);

__global__ void muillm_int8_gateupsilu_dequantize_kernel(
    const uint8_t* __restrict__ GUW, // weight matrix - size N x K x 2
    const half* __restrict__ GUQSMV, // quantization scales and minimum values matrix - size N x G x 2 x 2
    half* __restrict__ DEQUANT_GW, // output - size N x K
    half* __restrict__ DEQUANT_UW, // output - size N x K
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
  const uint8_t* GUW0 = &GUW[(current_row + 0) * K * 2];

  const half* GUQSMV0 = &GUQSMV[(current_row + 0) * G * 2 * 2];


  half* DQGW0 = &DEQUANT_GW[(current_row + 0) * K];
  half* DQUW0 = &DEQUANT_UW[(current_row + 0) * K];

  unsigned k; // should be 2 * tidx ?
  //*
  for (k = threadIdx.x * 8; k + 7 < K; k += (THREADS_PER_BLOCK * 8)) {
    // vectorized

    // the quantized group index is the same for all rows
    unsigned qgidx = k >> group_size_shift;

    kuint8x2x8 guw0 = kuint8x2x8::load((const kuint8x2*)addr(GUW0, 2 * k));
    kuint8x8x2 tguw0 = transpose(guw0);

    karray<half2, 2> scales_mins = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
    khalf8 dqgw = qint8x8tohalf8(tguw0[0], (scales_mins[0]));
    khalf8 dquw = qint8x8tohalf8(tguw0[1], (scales_mins[1]));

    khalf8::store(waddr(DQGW0, k), dqgw);
    khalf8::store(waddr(DQUW0, k), dquw);
  }
  if (k + 3 < K) {
    // vectorized

    // the quantized group index is the same for all rows
    unsigned qgidx = k >> group_size_shift;

    kuint8x2x4 guw0 = kuint8x2x4::load((const kuint8x2*)addr(GUW0, 2 * k));
    kuint8x4x2 tguw0 = transpose(guw0);

    karray<half2, 2> scales_mins = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
    khalf4 dqgw = qint8x4tohalf4(tguw0[0], (scales_mins[0]));
    khalf4 dquw = qint8x4tohalf4(tguw0[1], (scales_mins[1]));

    khalf4::store(waddr(DQGW0, k), dqgw);
    khalf4::store(waddr(DQUW0, k), dquw);

    k += 4;
  }
  if (k + 1 < K) {
    // vectorized

    // the quantized group index is the same for all rows
    unsigned qgidx = k >> group_size_shift;

    kuint8x2x2 guw0 = kuint8x2x2::load((const kuint8x2*)addr(GUW0, 2 * k));
    kuint8x2x2 tguw0 = transpose(guw0);

    karray<half2, 2> scales_mins = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
    khalf2 dqgw = qint8x2tohalf2(tguw0[0], (scales_mins[0]));
    khalf2 dquw = qint8x2tohalf2(tguw0[1], (scales_mins[1]));

    khalf2::store(waddr(DQGW0, k), dqgw);
    khalf2::store(waddr(DQUW0, k), dquw);

    k += 2;
  }

  if (k < K) {
    // remainder

    // the quantized group index is the same for all rows
    unsigned qgidx = k >> group_size_shift;

    karray<half2, 2> scales_mins = karray<half2, 2>::load((const half2*)addr(GUQSMV0, 4 * qgidx));
    half dqgw = qint8tohalf(*((const uint8_t*)addr(GUW0, 2 * k + 0)), (scales_mins[0]));
    half dquw = qint8tohalf(*((const uint8_t*)addr(GUW0, 2 * k + 1)), (scales_mins[1]));

    DQGW0[k] = dqgw;
    DQUW0[k] = dquw;
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<at::Tensor, at::Tensor> muillm_int8_gateupsilu_dequantize_forward(
    torch::Tensor gate_up_weights,
    torch::Tensor gate_up_scales_min_vals,
    int group_size_shift) {
  CHECK_INPUT(gate_up_weights);
  CHECK_INPUT(gate_up_scales_min_vals);


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

  auto dequantized_gate_weights = torch::empty({N, K}, output_options);
  auto dequantized_up_weights = torch::empty({N, K}, output_options);

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = N;

  muillm_int8_gateupsilu_dequantize_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint8_t*)gate_up_weights.data_ptr(),
    (const half*)gate_up_scales_min_vals.data_ptr(),
    (half*)dequantized_gate_weights.data_ptr(),
    (half*)dequantized_up_weights.data_ptr(),
    N,
    K,
    G,
    group_size_shift
  );

  return std::make_tuple(dequantized_gate_weights, dequantized_up_weights);
}