#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#include "linear_kernels.cuh"

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 64

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

static inline void __device__ dot2(float& acc, const float2& a, const float2& b) {
  acc += a.x * b.x;
  acc += a.y * b.y;
}

struct __align__(8) half4 {
    half x;
    half y;
    half z;
    half w;
};

static inline float4 __device__ __half42float4(const half4& v) {
  float4 f;
  f.x = __half2float(v.x);
  f.y = __half2float(v.y);
  f.z = __half2float(v.z);
  f.w = __half2float(v.w);

  return f;
}

static inline void __device__ dot4(float& acc, const float4& a, const float4& b) {
  acc += a.x * b.x;
  acc += a.y * b.y;
  acc += a.z * b.z;
  acc += a.w * b.w;
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

__global__ void muillm_gemv_kernel(
    const half* __restrict__ W, // weight matrix - size N x K
    const half* __restrict__ X, // input = size K
    mui_activation activation, // activation function 
    const half* __restrict__ MB, // optional multiplicative bias - size N (applied before additive bias)
    const half* __restrict__ AB, // optional additive bias - size N
    half* __restrict__ Y, // output - size N
    unsigned N,
    unsigned K
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
        for (k = threadIdx.x * 2; k + 1 < K; k += (THREADS_PER_BLOCK * 2)) {
          // vectorized
          float2 x = __half22float2(*((const half2*)addr(X, k)));
          float2 w0 = __half22float2(*((const half2*)addr(W0, k)));
          float2 w1 = __half22float2(*((const half2*)addr(W1, k)));
          float2 w2 = __half22float2(*((const half2*)addr(W2, k)));
          float2 w3 = __half22float2(*((const half2*)addr(W3, k)));

          dot2(acc0, w0, x);
          dot2(acc1, w1, x);
          dot2(acc2, w2, x);
          dot2(acc3, w3, x);
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
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
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

      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}


__global__ void muillm_gemv_norm_inputs_kernel(
    const half* __restrict__ NW, // input normalization weights matrix - size K
    const half* __restrict__ W, // weight matrix - size N x K
    const half* __restrict__ X, // input = size K
    mui_activation activation, // activation function 
    const half* __restrict__ MB, // optional multiplicative bias - size N (applied before additive bias)
    const half* __restrict__ AB, // optional additive bias - size N
    half* __restrict__ Y, // output - size N
    unsigned N,
    unsigned K,
    float epsilon,
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
        for (k = threadIdx.x * 2; k + 1 < K; k += (THREADS_PER_BLOCK * 2)) {
          // vectorized
          float2 x = __half22float2(*((const half2*)addr(X, k)));
          float2 nw = __half22float2(*((const half2*)addr(NW, k)));

          float2 w0 = __half22float2(*((const half2*)addr(W0, k)));
          float2 w1 = __half22float2(*((const half2*)addr(W1, k)));
          float2 w2 = __half22float2(*((const half2*)addr(W2, k)));
          float2 w3 = __half22float2(*((const half2*)addr(W3, k)));

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
          float nw = __half2float(*addr(NW,k));


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
            float nw = __half2float(NW[k]);

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
            float nw = __half2float(NW[k]);

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

at::Tensor muillm_linear_activ_forward(
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor weights,
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

    muillm_gemv_norm_inputs_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      norm_weights.defined() ? (const half*)norm_weights.data_ptr() : nullptr,
      (const half*)weights.data_ptr(),
      (const half*)x.data_ptr(),
      activ,
      mul_bias.defined() ? (const half*)mul_bias.data_ptr() : nullptr,
      add_bias.defined() ? (const half*)add_bias.data_ptr() : nullptr,
      (half*)y.data_ptr(),
      N,
      K,
      epsilon,
      scale
    );
  } else {
    muillm_gemv_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      (const half*)weights.data_ptr(),
      (const half*)x.data_ptr(),
      activ,
      mul_bias.defined() ? (const half*)mul_bias.data_ptr() : nullptr,
      add_bias.defined() ? (const half*)add_bias.data_ptr() : nullptr,
      (half*)y.data_ptr(),
      N,
      K
    );
  }

  return y;
}