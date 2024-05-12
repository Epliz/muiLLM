#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

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

__global__ void muillm_gemv_kernel(
    const half* __restrict__ W, // weight matrix - size N x K
    const half* __restrict__ B, // optional bias - size N
    const half* __restrict__ X, // input = size K
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
        for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
          float w = __half2float(W_[k]);
          acc += w * __half2float(X[k]);
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
      if (B != nullptr) { // add the bias first if there is one
        acc += __half2float(B[current_row]);
      }

      // write the output value
      Y[current_row] = __float2half(acc);
    }
  }
}

at::Tensor muillm_linear_forward_cuda(
    torch::Tensor& weights,
    torch::Tensor* bias,
    torch::Tensor& x) {

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

  muillm_gemv_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const half*)weights.data_ptr(),
    bias == nullptr ? nullptr : (const half*)bias->data_ptr(),
    (const half*)x.data_ptr(),
    (half*)y.data_ptr(),
    N,
    K
  );

  return y;
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor muillm_linear_forward(
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor x) {
  //return torch::addmm(bias, x, weights.transpose(0, 1));
  return muillm_linear_forward_cuda(weights, &bias, x);
}

at::Tensor muillm_linear_forward_no_bias(
    torch::Tensor weights,
    torch::Tensor x) {
  CHECK_INPUT(weights);
  CHECK_INPUT(x);

  return muillm_linear_forward_cuda(weights, nullptr, x);
  //return torch::matmul(x, weights.transpose(0, 1));
}