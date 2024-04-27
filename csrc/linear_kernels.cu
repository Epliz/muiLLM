#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

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

__global__ void muillm_gemv_kernel(
    const half* __restrict__ W, // size N x K
    const half* __restrict__ B, // size N
    const half* __restrict__ X, // size K
    half* __restrict__ Y, // size N
    unsigned N,
    unsigned K
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  // shared state to do the reductions
  __shared__ float shared_accs[ROWS_PER_BLOCK];
  __shared__ int shared_reduction_counters[ROWS_PER_BLOCK];

  if (laneId == 0) {
    shared_accs[warpId] = 0.f;
    shared_reduction_counters[warpId] = 0;
  }
  __syncthreads();

  for (int i = 0; i < ROWS_PER_BLOCK; i++) {
    // compute the t-th element of Y. by doing the dot product with the
    // t-th row of W
    int current_row = blockIdx.x * ROWS_PER_BLOCK + i;
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
      int old_count = atomicAdd(&shared_reduction_counters[i], 1);

      if (old_count == warpCounts - 1) {
        // we are the last warp to contribute
        // do the final write to memory

        acc = shared_accs[i]; // read the fully reduced value
        if (B != nullptr) { // add the bias first if there is one
          acc += __half2float(B[current_row]);
        }

        // write the output value
        Y[current_row] = __float2half(acc);
      }
    }
  }
}

#include <iostream>

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

  const int threads_per_blocks = 256;
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