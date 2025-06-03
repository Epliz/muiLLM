#include "qkl2norm_kernels.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_BLOCK (2 * THREADS_PER_BLOCK)

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

// TODO: variance is computed by every block
//  each block scales and normalizes only a slice
__global__ void muillm_qkl2norm_kernel(
    const half* __restrict__ Q, // input = size BxK
    const half* __restrict__ K, // input = size BxK
    half* __restrict__ Q_NORM, // output = size BxK
    half* __restrict__ K_NORM, // output = size BxK
    float epsilon,
    unsigned BQ, // batch size for Q
    unsigned N,
    float scale // 1/K
) {
    int warpCounts = THREADS_PER_BLOCK / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    unsigned B = blockIdx.y;

    const half* __restrict__ X;
    half* __restrict__ Y;

    if (B < BQ) {
        // Q
        X = Q;
        Y = Q_NORM;
    } else {
        // K
        B -= BQ;
        X = K;
        Y = K_NORM;
    }

    // shared state to do the reductions
    __shared__ float shared_acc_var;

    if (threadIdx.x == 0) {
        shared_acc_var = epsilon;
    }
    __syncthreads();

    int current_row = B;
    // align X and Y to the current row
    X = &X[current_row * N];
    Y = &Y[current_row * N];

    // compute the variance (all blocks compute it fully)
    float acc_var = 0.f;
    {
      unsigned nStart = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x * 2;
      // first slice
      {
        unsigned n = nStart;
        for (; n + 1 < N; n += ELEMENTS_PER_BLOCK) {
          float2 x = __half22float2(*((const half2*)&X[n]));
          acc_var += x.x * x.x;
          acc_var += x.y * x.y;
        }
        if (n < N) {
          float x = __half2float(X[n]);
          acc_var += x * x;
        }
      }
      // second slice
      {
        unsigned n = threadIdx.x * 2;
        for (; n + 1 < nStart; n += ELEMENTS_PER_BLOCK) {
          float2 x = __half22float2(*((const half2*)&X[n]));
          acc_var += x.x * x.x;
          acc_var += x.y * x.y;
        }
        if (n < nStart) {
          float x = __half2float(X[n]);
          acc_var += x * x;
        }
      }

      // warp reduce
      acc_var = warpReduce(acc_var);
      // reduce accross warps
      if (laneId == 0) {
          atomicAdd(&shared_acc_var, acc_var);
      }
      __syncthreads();
    }

    // reload reduced sum and finalize variance by computing mean
    float rsqrt_var = rsqrtf(shared_acc_var * scale);

    // normalize & output
    {
      // one thread processes 2 elements
      unsigned n = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x * 2;
      if (n + 1 < N) {
        float2 x = __half22float2(*((const half2*)&X[n]));

        float yx = (x.x * rsqrt_var);
        float yy = (x.y * rsqrt_var);
        
        Y[n + 0] = __float2half(yx);
        Y[n + 1] = __float2half(yy);
      }
      if (n < N) {
        float x = __half2float(X[n]);

        float y = (x * rsqrt_var);
        
        Y[n] = __float2half(y);
      }
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<at::Tensor, at::Tensor> muillm_qkl2norm_forward(
    torch::Tensor q,
    torch::Tensor k,
    float epsilon) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);

  auto device = q.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  // q and k have different sizes, but the last dimension must be the same
  const auto NQ = q.size(q.dim() - 1);
  const auto NK = q.size(q.dim() - 1);
  TORCH_CHECK(NQ == NK, "The last dimension of q and k must be the same");
  const auto N = NQ; // last dimension size, same for q and k

  // batch size
  // TODO: is numel slow?
  const auto BQ = q.numel() / N;
  const auto BK = k.numel() / N;

  // q and k have different sizes
  auto qoutput_sizes = q.sizes().vec();
  auto koutput_sizes = k.sizes().vec();

  auto dtype = torch::kFloat16;
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto q_norm = torch::empty(qoutput_sizes, output_options);
  auto k_norm = torch::empty(koutput_sizes, output_options);

  const int threads_per_blocks = THREADS_PER_BLOCK;
  // launch enough blocks to cover all elements in q and k
  const dim3 num_blocks = dim3(DIV_ROUND_UP(N, ELEMENTS_PER_BLOCK), BQ+BK, 1);

  float scale = 1.f / N;

  muillm_qkl2norm_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const half*)q.data_ptr(),
    (const half*)k.data_ptr(),
    (half*)q_norm.data_ptr(),
    (half*)k_norm.data_ptr(),
    epsilon,
    BQ,
    N,
    scale
  );

  return std::make_tuple(q_norm, k_norm);
}