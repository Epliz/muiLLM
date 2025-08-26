
#include <hip/hip_fp16.h>

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
__global__ void muillm_rmsnorm_fp16_kernel(
    const half* __restrict__ W, // weight matrix - size K
    const half* __restrict__ X, // input = size BxK
    const half* __restrict__ RB, // optional residual = size BxK
    half* __restrict__ Y, // output = size BxK
    float epsilon,
    float weight_offset,
    unsigned K,
    float scale // 1/K
) {
    int warpCounts = THREADS_PER_BLOCK / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    // shared state to do the reductions
    __shared__ float shared_acc_var;

    if (threadIdx.x == 0) {
        shared_acc_var = epsilon;
    }
    __syncthreads();

    int current_row = blockIdx.y;

    bool has_residual = RB != nullptr;
    // align X, RB and Y to the current row
    X = &X[current_row * K];
    RB = has_residual ? &RB[current_row * K] : nullptr;
    Y = &Y[current_row * K];

    // compute the variance (all blocks compute it fully)
    float acc_var = 0.f;
    {
      unsigned kStart = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x * 2;
      // first slice
      {
        unsigned k = kStart;
        for (; k + 1 < K; k += ELEMENTS_PER_BLOCK) {
          float2 x = __half22float2(*((const half2*)&X[k]));
          acc_var += x.x * x.x;
          acc_var += x.y * x.y;
        }
        if (k < K) {
          float x = __half2float(X[k]);
          acc_var += x * x;
        }
      }
      // second slice
      {
        unsigned k = threadIdx.x * 2;
        for (; k + 1 < kStart; k += ELEMENTS_PER_BLOCK) {
          float2 x = __half22float2(*((const half2*)&X[k]));
          acc_var += x.x * x.x;
          acc_var += x.y * x.y;
        }
        if (k < kStart) {
          float x = __half2float(X[k]);
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
    if (W != nullptr) {
      // one thread processes 2 elements
      unsigned k = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x * 2;
      if (k + 1 < K) {
        float2 x = __half22float2(*((const half2*)&X[k]));
        float2 w = __half22float2(*((const half2*)&W[k])) + weight_offset;
        float2 r = has_residual ? __half22float2(*((const half2*)&RB[k])) : make_float2(0.f, 0.f);

        float yx = w.x * (x.x * rsqrt_var) + r.x;
        float yy = w.y * (x.y * rsqrt_var) + r.y;
        
        Y[k + 0] = __float2half(yx);
        Y[k + 1] = __float2half(yy);
      }
      if (k < K) {
        float x = __half2float(X[k]);
        float w = __half2float(W[k]) + weight_offset;
        float r = has_residual ? __half2float(RB[k]) : 0.f;

        float y = w * (x * rsqrt_var) + r;

        Y[k] = __float2half(y);
      }
    } else {
      // one thread processes 2 elements
      unsigned k = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x * 2;
      if (k + 1 < K) {
        float2 x = __half22float2(*((const half2*)&X[k]));
        float2 r = has_residual ? __half22float2(*((const half2*)&RB[k])) : make_float2(0.f, 0.f);

        float yx = (x.x * rsqrt_var) + r.x;
        float yy = (x.y * rsqrt_var) + r.y;
        
        Y[k + 0] = __float2half(yx);
        Y[k + 1] = __float2half(yy);
      }
      if (k < K) {
        float x = __half2float(X[k]);
        float r = has_residual ? __half2float(RB[k]) : 0.f;

        float y = (x * rsqrt_var) + r;

        Y[k] = __float2half(y);
      }
    }
}

void muillm_rmsnorm_fp16(
  hipStream_t stream,
  unsigned B,
  unsigned K,
  const half* __restrict__ W, // weight matrix - size K
  const half* __restrict__ X, // input = size BxK
  const half* __restrict__ RB, // optional residual = size BxK
  half* __restrict__ Y, // output = size BxK
  float epsilon,
  float weight_offset
) {

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const dim3 num_blocks = dim3(DIV_ROUND_UP(K, ELEMENTS_PER_BLOCK), B, 1);

  float scale = 1.f / K;

  muillm_rmsnorm_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    W,
    X,
    RB,
    Y,
    epsilon,
    weight_offset,
    K,
    scale
  );
}