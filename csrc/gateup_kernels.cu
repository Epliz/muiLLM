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

static inline float __device__ silu(float x) {
  return x / (1.0f + expf(-x));
}
__global__ void muillm_gateupsilu_gemv_kernel(
    const half* __restrict__ GW, // weight matrix - size N x K
    const half* __restrict__ UW, // weight matrix - size N x K
    const half* __restrict__ X, // input = size K
    half* __restrict__ Y, // output - size N
    unsigned N,
    unsigned K
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
      const half* GW0 = &GW[(current_row + 0) * K];
      const half* GW1 = &GW[(current_row + 1) * K];
      const half* GW2 = &GW[(current_row + 2) * K];
      const half* GW3 = &GW[(current_row + 3) * K];

      float gacc0 = 0.f;
      float gacc1 = 0.f;
      float gacc2 = 0.f;
      float gacc3 = 0.f;

      // do the dot product
      {
        unsigned k; // should be 2 * tidx ?
        //*
        for (k = threadIdx.x * 2; k + 1 < K; k += (THREADS_PER_BLOCK * 2)) {
          // vectorized
          float2 x = __half22float2(*((const half2*)addr(X, k)));

          float2 gw0 = __half22float2(*((const half2*)addr(GW0, k)));
          float2 gw1 = __half22float2(*((const half2*)addr(GW1, k)));
          float2 gw2 = __half22float2(*((const half2*)addr(GW2, k)));
          float2 gw3 = __half22float2(*((const half2*)addr(GW3, k)));

          dot2(gacc0, gw0, x);
          dot2(gacc1, gw1, x);
          dot2(gacc2, gw2, x);
          dot2(gacc3, gw3, x);
        }

        if (k < K) {
          // remainder
          float x = __half2float(*addr(X,k));

          float gw0 = __half2float(*addr(GW0,k));
          float gw1 = __half2float(*addr(GW1,k));
          float gw2 = __half2float(*addr(GW2,k));
          float gw3 = __half2float(*addr(GW3,k));
          
          gacc0 += gw0 * x;
          gacc1 += gw1 * x;
          gacc2 += gw2 * x;
          gacc3 += gw3 * x;
        }
      }

      // warp reduce
      gacc0 = warpReduce(gacc0);
      gacc1 = warpReduce(gacc1);
      gacc2 = warpReduce(gacc2);
      gacc3 = warpReduce(gacc3);

      // reduce accross warps
      if (laneId == 0) {
        atomicAdd(&shared_gaccs[0], gacc0);
        atomicAdd(&shared_gaccs[1], gacc1);
        atomicAdd(&shared_gaccs[2], gacc2);
        atomicAdd(&shared_gaccs[3], gacc3);
      }
    } else {
      for (int i = 0; i < ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const half* GW_ = &GW[current_row * K];
      
        // do the dot product
        float gacc = 0.f;
        for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
          float x =  __half2float(X[k]);
          float gw = __half2float(GW_[k]);
          gacc += gw * x;
        }

        // warp reduce
        gacc = warpReduce(gacc);

        // reduce accross warps
        if (laneId == 0) {
          atomicAdd(&shared_gaccs[i], gacc);
        }
      }
    }

    // UW
    if (current_row + 3 < N) {
      // compute the t-th element of Y. by doing the dot product with the
      // t-th row of W
      const half* UW0 = &UW[(current_row + 0) * K];
      const half* UW1 = &UW[(current_row + 1) * K];
      const half* UW2 = &UW[(current_row + 2) * K];
      const half* UW3 = &UW[(current_row + 3) * K];

      float uacc0 = 0.f;
      float uacc1 = 0.f;
      float uacc2 = 0.f;
      float uacc3 = 0.f;

      // do the dot product
      {
        unsigned k; // should be 2 * tidx ?
        //*
        for (k = threadIdx.x * 2; k + 1 < K; k += (THREADS_PER_BLOCK * 2)) {
          // vectorized
          float2 x = __half22float2(*((const half2*)addr(X, k)));

          float2 uw0 = __half22float2(*((const half2*)addr(UW0, k)));
          float2 uw1 = __half22float2(*((const half2*)addr(UW1, k)));
          float2 uw2 = __half22float2(*((const half2*)addr(UW2, k)));
          float2 uw3 = __half22float2(*((const half2*)addr(UW3, k)));
      
          dot2(uacc0, uw0, x);
          dot2(uacc1, uw1, x);
          dot2(uacc2, uw2, x);
          dot2(uacc3, uw3, x);
        }

        if (k < K) {
          // remainder
          float x = __half2float(*addr(X,k));

          float uw0 = __half2float(*addr(UW0,k));
          float uw1 = __half2float(*addr(UW1,k));
          float uw2 = __half2float(*addr(UW2,k));
          float uw3 = __half2float(*addr(UW3,k));

          uacc0 += uw0 * x;
          uacc1 += uw1 * x;
          uacc2 += uw2 * x;
          uacc3 += uw3 * x;
        }
      }

      // warp reduce

      uacc0 = warpReduce(uacc0);
      uacc1 = warpReduce(uacc1);
      uacc2 = warpReduce(uacc2);
      uacc3 = warpReduce(uacc3);

      // reduce accross warps
      if (laneId == 0) {
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

        const half* UW_ = &UW[current_row * K];
      
        // do the dot product
        float uacc = 0.f;
        for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
          float x =  __half2float(X[k]);
          float uw = __half2float(UW_[k]);
          uacc += uw * x;
        }

        // warp reduce
        uacc = warpReduce(uacc);

        // reduce accross warps
        if (laneId == 0) {
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

__global__ void muillm_gateupsilu_gemv_norm_inputs_kernel(
    const half* __restrict__ NW, // input normalization weights matrix - size K
    const half* __restrict__ GW, // weight matrix - size N x K
    const half* __restrict__ UW, // weight matrix - size N x K
    const half* __restrict__ X, // input = size K
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
      const half* GW0 = &GW[(current_row + 0) * K];
      const half* GW1 = &GW[(current_row + 1) * K];
      const half* GW2 = &GW[(current_row + 2) * K];
      const half* GW3 = &GW[(current_row + 3) * K];

      float gacc0 = 0.f;
      float gacc1 = 0.f;
      float gacc2 = 0.f;
      float gacc3 = 0.f;

      // do the dot product
      {
        unsigned k; // should be 2 * tidx ?
        //*
        for (k = threadIdx.x * 2; k + 1 < K; k += (THREADS_PER_BLOCK * 2)) {
          // vectorized
          float2 x = __half22float2(*((const half2*)addr(X, k)));
          float2 nw = __half22float2(*((const half2*)addr(NW, k)));

          float2 gw0 = __half22float2(*((const half2*)addr(GW0, k)));
          float2 gw1 = __half22float2(*((const half2*)addr(GW1, k)));
          float2 gw2 = __half22float2(*((const half2*)addr(GW2, k)));
          float2 gw3 = __half22float2(*((const half2*)addr(GW3, k)));

          // accumulate for the variance
          dot2(var_x, x, x);

          // multiply with normalization weights
          x.x = x.x * nw.x;
          x.y = x.y * nw.y;

          dot2(gacc0, gw0, x);
          dot2(gacc1, gw1, x);
          dot2(gacc2, gw2, x);
          dot2(gacc3, gw3, x);
        }

        if (k < K) {
          // remainder
          float x = __half2float(*addr(X,k));
          float nw = __half2float(*addr(NW,k));

          float gw0 = __half2float(*addr(GW0,k));
          float gw1 = __half2float(*addr(GW1,k));
          float gw2 = __half2float(*addr(GW2,k));
          float gw3 = __half2float(*addr(GW3,k));
          
          // accumulate for the variance
          var_x += x * x;

          // multiply with normalization weights
          x *= nw;

          gacc0 += gw0 * x;
          gacc1 += gw1 * x;
          gacc2 += gw2 * x;
          gacc3 += gw3 * x;
        }
      }

      // warp reduce
      var_x = warpReduce(var_x);
      gacc0 = warpReduce(gacc0);
      gacc1 = warpReduce(gacc1);
      gacc2 = warpReduce(gacc2);
      gacc3 = warpReduce(gacc3);

      // reduce accross warps
      if (laneId == 0) {
        atomicAdd(&shared_var_x, var_x);
        atomicAdd(&shared_gaccs[0], gacc0);
        atomicAdd(&shared_gaccs[1], gacc1);
        atomicAdd(&shared_gaccs[2], gacc2);
        atomicAdd(&shared_gaccs[3], gacc3);
      }
    } else {
      for (int i = 0; i < ROWS_PER_BLOCK; i++) {
        // compute the t-th element of Y. by doing the dot product with the
        // t-th row of W
        int current_row = blockIdx.x * ROWS_PER_BLOCK + i;

        if (current_row >= N)
          break;

        const half* GW_ = &GW[current_row * K];
      
        // do the dot product
        float gacc = 0.f;
        if (i == 0) {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float x =  __half2float(X[k]);
            float nw = __half2float(NW[k]);

            // accumuate the variance
            var_x += x * x;

            // multiply with normalization weights
            x *= nw;

            float gw = __half2float(GW_[k]);
            gacc += gw * x;
          }
        } else {
          for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
            float x =  __half2float(X[k]);
            float nw = __half2float(NW[k]);

            // don't accumulate the variance (we already have done it with i == 0)

            // multiply with normalization weights
            x *= nw;

            float gw = __half2float(GW_[k]);
            gacc += gw * x;
          }
        }

        // warp reduce
        var_x = warpReduce(var_x);
        gacc = warpReduce(gacc);

        // reduce accross warps
        if (laneId == 0) {
          atomicAdd(&shared_var_x, var_x);
          atomicAdd(&shared_gaccs[i], gacc);
        }
      }
    }

    // UW
    if (current_row + 3 < N) {
      // compute the t-th element of Y. by doing the dot product with the
      // t-th row of W
      const half* UW0 = &UW[(current_row + 0) * K];
      const half* UW1 = &UW[(current_row + 1) * K];
      const half* UW2 = &UW[(current_row + 2) * K];
      const half* UW3 = &UW[(current_row + 3) * K];

      float uacc0 = 0.f;
      float uacc1 = 0.f;
      float uacc2 = 0.f;
      float uacc3 = 0.f;

      // do the dot product
      {
        unsigned k; // should be 2 * tidx ?
        //*
        for (k = threadIdx.x * 2; k + 1 < K; k += (THREADS_PER_BLOCK * 2)) {
          // vectorized
          float2 x = __half22float2(*((const half2*)addr(X, k)));
          float2 nw = __half22float2(*((const half2*)addr(NW, k)));

          float2 uw0 = __half22float2(*((const half2*)addr(UW0, k)));
          float2 uw1 = __half22float2(*((const half2*)addr(UW1, k)));
          float2 uw2 = __half22float2(*((const half2*)addr(UW2, k)));
          float2 uw3 = __half22float2(*((const half2*)addr(UW3, k)));
      
          // multiply with normalization weights
          x.x = x.x * nw.x;
          x.y = x.y * nw.y;

          dot2(uacc0, uw0, x);
          dot2(uacc1, uw1, x);
          dot2(uacc2, uw2, x);
          dot2(uacc3, uw3, x);
        }

        if (k < K) {
          // remainder
          float x = __half2float(*addr(X,k));
          float nw = __half2float(*addr(NW,k));

          float uw0 = __half2float(*addr(UW0,k));
          float uw1 = __half2float(*addr(UW1,k));
          float uw2 = __half2float(*addr(UW2,k));
          float uw3 = __half2float(*addr(UW3,k));

          // multiply with normalization weights
          x *= nw;

          uacc0 += uw0 * x;
          uacc1 += uw1 * x;
          uacc2 += uw2 * x;
          uacc3 += uw3 * x;
        }
      }

      // warp reduce

      uacc0 = warpReduce(uacc0);
      uacc1 = warpReduce(uacc1);
      uacc2 = warpReduce(uacc2);
      uacc3 = warpReduce(uacc3);

      // reduce accross warps
      if (laneId == 0) {
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

        const half* UW_ = &UW[current_row * K];
      
        // do the dot product
        float uacc = 0.f;
        for (int k = threadIdx.x; k < K; k += THREADS_PER_BLOCK) {
          float x =  __half2float(X[k]);
          float nw = __half2float(NW[k]);

          // don't accumulate the variance (we already have done it with i == 0)

          // multiply with normalization weights
          x *= nw;
  
          float uw = __half2float(UW_[k]);
          uacc += uw * x;
        }

        // warp reduce
        uacc = warpReduce(uacc);

        // reduce accross warps
        if (laneId == 0) {
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

at::Tensor muillm_gateupsilu_forward(
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor x) {
  bool normalize = norm_weights.defined();
  if (normalize) {
    CHECK_INPUT(norm_weights);
  }
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(up_weights);
  CHECK_INPUT(x);


  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto N = gate_weights.size(0);
  const auto K = gate_weights.size(1);

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

    muillm_gateupsilu_gemv_norm_inputs_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      norm_weights.defined() ? (const half*)norm_weights.data_ptr() : nullptr,
      (const half*)gate_weights.data_ptr(),
      (const half*)up_weights.data_ptr(),
      (const half*)x.data_ptr(),
      (half*)y.data_ptr(),
      N,
      K,
      epsilon,
      scale
    );
  } else {

    muillm_gateupsilu_gemv_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      (const half*)gate_weights.data_ptr(),
      (const half*)up_weights.data_ptr(),
      (const half*)x.data_ptr(),
      (half*)y.data_ptr(),
      N,
      K
    );
  }

  return y;
}