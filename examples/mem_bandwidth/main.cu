#include <iostream>
#include <chrono>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

template <typename Func>
long long profile_time(unsigned warmup_reps, unsigned reps, Func func) {
    func(warmup_reps);

    // Get the start time
    auto start = std::chrono::high_resolution_clock::now();

    // Call the function to measure
    func(reps);

    // Get the end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    return duration.count();
}


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

template <typename T>
static inline const T* __device__ addr(const T* p, unsigned index) {
  // helps the AMDGPU compiler understand it can use the sgrp pair + single vgpr addressing mode
  unsigned byte_offset = sizeof(T) * index;
  const uint8_t* p8 = (const uint8_t*)p;
  return (const T*) (p8 + byte_offset);
}

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define ALIGN_UP(a, b) (DIV_ROUND_UP((a), (b)) * (b))

#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK 4

#define FLOAT_ELEMENTS_PER_THREAD 1
#define FLOAT_ELEMENTS_PER_BLOCK ((THREADS_PER_BLOCK) * (FLOAT_ELEMENTS_PER_THREAD))

__global__ void float_bandwidth_kernel(
  const float* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  int tid = blockIdx.x * FLOAT_ELEMENTS_PER_BLOCK + threadIdx.x;

  __shared__ float shared_acc;
  if (threadIdx.x == 0) {
    shared_acc = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  float r = 0.f;
  if (tid < N) {
    r = A[tid];
  }

  r = warpReduce(r);

  if (laneId == 0) {
    atomicAdd(&shared_acc, r);
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *out_flag = (shared_acc > 0.f);
  }
}

void float_bandwidth(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(N, FLOAT_ELEMENTS_PER_BLOCK);
  float_bandwidth_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (bool*)out_flag, N);
}

#define FLOAT2_ELEMENTS_PER_THREAD 2
#define FLOAT2_ELEMENTS_PER_BLOCK ((THREADS_PER_BLOCK) * (FLOAT2_ELEMENTS_PER_THREAD))

__global__ void float2_bandwidth_kernel(
  const float* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  int tid = blockIdx.x * FLOAT2_ELEMENTS_PER_BLOCK + (2 * threadIdx.x);

  __shared__ float shared_acc;
  if (threadIdx.x == 0) {
    shared_acc = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }


  float r = 0.f;
  if (tid + 1 < N) {
    float2 v = *reinterpret_cast<const float2*>(&A[tid]);
    r = v.x + v.y;
  }

  r = warpReduce(r);

  if (laneId == 0) {
    atomicAdd(&shared_acc, r);
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *out_flag = (shared_acc > 0.f);
  }
}

void float2_bandwidth(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(N, FLOAT2_ELEMENTS_PER_BLOCK);
  float2_bandwidth_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (bool*)out_flag, N);
}


#define FLOAT4_ELEMENTS_PER_THREAD 4
#define FLOAT4_ELEMENTS_PER_BLOCK ((THREADS_PER_BLOCK) * (FLOAT4_ELEMENTS_PER_THREAD))

__global__ void float4_bandwidth_kernel(
  const float* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  int tid = blockIdx.x * FLOAT4_ELEMENTS_PER_BLOCK + (4 * threadIdx.x);

  __shared__ float shared_acc;
  if (threadIdx.x == 0) {
    shared_acc = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }


  float r = 0.f;
  if (tid + 3 < N) {
    float4 v = *reinterpret_cast<const float4*>(&A[tid]);
    float vxy = (v.x + v.y);
    float vzw = (v.z + v.w);
    r = vxy + vzw;
  }

  r = warpReduce(r);

  if (laneId == 0) {
    atomicAdd(&shared_acc, r);
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *out_flag = (shared_acc > 0.f);
  }
}

void float4_bandwidth(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(N, FLOAT4_ELEMENTS_PER_BLOCK);
  float4_bandwidth_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (bool*)out_flag, N);
}


#define FLOAT8_ELEMENTS_PER_THREAD 8
#define FLOAT8_ELEMENTS_PER_BLOCK ((THREADS_PER_BLOCK) * (FLOAT8_ELEMENTS_PER_THREAD))

__global__ void float8_bandwidth_kernel(
  const float* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  int tid = blockIdx.x * FLOAT8_ELEMENTS_PER_BLOCK + (4 * threadIdx.x);

  __shared__ float shared_acc;
  if (threadIdx.x == 0) {
    shared_acc = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }


  float r = 0.f;
  if ((blockIdx.x + 1) * FLOAT8_ELEMENTS_PER_BLOCK <= N) {
    unsigned off = tid;
    float4 v0 = *reinterpret_cast<const float4*>(addr(A, off));
    off += 4 * THREADS_PER_BLOCK;
    float4 v1 = *reinterpret_cast<const float4*>(addr(A, off));

    float4 v01 = v0 + v1;

    r = v01.x + v01.y + v01.z + v01.w;
  }

  r = warpReduce(r);

  if (laneId == 0) {
    atomicAdd(&shared_acc, r);
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *out_flag = (shared_acc > 0.f);
  }
}

void float8_bandwidth(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(N, FLOAT8_ELEMENTS_PER_BLOCK);
  float8_bandwidth_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (bool*)out_flag, N);
}

typedef float mui_float4 __attribute__((vector_size(4)));

__device__ float4 load_nontemporal_float4(const float* p) {
  float x = __builtin_nontemporal_load(p);
  float y = __builtin_nontemporal_load(p + 1);
  float z = __builtin_nontemporal_load(p + 2);
  float w = __builtin_nontemporal_load(p + 3);

  float4 v = make_float4(x, y, z, w);
  return v;
}

__global__ void float8_nt_bandwidth_kernel(
  const float* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  int tid = blockIdx.x * FLOAT8_ELEMENTS_PER_BLOCK + (4 * threadIdx.x);

  __shared__ float shared_acc;
  if (threadIdx.x == 0) {
    shared_acc = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }


  float r = 0.f;
  if ((blockIdx.x + 1) * FLOAT8_ELEMENTS_PER_BLOCK <= N) {
    unsigned off = tid;
    float4 v0 = load_nontemporal_float4(addr(A, off));
    off += 4 * THREADS_PER_BLOCK;
    float4 v1 = load_nontemporal_float4(addr(A, off));

    float4 v01 = v0 + v1;

    r = (v01.x+ v01.y) + (v01.z + v01.w);
  }

  r = warpReduce(r);

  if (laneId == 0) {
    atomicAdd(&shared_acc, r);
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *out_flag = (shared_acc > 0.f);
  }
}

void float8_nt_bandwidth(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(N, FLOAT8_ELEMENTS_PER_BLOCK);
  float8_nt_bandwidth_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (bool*)out_flag, N);
}

#define FLOAT16_ELEMENTS_PER_THREAD 16
#define FLOAT16_ELEMENTS_PER_BLOCK ((THREADS_PER_BLOCK) * (FLOAT16_ELEMENTS_PER_THREAD))

__global__ void float16_bandwidth_kernel(
  const float* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  int tid = blockIdx.x * FLOAT16_ELEMENTS_PER_BLOCK + (4 * threadIdx.x);

  __shared__ float shared_acc;
  if (threadIdx.x == 0) {
    shared_acc = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }


  float r = 0.f;
  if ((blockIdx.x + 1) * FLOAT16_ELEMENTS_PER_BLOCK <= N) {
    unsigned off = tid;

    // first half
    float4 v0 = *reinterpret_cast<const float4*>(addr(A, off));
    off += 4 * THREADS_PER_BLOCK;
    float4 v1 = *reinterpret_cast<const float4*>(addr(A, off));

    float4 v01 = v0 + v1;

    // second half
    off += 4 * THREADS_PER_BLOCK;
    float4 v2 = *reinterpret_cast<const float4*>(addr(A, off));
    off += 4 * THREADS_PER_BLOCK;
    float4 v3 = *reinterpret_cast<const float4*>(addr(A, off));

    float4 v23 = v2 + v3;

    float4 v03 = v01 + v23;

    r += (v03.x + v03.y) + (v03.z + v03.w);
  }

  r = warpReduce(r);

  if (laneId == 0) {
    atomicAdd(&shared_acc, r);
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *out_flag = (shared_acc > 0.f);
  }
}

void float16_bandwidth(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(N, FLOAT16_ELEMENTS_PER_BLOCK);
  float16_bandwidth_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (bool*)out_flag, N);
}


#define FLOAT32_ELEMENTS_PER_THREAD 32
#define FLOAT32_ELEMENTS_PER_BLOCK ((THREADS_PER_BLOCK) * (FLOAT32_ELEMENTS_PER_THREAD))

__global__ void float32_bandwidth_kernel(
  const float* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  int tid = blockIdx.x * FLOAT32_ELEMENTS_PER_BLOCK + (4 * threadIdx.x);

  __shared__ float shared_acc;
  if (threadIdx.x == 0) {
    shared_acc = 0.f;
  }
  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }


  float r = 0.f;
  if ((blockIdx.x + 1) * FLOAT32_ELEMENTS_PER_BLOCK <= N) {
    unsigned off = tid;

    // first quarter
    float4 v0 = *reinterpret_cast<const float4*>(addr(A, off));
    off += 4 * THREADS_PER_BLOCK;
    float4 v1 = *reinterpret_cast<const float4*>(addr(A, off));

    float4 v01 = v0 + v1;

    // second quarter
    off += 4 * THREADS_PER_BLOCK;
    float4 v2 = *reinterpret_cast<const float4*>(addr(A, off));
    off += 4 * THREADS_PER_BLOCK;
    float4 v3 = *reinterpret_cast<const float4*>(addr(A, off));

    float4 v23 = v2 + v3;
    float4 v03 = v01 + v23;

    // third quarter
    off += 4 * THREADS_PER_BLOCK;
    float4 v4 = *reinterpret_cast<const float4*>(addr(A, off));
    off += 4 * THREADS_PER_BLOCK;
    float4 v5 = *reinterpret_cast<const float4*>(addr(A, off));

    float4 v45 = v4 + v5;


    // fourth quarter
    off += 4 * THREADS_PER_BLOCK;
    float4 v6 = *reinterpret_cast<const float4*>(addr(A, off));
    off += 4 * THREADS_PER_BLOCK;
    float4 v7 = *reinterpret_cast<const float4*>(addr(A, off));

    float4 v67 = v6 + v7;
    float4 v47 = v45 + v67;

    float4 v07 = v03 + v47;

    r += (v07.x + v07.y) + (v07.z + v07.w);
  }

  r = warpReduce(r);

  if (laneId == 0) {
    atomicAdd(&shared_acc, r);
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *out_flag = (shared_acc > 0.f);
  }
}

void float32_bandwidth(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(N, FLOAT32_ELEMENTS_PER_BLOCK);
  float32_bandwidth_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (bool*)out_flag, N);
}

typedef void (*KERNEL_FUNC_PTR)(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
);

struct NamedKernel {
  KERNEL_FUNC_PTR kernel;
  std::string name;
};

#define KERNEL_WITH_NAME(kernel) {kernel, #kernel}

int main(int argc, char** argv) {
  unsigned N;

  if (argc < 2) {
    std::cerr<<"Usage: "<<argv[0]<<" <N>"<<std::endl;
    return -1;
  }

  N = std::stoi(argv[1]);
  N = ALIGN_UP(N, 16);

  std::cout<<"Allocating memory..."<<std::endl;
  // allocate array on GPU
  float* A[4];

  for (int i = 0; i < 4; i++) {
    if (hipMalloc(&A[i], N * sizeof(float)) != hipSuccess) {
      std::cerr<<"Failed to allocate memory for A"<<std::endl;
      return -1;
    }

    if (hipMemset(A[i], 0, N * sizeof(float)) != hipSuccess) {
      std::cerr<<"Failed to initialize memory for A"<<std::endl;
      return -1;
    }
  }

  bool* out_flag;
  if (hipMalloc(&out_flag, sizeof(bool)) != hipSuccess) {
    std::cerr<<"Failed to allocate memory for out_flag"<<std::endl;
    return -1;
  }

  if (hipMemset(out_flag, 0, sizeof(bool)) != hipSuccess) {
    std::cerr<<"Failed to initialize memory for out_flag"<<std::endl;
    return -1;
  }

  std::cout<<"Executing..."<<std::endl;

  hipDeviceSynchronize();
  // Executing kernels
 
  std::vector<NamedKernel> kernels = {
    KERNEL_WITH_NAME(float_bandwidth),
    KERNEL_WITH_NAME(float2_bandwidth),
    KERNEL_WITH_NAME(float4_bandwidth),
    KERNEL_WITH_NAME(float8_bandwidth),
    KERNEL_WITH_NAME(float8_nt_bandwidth),
    KERNEL_WITH_NAME(float16_bandwidth),
    KERNEL_WITH_NAME(float32_bandwidth)
  };


  int warmup_reps = 10;
  int bench_reps = 10000;

  for (NamedKernel named_kernel: kernels) {
    uint64_t elasped_time = profile_time(warmup_reps, bench_reps,
    [=] (unsigned R) {
      for (int r = 0; r < R; r++) {
        named_kernel.kernel(
          A[r % 4], // rotate arrays to avoid caching effects
          out_flag,
          N
        );
      }

      hipDeviceSynchronize();
      }
    );

    std::cout<<named_kernel.name<<" DONE."<<std::endl;
    std::cout<<"time: "<<elasped_time / bench_reps<<"us/it"<<std::endl;
    std::cout<<"bandwidth: "<<(N * sizeof(float) * 1e-3) / (elasped_time / bench_reps)<<"GB/s"<<std::endl;
  }

  return 0;
}
