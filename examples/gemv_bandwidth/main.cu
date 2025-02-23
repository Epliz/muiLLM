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


#define FLOAT8_ELEMENTS_PER_THREAD 8
#define FLOAT8_ELEMENTS_PER_WARP (FLOAT8_ELEMENTS_PER_THREAD * warpSize)
#define FLOAT8_ELEMENTS_PER_BLOCK ((THREADS_PER_BLOCK) * (FLOAT8_ELEMENTS_PER_THREAD))


__global__ void float8_empty_bandwidth_kernel(
  const float* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  int tid = blockIdx.x * FLOAT8_ELEMENTS_PER_BLOCK + (4 * threadIdx.x);


  __shared__ float shared_accs[8];


  float r = 0.f;

  r = warpReduce(r);

  if (laneId == 0) {
    shared_accs[warpId] = r;
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float v = 0.f;
    if ((THREADS_PER_BLOCK / warpSize) == 4) {
      v = (shared_accs[0] + shared_accs[1]) + (shared_accs[2] + shared_accs[3]);
    }
    *out_flag = (v > 0.f);
  }
}

void float8_empty_bandwidth(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N,
  unsigned M
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(2* N * M, FLOAT8_ELEMENTS_PER_BLOCK);
  float8_empty_bandwidth_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (bool*)out_flag, 2 * N * M);
}

__global__ void float8_bandwidth_kernel(
  const float* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;
  int tid = blockIdx.x * FLOAT8_ELEMENTS_PER_BLOCK + (4 * threadIdx.x);


  __shared__ float shared_accs[8];


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
    shared_accs[warpId] = r;
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float v = 0.f;
    if ((THREADS_PER_BLOCK / warpSize) == 4) {
      v = (shared_accs[0] + shared_accs[1]) + (shared_accs[2] + shared_accs[3]);
    }
    *out_flag = (v > 0.f);
  }
}

void float8_bandwidth(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N,
  unsigned M
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(2 * N * M, FLOAT8_ELEMENTS_PER_BLOCK);
  float8_bandwidth_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (bool*)out_flag, 2* N * M);
}

// GEMV 
#define ROWS_PER_BLOCK 4

__global__ void float4_gemv_kernel(
  const float* __restrict__ A,
  const float* __restrict__ B,
  bool* __restrict__ out_flag,
  unsigned M // number of columns
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  int row = blockIdx.x * ROWS_PER_BLOCK;

  const float* __restrict__ A0 = &A[(row + 0) * M];
  const float* __restrict__ A1 = &A[(row + 1) * M];
  const float* __restrict__ A2 = &A[(row + 2) * M];
  const float* __restrict__ A3 = &A[(row + 3) * M];

  const float* __restrict__ B0 = &B[(row + 0) * M];
  const float* __restrict__ B1 = &B[(row + 1) * M];
  const float* __restrict__ B2 = &B[(row + 2) * M];
  const float* __restrict__ B3 = &B[(row + 3) * M];

  __shared__ float shared_accs_a[8][ROWS_PER_BLOCK];
  __shared__ float shared_accs_b[8][ROWS_PER_BLOCK];

  float a0 = 0.f;
  float a1 = 0.f;
  float a2 = 0.f;
  float a3 = 0.f;

  float b0 = 0.f;
  float b1 = 0.f;
  float b2 = 0.f;
  float b3 = 0.f;
  for (int off = (4 * threadIdx.x); off + 3 < M; off += 4 * THREADS_PER_BLOCK) {
    float4 v0 = *reinterpret_cast<const float4*>(addr(A0, off));
    float4 p0 = *reinterpret_cast<const float4*>(addr(B0, off));

    float4 v1 = *reinterpret_cast<const float4*>(addr(A1, off));
    float4 p1 = *reinterpret_cast<const float4*>(addr(B1, off));

    float4 v2 = *reinterpret_cast<const float4*>(addr(A2, off));
    float4 p2 = *reinterpret_cast<const float4*>(addr(B2, off));

    float4 v3 = *reinterpret_cast<const float4*>(addr(A3, off));
    float4 p3 = *reinterpret_cast<const float4*>(addr(B3, off));

    a0 += (v0.x + v0.z) + (v0.y + v0.w);
    a1 += (v1.x + v1.z) + (v1.y + v1.w);
    a2 += (v2.x + v2.z) + (v2.y + v2.w);
    a3 += (v3.x + v3.z) + (v3.y + v3.w);

    b0 += (p0.x + p0.z) + (p0.y + p0.w);
    b1 += (p1.x + p1.z) + (p1.y + p1.w);
    b2 += (p2.x + p2.z) + (p2.y + p2.w);
    b3 += (p3.x + p3.z) + (p3.y + p3.w);
  }

  a0 = warpReduce(a0);
  a1 = warpReduce(a1);
  a2 = warpReduce(a2);
  a3 = warpReduce(a3);

  b0 = warpReduce(b0);
  b1 = warpReduce(b1);
  b2 = warpReduce(b2);
  b3 = warpReduce(b3);

  if (laneId == 0) {
    shared_accs_a[warpId][0] = a0;
    shared_accs_a[warpId][1] = a1;
    shared_accs_a[warpId][2] = a2;
    shared_accs_a[warpId][3] = a3;
    shared_accs_b[warpId][0] = b0;
    shared_accs_b[warpId][1] = b1;
    shared_accs_b[warpId][2] = b2;
    shared_accs_b[warpId][3] = b3;
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x < ROWS_PER_BLOCK) {
    float v = 0.f;
    float p = 0.f;
    if ((THREADS_PER_BLOCK / warpSize) == 4) {
      v = (shared_accs_a[0][threadIdx.x] + shared_accs_a[1][threadIdx.x]) + (shared_accs_a[2][threadIdx.x] + shared_accs_a[3][threadIdx.x]);
      p = (shared_accs_b[0][threadIdx.x] + shared_accs_b[1][threadIdx.x]) + (shared_accs_b[2][threadIdx.x] + shared_accs_b[3][threadIdx.x]);
    }
    *out_flag = (v * p > 0.f);
  }
}

void float4_gemv(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N,
  unsigned M
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(N, ROWS_PER_BLOCK);
  float4_gemv_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (const float*) A + (N * M), (bool*)out_flag, M);
}


__global__ void float8_gemv_kernel(
  const float* __restrict__ A,
  const float* __restrict__ B,
  bool* __restrict__ out_flag,
  unsigned M // number of columns
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  int row = blockIdx.x * ROWS_PER_BLOCK;

  const float* __restrict__ A0 = &A[(row + 0) * M];
  const float* __restrict__ A1 = &A[(row + 1) * M];
  const float* __restrict__ A2 = &A[(row + 2) * M];
  const float* __restrict__ A3 = &A[(row + 3) * M];

  const float* __restrict__ B0 = &B[(row + 0) * M];
  const float* __restrict__ B1 = &B[(row + 1) * M];
  const float* __restrict__ B2 = &B[(row + 2) * M];
  const float* __restrict__ B3 = &B[(row + 3) * M];

  __shared__ float shared_accs_a[8][ROWS_PER_BLOCK];
  __shared__ float shared_accs_b[8][ROWS_PER_BLOCK];

  float a0 = 0.f;
  float a1 = 0.f;
  float a2 = 0.f;
  float a3 = 0.f;

  float b0 = 0.f;
  float b1 = 0.f;
  float b2 = 0.f;
  float b3 = 0.f;

  int off = (4 * threadIdx.x);
  for (; off + 4 * THREADS_PER_BLOCK + 3 < M; off += 8 * THREADS_PER_BLOCK) {
    float4 v0_0 = *reinterpret_cast<const float4*>(addr(A0, off));
    float4 v0_1 = *reinterpret_cast<const float4*>(addr(A0, off + 4 * THREADS_PER_BLOCK));
    float4 v0 = v0_0 + v0_1;
    float4 p0_0 = *reinterpret_cast<const float4*>(addr(B0, off));
    float4 p0_1 = *reinterpret_cast<const float4*>(addr(B0, off + 4 * THREADS_PER_BLOCK));
    float4 p0 = p0_0 + p0_1;

    float4 v1_0 = *reinterpret_cast<const float4*>(addr(A1, off));
    float4 v1_1 = *reinterpret_cast<const float4*>(addr(A1, off + 4 * THREADS_PER_BLOCK));
    float4 v1 = v1_0 + v1_1;
    float4 p1_0 = *reinterpret_cast<const float4*>(addr(B1, off));
    float4 p1_1 = *reinterpret_cast<const float4*>(addr(B1, off + 4 * THREADS_PER_BLOCK));
    float4 p1 = p1_0 + p1_1;

    float4 v2_0 = *reinterpret_cast<const float4*>(addr(A2, off));
    float4 v2_1 = *reinterpret_cast<const float4*>(addr(A2, off + 4 * THREADS_PER_BLOCK));
    float4 v2 = v2_0 + v2_1;
    float4 p2_0 = *reinterpret_cast<const float4*>(addr(B2, off));
    float4 p2_1 = *reinterpret_cast<const float4*>(addr(B2, off + 4 * THREADS_PER_BLOCK));
    float4 p2 = p2_0 + p2_1;

    float4 v3_0 = *reinterpret_cast<const float4*>(addr(A3, off));
    float4 v3_1 = *reinterpret_cast<const float4*>(addr(A3, off + 4 * THREADS_PER_BLOCK));
    float4 v3 = v3_0 + v3_1;
    float4 p3_0 = *reinterpret_cast<const float4*>(addr(B3, off));
    float4 p3_1 = *reinterpret_cast<const float4*>(addr(B3, off + 4 * THREADS_PER_BLOCK));
    float4 p3 = p3_0 + p3_1;

    a0 += (v0.x + v0.z) + (v0.y + v0.w);
    a1 += (v1.x + v1.z) + (v1.y + v1.w);
    a2 += (v2.x + v2.z) + (v2.y + v2.w);
    a3 += (v3.x + v3.z) + (v3.y + v3.w);

    b0 += (p0.x + p0.z) + (p0.y + p0.w);
    b1 += (p1.x + p1.z) + (p1.y + p1.w);
    b2 += (p2.x + p2.z) + (p2.y + p2.w);
    b3 += (p3.x + p3.z) + (p3.y + p3.w);
  }

  if (off + 3 < M) {
    float4 v0 = *reinterpret_cast<const float4*>(addr(A0, off));
    float4 p0 = *reinterpret_cast<const float4*>(addr(B0, off));

    float4 v1 = *reinterpret_cast<const float4*>(addr(A1, off));
    float4 p1 = *reinterpret_cast<const float4*>(addr(B1, off));

    float4 v2 = *reinterpret_cast<const float4*>(addr(A2, off));
    float4 p2 = *reinterpret_cast<const float4*>(addr(B2, off));

    float4 v3 = *reinterpret_cast<const float4*>(addr(A3, off));
    float4 p3 = *reinterpret_cast<const float4*>(addr(B3, off));

    a0 += (v0.x + v0.z) + (v0.y + v0.w);
    a1 += (v1.x + v1.z) + (v1.y + v1.w);
    a2 += (v2.x + v2.z) + (v2.y + v2.w);
    a3 += (v3.x + v3.z) + (v3.y + v3.w);

    b0 += (p0.x + p0.z) + (p0.y + p0.w);
    b1 += (p1.x + p1.z) + (p1.y + p1.w);
    b2 += (p2.x + p2.z) + (p2.y + p2.w);
    b3 += (p3.x + p3.z) + (p3.y + p3.w);
  }

  a0 = warpReduce(a0);
  a1 = warpReduce(a1);
  a2 = warpReduce(a2);
  a3 = warpReduce(a3);

  b0 = warpReduce(b0);
  b1 = warpReduce(b1);
  b2 = warpReduce(b2);
  b3 = warpReduce(b3);

  if (laneId == 0) {
    shared_accs_a[warpId][0] = a0;
    shared_accs_a[warpId][1] = a1;
    shared_accs_a[warpId][2] = a2;
    shared_accs_a[warpId][3] = a3;
    shared_accs_b[warpId][0] = b0;
    shared_accs_b[warpId][1] = b1;
    shared_accs_b[warpId][2] = b2;
    shared_accs_b[warpId][3] = b3;
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x < ROWS_PER_BLOCK) {
    float v = 0.f;
    float p = 0.f;
    if ((THREADS_PER_BLOCK / warpSize) == 4) {
      v = (shared_accs_a[0][threadIdx.x] + shared_accs_a[1][threadIdx.x]) + (shared_accs_a[2][threadIdx.x] + shared_accs_a[3][threadIdx.x]);
      p = (shared_accs_b[0][threadIdx.x] + shared_accs_b[1][threadIdx.x]) + (shared_accs_b[2][threadIdx.x] + shared_accs_b[3][threadIdx.x]);
    }
    *out_flag = (v * p > 0.f);
  }
}

void float8_gemv(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N,
  unsigned M
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(N, ROWS_PER_BLOCK);
  float8_gemv_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (const float*) A + (N * M), (bool*)out_flag, M);
}


__global__ void float8_r1_gemv_kernel(
  const float* __restrict__ A,
  const float* __restrict__ B,
  bool* __restrict__ out_flag,
  unsigned M // number of columns
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  int row = blockIdx.x;

  const float* __restrict__ A0 = &A[(row + 0) * M];
  const float* __restrict__ B0 = &B[(row + 0) * M];

  __shared__ float shared_accs_a[8][1];
  __shared__ float shared_accs_b[8][1];

  float a0 = 0.f;
  float b0 = 0.f;

  int off = (4 * threadIdx.x);
  for (; off + 4 * THREADS_PER_BLOCK + 3 < M; off += 8 * THREADS_PER_BLOCK) {
    float4 v0_0 = *reinterpret_cast<const float4*>(addr(A0, off));
    float4 v0_1 = *reinterpret_cast<const float4*>(addr(A0, off + 4 * THREADS_PER_BLOCK));

    float4 v0 = v0_0 + v0_1;

    a0 += (v0.x + v0.z) + (v0.y + v0.w);


    float4 p0_0 = *reinterpret_cast<const float4*>(addr(B0, off));
    float4 p0_1 = *reinterpret_cast<const float4*>(addr(B0, off + 4 * THREADS_PER_BLOCK));

    float4 p0 = p0_0 + p0_1;

    b0 += (p0.x + p0.z) + (p0.y + p0.w);
  }

  if (off + 3 < M) {
    float4 v0 = *reinterpret_cast<const float4*>(addr(A0, off));

    float4 p0 = *reinterpret_cast<const float4*>(addr(B0, off));

    a0 += (v0.x + v0.z) + (v0.y + v0.w);
    b0 += (p0.x + p0.z) + (p0.y + p0.w);
  }

  a0 = warpReduce(a0);
  b0 = warpReduce(b0);

  if (laneId == 0) {
    shared_accs_a[warpId][0] = a0;
    shared_accs_b[warpId][0] = b0;
  }

  if (THREADS_PER_BLOCK > warpSize) {
    __syncthreads();
  }

  if (threadIdx.x < 1) {
    float v = 0.f;
    float p = 0.f;
    if ((THREADS_PER_BLOCK / warpSize) == 4) {
      v = (shared_accs_a[0][threadIdx.x] + shared_accs_a[1][threadIdx.x]) + (shared_accs_a[2][threadIdx.x] + shared_accs_a[3][threadIdx.x]);
      p = (shared_accs_b[0][threadIdx.x] + shared_accs_b[1][threadIdx.x]) + (shared_accs_b[2][threadIdx.x] + shared_accs_b[3][threadIdx.x]);
    }
    *out_flag = (v * p > 0.f);
  }
}

void float8_r1_gemv(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N,
  unsigned M
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = N;
  float8_r1_gemv_kernel<<<num_blocks, threads_per_blocks>>>((const float*)A, (const float*) A + (N * M), (bool*)out_flag, M);
}

typedef void (*KERNEL_FUNC_PTR)(
  const void* __restrict__ A,
  bool* __restrict__ out_flag,
  unsigned N,
  unsigned M
);

struct NamedKernel {
  KERNEL_FUNC_PTR kernel;
  std::string name;
};

#define KERNEL_WITH_NAME(kernel) {kernel, #kernel}

int main(int argc, char** argv) {
  unsigned N; // number of rows
  unsigned M; // number of columns

  if (argc < 3) {
    std::cerr<<"Usage: "<<argv[0]<<" <N> <M>"<<std::endl;
    return -1;
  }

  N = std::stoi(argv[1]);
  N = ALIGN_UP(N, 16);
  M = std::stoi(argv[2]);
  M = ALIGN_UP(M, 16);

  std::cout<<"Allocating memory..."<<std::endl;
  // allocate arrays on GPU
  float* A[4];

  for (int i = 0; i < 4; i++) {
    if (hipMalloc(&A[i], 2 * N * M * sizeof(float)) != hipSuccess) {
      std::cerr<<"Failed to allocate memory for A"<<std::endl;
      return -1;
    }

    if (hipMemset(A[i], 0, 2 * N * M * sizeof(float)) != hipSuccess) {
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
    KERNEL_WITH_NAME(float8_empty_bandwidth),
    KERNEL_WITH_NAME(float8_bandwidth),
    KERNEL_WITH_NAME(float4_gemv),
    KERNEL_WITH_NAME(float8_gemv),
    KERNEL_WITH_NAME(float8_r1_gemv)
  };


  int warmup_reps = 100;
  int bench_reps = 10000;

  for (NamedKernel named_kernel: kernels) {
    uint64_t elasped_time = profile_time(warmup_reps, bench_reps,
    [=] (unsigned R) {
      for (int r = 0; r < R; r++) {
        named_kernel.kernel(
          A[r % 4], // rotate arrays to avoid caching effects
          out_flag,
          N,
          M
        );
      }

      hipDeviceSynchronize();
      }
    );

    std::cout<<named_kernel.name<<" DONE."<<std::endl;
    std::cout<<"time: "<<elasped_time / bench_reps<<"us/it"<<std::endl;
    std::cout<<"bandwidth: "<<(2 * N * M * sizeof(float) * 1e-3) / (elasped_time / bench_reps)<<"GB/s"<<std::endl;
  }

  return 0;
}
