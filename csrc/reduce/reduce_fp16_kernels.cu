#include <hip/hip_fp16.h>

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

// kernel to do the reduction
__global__ void muillm_reduce_sum_fp16_kernel(
    const half* __restrict__ X, // size BxMxN
    half* __restrict__ Y, // size BxMx1 (if we reduce columns) or Bx1xN (if we reduce rows)
    int M, // number of rows
    int N, // number of columns
    bool reduce_last_dim
) {
  int batch_idx = blockIdx.y;

  if (reduce_last_dim) {
    // reduce across columns (we have N columns)
    int warpCounts = THREADS_PER_BLOCK / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    // we use shared memory to accumulate results from all threads in a block
    __shared__ float shared_sum;

    if (threadIdx.x == 0) {
      shared_sum = 0.0f; // initialize shared memory
    }
    __syncthreads(); // ensure shared memory is initialized before we start

    int row_idx = blockIdx.x; // each block handles one row

    // realign X and Y
    X = &X[(batch_idx * M + row_idx) * N];
    Y = &Y[batch_idx * N + row_idx]; // output is Bx1xN

    // the entire block handles one row, we use warp reduction at the end
    float sum = 0.0f;
    for (int col_idx = threadIdx.x; col_idx < N; col_idx += THREADS_PER_BLOCK) {
      // TODO: vectorize by processing several columns with one thread
      sum += __half2float(X[col_idx]);
    }

    sum = warpReduce(sum);

    // accumulate into shared memory
    if (laneId == 0) {
      atomicAdd(&shared_sum, sum);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      // write the result to the output tensor
      Y[0] = __float2half(shared_sum);
    }
  } else {
    // we are reducing across rows (we have M rows)
    // each thread will handle one column
    int col_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x; // each block handles a group of columns

    // realign X and Y
    X = &X[batch_idx * M * N];
    Y = &Y[batch_idx * M]; // output is BxMx1

    // reduce across rows
    float sum = 0.0f;

    if (col_idx < N) {
      for (int row_idx = 0; row_idx < M; ++row_idx) {
        // TODO: vectorize by processing several columns with one thread
        sum += __half2float(X[row_idx * N + col_idx]);
      }

      // write the result to the output tensor
      Y[col_idx] = __float2half(sum);
    }

  }
}

void muillm_reduce_sum_fp16(
  hipStream_t stream,
  unsigned B,
  unsigned M,
  unsigned N,
  bool reduce_last_dim,
  const half* x,
  half* y
) {
  // calculate the number of blocks and threads
  if (reduce_last_dim) {
    // as many block as needed to cover all rows
    const int num_blocks_x = M;
    const int num_blocks_y = B;
    const dim3 num_blocks = dim3(num_blocks_x, num_blocks_y);

    int threads_per_blocks = THREADS_PER_BLOCK;

    // launch the kernel
    muillm_reduce_sum_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
        x,
        y,
        M,
        N,
        reduce_last_dim
    );
  } else {

    // as many block as needed to cover all columns
    const int num_blocks_x = DIV_ROUND_UP(N, THREADS_PER_BLOCK);
    const int num_blocks_y = B;
    const dim3 num_blocks = dim3(num_blocks_x, num_blocks_y);

    int threads_per_blocks = THREADS_PER_BLOCK;

    // launch the kernel
    muillm_reduce_sum_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
        x,
        y,
        M,
        N,
        reduce_last_dim
    );

  }
}