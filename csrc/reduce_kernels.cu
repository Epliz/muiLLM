#include "reduce_kernels.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>


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
__global__ void muillm_reduce_sum_kernel(
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

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void muillm_reduce_sum_forward_placed_output(
    torch::Tensor x, // shape ()
    int dim,
    bool keep_dim,
    void* output_ptr,
    hipStream_t stream
) {
  CHECK_INPUT(x);

  int num_dims = x.dim();
  TORCH_CHECK(x.dtype() == torch::kFloat16, "Input tensor must be of type float16");
  TORCH_CHECK(num_dims >= 1, "Input tensor must have at least one dimension");

  if (dim < 0) {
    dim += num_dims; // convert negative dimension to positive
  }

  auto input_sizes = x.sizes();

  bool reduce_last_dim = (dim == num_dims - 1);

  int B, M, N;
  // calculate the number of blocks and threads
  if (reduce_last_dim) {
    // B can be 1
    B = 1;
    // M is the product of all dimensions before the last dimension
    M = 1;
    for (int i = 0; i < num_dims - 1; ++i) {
      M *= input_sizes[i];
    }
    // N is the last dimension
    N = input_sizes[num_dims - 1];

    // as many block as needed to cover all rows
    const int num_blocks_x = M;
    const int num_blocks_y = B;
    const dim3 num_blocks = dim3(num_blocks_x, num_blocks_y);

    int threads_per_blocks = THREADS_PER_BLOCK;

    // launch the kernel
    muillm_reduce_sum_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const half*) x.data_ptr(),
        (half*)output_ptr,
        M,
        N,
        reduce_last_dim
    );
  } else {
    // B is the product of all dimensions before the reduced dimension
    B = 1;
    for (int i = 0; i < dim; ++i) {
      B *= input_sizes[i];
    }
    // M is the size of the reduced dimension
    M = input_sizes[dim];
    // N is the product of all dimensions after the reduced dimension
    N = 1;
    for (int i = dim + 1; i < num_dims; ++i) {
      N *= input_sizes[i];
    }

    // as many block as needed to cover all columns
    const int num_blocks_x = DIV_ROUND_UP(N, THREADS_PER_BLOCK);
    const int num_blocks_y = B;
    const dim3 num_blocks = dim3(num_blocks_x, num_blocks_y);

    int threads_per_blocks = THREADS_PER_BLOCK;

    // launch the kernel
    muillm_reduce_sum_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const half*) x.data_ptr(),
        (half*)output_ptr,
        M,
        N,
        reduce_last_dim
    );

  }
}

at::Tensor muillm_reduce_sum_forward(
    torch::Tensor x,
    int dim,
    bool keep_dim
) {

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  if (dim < 0) {
    dim += x.dim();
  }

  TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension for reduction");

  // the output size is the same as input size except for the reduced dimension
  auto output_sizes = x.sizes().vec();
  if (!keep_dim) {
    TORCH_CHECK(x.dim() > 0, "Cannot remove dimension for a scalar tensor");
    // get the size of the output
    output_sizes.erase(output_sizes.begin() + dim); // remove the dimension
  } else {
    // keep dim
    output_sizes[dim] = 1; // reduce the dimension
  }

  auto dtype = torch::kFloat16;
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto y = torch::empty(output_sizes, output_options);

  muillm_reduce_sum_forward_placed_output(
      x,
      dim,
      keep_dim,
      y.data_ptr(),
      stream
  );

  return y;
}