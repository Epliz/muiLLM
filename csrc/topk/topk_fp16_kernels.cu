#include <hip/hip_fp16.h>

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_BLOCK 4096

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define FULL_MASK32 0xffffffff
#define FULL_MASK64 0xffffffffffffffff

#ifdef  __CUDA_ARCH__
#define __xx_shfl_down(mask, val, offset) __shfl_down_sync((mask), (val), (offset))
#elif defined(__HIP_PLATFORM_AMD__) // AMD
#define __xx_shfl_down(mask, val, offset) __shfl_down((val), (offset))
#else
#error "Unsupported compiler"
#endif

typedef struct value_index_pair {
  float value;
  int index;
} value_index_pair_t;

__device__ value_index_pair_t warpMax(float val, int index) {
  if (warpSize == 32) {
    for (int offset = 16; offset > 0; offset /= 2) {
      float other_val = __xx_shfl_down(FULL_MASK32, val, offset);
      int other_index = __xx_shfl_down(FULL_MASK32, index, offset);

      if (other_val > val) {
        val = other_val;
        index = other_index;
      }
    }
  }
  if (warpSize == 64) {
    for (int offset = 32; offset > 0; offset /= 2) {
      float other_val = __xx_shfl_down(FULL_MASK64, val, offset);
      float other_index = __xx_shfl_down(FULL_MASK64, index, offset);

      if (other_val > val) {
        val = other_val;
        index = other_index;
      }
    }
  }

  value_index_pair_t pair;
  pair.value = val;
  pair.index = index;
  return pair;
}

__device__ float sigmoid(float x) {
  // Sigmoid function: sigmoid(x) = 1 / (1 + exp(-x))
  return 1.0f / (1.0f + __expf(-x));
}

// finds the maximum value in a row and applies the sigmoid function
__global__ void muillm_max_sigmoid_fp16_kernel(
    const half* __restrict__ X, // input values = size BxM
    half* __restrict__ Y, // output values = size Bxk
    int64_t* __restrict__ indices, // output indices = size Bxk
    int M
) {
  int current_row = blockIdx.x;
  // align X and Y to the current row
  X = &X[current_row * M];
  Y = &Y[current_row];
  indices = &indices[current_row];

  int max_index = -1;
  half max_value = __float2half(-INFINITY);

  // compute the per thread max
  for (int i = threadIdx.x; i < M; i += THREADS_PER_BLOCK) {
    half value = X[i];
    if (__hgt(value, max_value)) {
      max_value = value;
      max_index = i;
    }
  }
  __syncthreads();
  // reduce the max value across threads in the block
  value_index_pair_t pair = warpMax(__half2float(max_value), max_index);

  // output the values and indices
  if (threadIdx.x == 0) {
    // apply the sigmoid function to the max value
    Y[0] = __float2half(sigmoid(pair.value));
    indices[0] = pair.index;
  }
}

__global__ void muillm_topk_sigmoid_fp16_kernel(
    const half* __restrict__ X, // input values = size BxM
    half* __restrict__ Y, // output values = size Bxk
    int64_t* __restrict__ indices, // output indices = size Bxk
    int M,
    int k
) {
  int current_row = blockIdx.x;
  // align X and Y to the current row
  X = &X[current_row * M];
  Y = &Y[current_row * k];
  indices = &indices[current_row * k];

  __shared__ half shared_values[ELEMENTS_PER_BLOCK];
  __shared__ int64_t shared_indices[ELEMENTS_PER_BLOCK];

  // compute the top K

  // the entire row of X fits into shared memory so load it all
  for (int i = threadIdx.x; i < M; i += THREADS_PER_BLOCK) {
    shared_values[i] = X[i];
    shared_indices[i] = i;
  }
  // get the next power of two greater than or equal to M
  int P = DIV_ROUND_UP(M, 2) * 2; // ensure M is even for bitonic sort

  // Fill remaining elements with negative infinity for proper sorting
  for (int i = threadIdx.x + M; i < P; i += THREADS_PER_BLOCK) {
    shared_values[i] = __float2half(-INFINITY);
    shared_indices[i] = -1;
  }
  __syncthreads();

  // sort the values in descending order
  // Bitonic sort - sort the values in descending order
  for (int size = 2; size <= P; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      int tid = threadIdx.x;
      
      for (int offset = 0; offset < ELEMENTS_PER_BLOCK; offset += THREADS_PER_BLOCK) {
        int idx = tid + offset;
        if (idx < ELEMENTS_PER_BLOCK) {
          int partner = idx ^ stride;
          
          if (partner < ELEMENTS_PER_BLOCK && idx < partner) {
            bool ascending = ((idx & size) == 0);
            
            // For descending order (top-k), we want larger values first
            bool should_swap = ascending ? 
              (__hlt(shared_values[idx], shared_values[partner])) :
              (__hgt(shared_values[idx], shared_values[partner]));
            
            if (should_swap) {
              // Swap values
              half temp_val = shared_values[idx];
              shared_values[idx] = shared_values[partner];
              shared_values[partner] = temp_val;
              
              // Swap indices
              int64_t temp_idx = shared_indices[idx];
              shared_indices[idx] = shared_indices[partner];
              shared_indices[partner] = temp_idx;
            }
          }
        }
      }
      __syncthreads();
    }
  }

  // apply the sigmoid function to the top k values
  for (int i = threadIdx.x; i < k; i += THREADS_PER_BLOCK) {
    // Apply sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
    shared_values[i] = __float2half(sigmoid(__half2float(shared_values[i])));
  }

  // output the values and indices
  for (int i = threadIdx.x; i < k; i += THREADS_PER_BLOCK) {
    Y[i] = shared_values[i];
    indices[i] = shared_indices[i];
  }
}

void muillm_topk_sigmoid_fp16(
    hipStream_t stream,
    const half* x, // input values = size BxM
    half* y, // output values = size Bxk
    int64_t* indices, // output indices = size Bxk
    int B,
    int M,
    int k
) {
  const int threads_per_blocks = THREADS_PER_BLOCK;

  if (k == 1) {
    // Special case for k == 1, we can use a simpler kernel
    muillm_max_sigmoid_fp16_kernel<<<B, threads_per_blocks, 0, stream>>>(
      (const half*)x,
      (half*)y,
      (int64_t*)indices,
      M
    );
  } else {
    muillm_topk_sigmoid_fp16_kernel<<<B, threads_per_blocks, 0, stream>>>(
      (const half*)x,
      (half*)y,
      (int64_t*)indices,
      M,
      k
    );
  }
}

