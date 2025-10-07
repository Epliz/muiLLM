#include <stdint.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <iostream>

#define MUILLM_MAX_GPUS 8

typedef enum muillm_comm_error {
  MUILLM_COMM_SUCCESS = 0,

  MUILLM_COMM_UNSUPPORTED_SIZE,

  MUILLM_COMM_SOCKET_CREATION_FAILED,
  MUILLM_COMM_SOCKET_BIND_FAILED,
  MUILLM_COMM_SOCKET_LISTEN_FAILED,
  MUILLM_COMM_SOCKET_ACCEPT_FAILED,
  MUILLM_COMM_SOCKET_CONNECT_FAILED,

  MUILLM_COMM_SOCKET_READ_ERROR,
  MUILLM_COMM_SOCKET_WRITE_ERROR,

  MUILLM_COMM_UNKNOWN_ERROR = -1
} muillm_comm_error_t;

#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_ULL 512

#define FULL_MASK32 0xffffffff
#define FULL_MASK64 0xffffffffffffffff

#ifdef  __CUDA_ARCH__
#define __xx_shfl(mask, val, offset) __shfl_sync(mask, val, offset)
#elif defined(__HIP_PLATFORM_AMD__) // AMD
#define __xx_shfl(mask, val, offset) __shfl(val, offset)
#else
#error "Unsupported compiler"
#endif


// warp broadcast using shuffle
__inline__ __device__ int __warp_broadcast(int val) {
  if (warpSize == 32) {
    val = __xx_shfl(FULL_MASK32, val, 0);
  }
  if (warpSize == 64) {
    val = __xx_shfl(FULL_MASK64, val, 0);
  }
  return val;
}

__inline__ __device__ int __block_broadcast(int val, int threads_per_block=THREADS_PER_BLOCK) {
  int lane_id = threadIdx.x % warpSize;

  if (threads_per_block > warpSize) {
    int __shared__ val_shared;
    if (threadIdx.x == 0) {
      val_shared = val;
    }
    __syncthreads();
    if (lane_id == 0) {
      val = val_shared;
    }
  }
  return __warp_broadcast(val);
}

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

typedef struct half8 {
  half x, y, z, w, a, b, c, d;
} half8;

typedef struct half4 {
  half x, y, z, w;
} half4;

__global__ void __muillm_inc_value_p2p_kernel(
  uint32_t* signal
) {
  if (threadIdx.x == 0) {
    atomicAdd_system(signal, 1);
    __threadfence_system();
  }
}

muillm_comm_error_t __mui_stream_inc_value(hipStream_t stream, uint32_t* signal) {
  __muillm_inc_value_p2p_kernel<<<1, 1, 0, stream>>>(signal);
  return MUILLM_COMM_SUCCESS;
}

__device__ void __do_inc_wait_value_p2p(
  volatile uint32_t* signal,
  uint32_t seq_no
) {
  if (threadIdx.x == 0) {
    // increment the value
    uint32_t value = atomicAdd_system((uint32_t*) signal, 1) + 1;
    __threadfence_system();

    // wait for the other ranks
    // we need the comparison to be >= as one GPU might already increment the value before all the other GPUs
    // have seen the previous one
    if (value < seq_no) {
      while (*signal < seq_no) __threadfence_system();
    }
  }
}

__global__ void __muillm_inc_wait_value_p2p_kernel(
  volatile uint32_t* signal,
  uint32_t seq_no
) {
  __do_inc_wait_value_p2p(signal, seq_no);
}

muillm_comm_error_t __mui_stream_inc_wait_value(hipStream_t stream, uint32_t* signal, uint32_t seq_no) {
  __muillm_inc_wait_value_p2p_kernel<<<1, 1, 0, stream>>>(signal, seq_no);
  return MUILLM_COMM_SUCCESS;
}



// each threads can copy 16 bytes
#define BYTES_PER_THREAD 16
#define BYTES_PER_BLOCK (THREADS_PER_BLOCK * BYTES_PER_THREAD)

typedef struct uint32x4{
uint32_t x, y, z, w;
} uint32x4_t;

__global__ void __muillm_copy_p2p_kernel(
  const uint8_t* src_ptr,
  uint8_t* dst_ptr,
  unsigned N
) {
  unsigned i = blockIdx.x * BYTES_PER_BLOCK + (threadIdx.x * BYTES_PER_THREAD);
  if (i + (BYTES_PER_THREAD - 1) < N) {
    // can copy 16 bytes

    const uint32x4_t* src_x16_ptr = (const uint32x4_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr = (uint32x4_t*)(&dst_ptr[i]);
    *dst_x16_ptr = *src_x16_ptr;

    i += BYTES_PER_THREAD;
  } else {
    // non vectorized copy
    for (unsigned b = 0; b < BYTES_PER_THREAD; b++) {
      if (i < N) {
        dst_ptr[i] = src_ptr[i];
        i++;
      }
    }
  }
}

muillm_comm_error_t __muillm_gpu_copy(void* dst, const void* src, size_t count, hipStream_t stream) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(count, BYTES_PER_BLOCK);

  // a copy kernel is faster than a hipMemcpyAsync
  __muillm_copy_p2p_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint8_t*) src,
    (uint8_t*) dst,
    count
  );

  if (hipPeekAtLastError() != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}