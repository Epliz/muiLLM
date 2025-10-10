#include <stdint.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <iostream>
#include <algorithm>

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


typedef enum muillm_comm_datatype {
  MUILLM_COMM_BOOL = 0,
  MUILLM_COMM_INT8,
  MUILLM_COMM_INT16,
  MUILLM_COMM_INT32,
  MUILLM_COMM_INT64,
  MUILLM_COMM_FP16,
  MUILLM_COMM_BF16,
  MUILLM_COMM_FP32,
  MUILLM_COMM_FP64
} muillm_comm_datatype_t;


// returns the size in bytes for the given datatype and number of elements
static inline size_t __comm_size(
    muillm_comm_datatype_t datatype,
    size_t count
) {
  switch (datatype) {
    case MUILLM_COMM_BOOL: {
      return 1 * count;
    }
    case MUILLM_COMM_INT8: {
      return 1 * count;
    }
    case MUILLM_COMM_INT16: {
      return 2 * count;
    }
    case MUILLM_COMM_INT32: {
      return 4 * count;
    }
    case MUILLM_COMM_INT64: {
      return 8 * count;
    }
    case MUILLM_COMM_FP16: {
      return 2 * count;
    }
    case MUILLM_COMM_BF16: {
      return 2 * count;
    }
    case MUILLM_COMM_FP32: {
      return 4 * count;
    }
    case MUILLM_COMM_FP64: {
      return 8 * count;
    }
    default: {
      return 0;
    }
  }
}

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
#define ALIGN_UP(a, b) (DIV_ROUND_UP(a, b) * (b))

typedef struct half8 {
  half x, y, z, w, a, b, c, d;
} half8;

typedef struct half4 {
  half x, y, z, w;
} half4;

typedef struct __hip_bfloat164 {
  __hip_bfloat16 x, y, z, w;
} __hip_bfloat164;

typedef struct __hip_bfloat168 {
  __hip_bfloat16 x, y, z, w, a, b, c, d;
} __hip_bfloat168;


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
#define BYTES_PER_BLOCK_LOOP (THREADS_PER_BLOCK * BYTES_PER_THREAD)

typedef struct uint32x4{
uint32_t x, y, z, w;
} uint32x4_t;

__global__ void __muillm_copy_p2p_kernel(
  const uint8_t* src_ptr,
  uint8_t* dst_ptr,
  unsigned N
) {
  unsigned i = blockIdx.x * BYTES_PER_BLOCK_LOOP + (threadIdx.x * BYTES_PER_THREAD);
  if (i + (BYTES_PER_THREAD - 1) < N) {
    // can copy 16 bytes

    const uint32x4_t* src_x16_ptr = (const uint32x4_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr = (uint32x4_t*)(&dst_ptr[i]);
    *dst_x16_ptr = *src_x16_ptr;

    i += BYTES_PER_BLOCK_LOOP;
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
  const int num_blocks = DIV_ROUND_UP(count, BYTES_PER_BLOCK_LOOP);

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

void __global__ scatter_all_tp8_kernel(
  const uint8_t* src,
  uint8_t* dst0,
  uint8_t* dst1,
  uint8_t* dst2,
  uint8_t* dst3,
  uint8_t* dst4,
  uint8_t* dst5,
  uint8_t* dst6,
  uint8_t* dst7,
  int bytes_per_block,
  int N,
  int local_rank
) {
  unsigned block_idx = blockIdx.y;

  unsigned block_start = blockIdx.x * bytes_per_block;
  unsigned block_end = std::min(block_start + bytes_per_block, N);
  unsigned i = block_start + (threadIdx.x * BYTES_PER_THREAD);

  uint8_t* dsts[8] = {dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7};
  uint8_t* dst = dsts[block_idx];

  // realign src and dst pointers according to the block we are working on
  src += block_idx * N;
  dst += local_rank * N;

  // TODO: unroll more, e.g. 4x more
  for (; i + (BYTES_PER_THREAD - 1) < block_end; i += BYTES_PER_BLOCK_LOOP) {
    // can copy 16 bytes, which is 4kB per iteration
    const uint32x4_t* src_x16_ptr = (const uint32x4_t*)(&src[i]);
    uint32x4_t* dst_x16_ptr = (uint32x4_t*)(&dst[i]);

    uint32x4_t v = *src_x16_ptr;
    *dst_x16_ptr = v;
  }
  
  // loop remainder
  if (i < block_end) {
    // only one thread will execute this at max
    // non vectorized copy
    for (unsigned b = 0; b < BYTES_PER_THREAD; b++) {
      if (i < block_end) {
        uint8_t v = src[i];
        dst[i] = v;
        i++;
      }
    }
  }
}

#define MAX_REDUCE_X_BLOCKS 8

muillm_comm_error_t __muillm_scatter_all(
  hipStream_t stream,
  // inputs
  const void* src,
  int scattered_size_bytes,
  int local_size,
  int local_rank,
  // outputs
  void* dst0,
  void* dst1,
  void* dst2,
  void* dst3,
  void* dst4,
  void* dst5,
  void* dst6,
  void* dst7
) {

  const int threads_per_blocks = THREADS_PER_BLOCK;
  // we want to avoid spawning too many blocks to copy the data and want instead
  // to make blocks process more data when we have more than MAX_REDUCE_X_BLOCKS
  //
  int num_small_x_blocks = DIV_ROUND_UP(scattered_size_bytes, BYTES_PER_BLOCK_LOOP);
  int num_x_blocks = std::min(num_small_x_blocks, MAX_REDUCE_X_BLOCKS);
  const dim3 num_blocks = dim3(num_x_blocks, local_size);

  int bytes_per_block = ALIGN_UP(DIV_ROUND_UP(scattered_size_bytes, num_x_blocks), 4096);

  scatter_all_tp8_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint8_t*) src,
    (uint8_t*) dst0,
    (uint8_t*) dst1,
    (uint8_t*) dst2,
    (uint8_t*) dst3,
    (uint8_t*) dst4,
    (uint8_t*) dst5,
    (uint8_t*) dst6,
    (uint8_t*) dst7,
    bytes_per_block,
    scattered_size_bytes,
    local_rank
  );

  return MUILLM_COMM_SUCCESS;
}

#define REDUCE_PER_THREAD 8
#define REDUCE_PER_BLOCK (THREADS_PER_BLOCK * REDUCE_PER_THREAD)

void __global__ reduce_x8_fp16_kernel(
  const half* __restrict__ src0, // shape [scattered_M, N]
  const half* __restrict__ src1, // shape [scattered_M, N]
  const half* __restrict__ src2, // shape [scattered_M, N]
  const half* __restrict__ src3, // shape [scattered_M, N]
  const half* __restrict__ src4, // shape [scattered_M, N]
  const half* __restrict__ src5, // shape [scattered_M, N]
  const half* __restrict__ src6, // shape [scattered_M, N]
  const half* __restrict__ src7, // shape [scattered_M, N]
  half* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    half8* dst_h8_ptr = (half8*)(&dst[i]);

    const half8* src0_h8_ptr = (const half8*)(&src0[i]);
    const half8* src1_h8_ptr = (const half8*)(&src1[i]);
    const half8* src2_h8_ptr = (const half8*)(&src2[i]);
    const half8* src3_h8_ptr = (const half8*)(&src3[i]);
    const half8* src4_h8_ptr = (const half8*)(&src4[i]);
    const half8* src5_h8_ptr = (const half8*)(&src5[i]);
    const half8* src6_h8_ptr = (const half8*)(&src6[i]);
    const half8* src7_h8_ptr = (const half8*)(&src7[i]);

    half8 v0 = *src0_h8_ptr;
    half8 v1 = *src1_h8_ptr;
    half8 v2 = *src2_h8_ptr;
    half8 v3 = *src3_h8_ptr;
    half8 v4 = *src4_h8_ptr;
    half8 v5 = *src5_h8_ptr;
    half8 v6 = *src6_h8_ptr;
    half8 v7 = *src7_h8_ptr;

    half8 r;
    r.x = __hadd(v0.x, __hadd(v1.x, __hadd(v2.x, __hadd(v3.x, __hadd(v4.x, __hadd(v5.x, __hadd(v6.x, v7.x)))))));
    r.y = __hadd(v0.y, __hadd(v1.y, __hadd(v2.y, __hadd(v3.y, __hadd(v4.y, __hadd(v5.y, __hadd(v6.y, v7.y)))))));
    r.z = __hadd(v0.z, __hadd(v1.z, __hadd(v2.z, __hadd(v3.z, __hadd(v4.z, __hadd(v5.z, __hadd(v6.z, v7.z)))))));
    r.w = __hadd(v0.w, __hadd(v1.w, __hadd(v2.w, __hadd(v3.w, __hadd(v4.w, __hadd(v5.w, __hadd(v6.w, v7.w)))))));
    r.a = __hadd(v0.a, __hadd(v1.a, __hadd(v2.a, __hadd(v3.a, __hadd(v4.a, __hadd(v5.a, __hadd(v6.a, v7.a)))))));
    r.b = __hadd(v0.b, __hadd(v1.b, __hadd(v2.b, __hadd(v3.b, __hadd(v4.b, __hadd(v5.b, __hadd(v6.b, v7.b)))))));
    r.c = __hadd(v0.c, __hadd(v1.c, __hadd(v2.c, __hadd(v3.c, __hadd(v4.c, __hadd(v5.c, __hadd(v6.c, v7.c)))))));
    r.d = __hadd(v0.d, __hadd(v1.d, __hadd(v2.d, __hadd(v3.d, __hadd(v4.d, __hadd(v5.d, __hadd(v6.d, v7.d)))))));

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        half v0 = src0[i];
        half v1 = src1[i];
        half v2 = src2[i];
        half v3 = src3[i];
        half v4 = src4[i];
        half v5 = src5[i];
        half v6 = src6[i];
        half v7 = src7[i];

        half r = __hadd(v0, __hadd(v1, __hadd(v2, __hadd(v3, __hadd(v4, __hadd(v5, __hadd(v6, v7)))))));

        dst[i] = r;
        i++;
      }
    }
  }
}

void __global__ reduce_x4_fp16_kernel(
  const half* __restrict__ src0, // shape [scattered_M, N]
  const half* __restrict__ src1, // shape [scattered_M, N]
  const half* __restrict__ src2, // shape [scattered_M, N]
  const half* __restrict__ src3, // shape [scattered_M, N]
  half* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    half8* dst_h8_ptr = (half8*)(&dst[i]);

    const half8* src0_h8_ptr = (const half8*)(&src0[i]);
    const half8* src1_h8_ptr = (const half8*)(&src1[i]);
    const half8* src2_h8_ptr = (const half8*)(&src2[i]);
    const half8* src3_h8_ptr = (const half8*)(&src3[i]);

    half8 v0 = *src0_h8_ptr;
    half8 v1 = *src1_h8_ptr;
    half8 v2 = *src2_h8_ptr;
    half8 v3 = *src3_h8_ptr;

    half8 r;
    r.x = __hadd(v0.x, __hadd(v1.x, __hadd(v2.x, v3.x)));
    r.y = __hadd(v0.y, __hadd(v1.y, __hadd(v2.y, v3.y)));
    r.z = __hadd(v0.z, __hadd(v1.z, __hadd(v2.z, v3.z)));
    r.w = __hadd(v0.w, __hadd(v1.w, __hadd(v2.w, v3.w)));
    r.a = __hadd(v0.a, __hadd(v1.a, __hadd(v2.a, v3.a)));
    r.b = __hadd(v0.b, __hadd(v1.b, __hadd(v2.b, v3.b)));
    r.c = __hadd(v0.c, __hadd(v1.c, __hadd(v2.c, v3.c)));
    r.d = __hadd(v0.d, __hadd(v1.d, __hadd(v2.d, v3.d)));

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        half v0 = src0[i];
        half v1 = src1[i];
        half v2 = src2[i];
        half v3 = src3[i];

        half r = __hadd(v0, __hadd(v1, __hadd(v2, v3)));

        dst[i] = r;
        i++;
      }
    }
  }
}


void __global__ reduce_x2_fp16_kernel(
  const half* __restrict__ src0, // shape [scattered_M, N]
  const half* __restrict__ src1, // shape [scattered_M, N]
  half* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    half8* dst_h8_ptr = (half8*)(&dst[i]);

    const half8* src0_h8_ptr = (const half8*)(&src0[i]);
    const half8* src1_h8_ptr = (const half8*)(&src1[i]);

    half8 v0 = *src0_h8_ptr;
    half8 v1 = *src1_h8_ptr;

    half8 r;
    r.x = __hadd(v0.x, v1.x);
    r.y = __hadd(v0.y, v1.y);
    r.z = __hadd(v0.z, v1.z);
    r.w = __hadd(v0.w, v1.w);
    r.a = __hadd(v0.a, v1.a);
    r.b = __hadd(v0.b, v1.b);
    r.c = __hadd(v0.c, v1.c);
    r.d = __hadd(v0.d, v1.d);

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        half v0 = src0[i];
        half v1 = src1[i];

        half r = __hadd(v0, v1);

        dst[i] = r;
        i++;
      }
    }
  }
}

muillm_comm_error_t __muillm_reduce_fp16(
  hipStream_t stream,
  // inputs
  const half* src, // shape [local_size, scattered_M, N]
  int scattered_count,
  int local_size,
  // outputs
  half* dst
) {

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(scattered_count, REDUCE_PER_BLOCK);

  if (local_size == 8) {
    // compute the src pointers by applying the offsets
    const half* src0 = src + (0 * scattered_count);
    const half* src1 = src + (1 * scattered_count);
    const half* src2 = src + (2 * scattered_count);
    const half* src3 = src + (3 * scattered_count);
    const half* src4 = src + (4 * scattered_count);
    const half* src5 = src + (5 * scattered_count);
    const half* src6 = src + (6 * scattered_count);
    const half* src7 = src + (7 * scattered_count);

    reduce_x8_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      src4,
      src5,
      src6,
      src7,
      dst,
      scattered_count
    );
  } else if (local_size == 4) {
    // compute the src pointers by applying the offsets
    const half* src0 = src + (0 * scattered_count);
    const half* src1 = src + (1 * scattered_count);
    const half* src2 = src + (2 * scattered_count);
    const half* src3 = src + (3 * scattered_count);

    reduce_x4_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      dst,
      scattered_count
    );
  } else if (local_size == 2) {
    // compute the src pointers by applying the offsets
    const half* src0 = src + (0 * scattered_count);
    const half* src1 = src + (1 * scattered_count);

    reduce_x2_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      dst,
      scattered_count
    );
  } else {
    return MUILLM_COMM_UNSUPPORTED_SIZE;
  }

  return MUILLM_COMM_SUCCESS;
}


void __global__ reduce_x8_bf16_kernel(
  const __hip_bfloat16* __restrict__ src0, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src1, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src2, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src3, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src4, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src5, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src6, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src7, // shape [scattered_M, N]
  __hip_bfloat16* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    __hip_bfloat168* dst_h8_ptr = (__hip_bfloat168*)(&dst[i]);

    const __hip_bfloat168* src0_h8_ptr = (const __hip_bfloat168*)(&src0[i]);
    const __hip_bfloat168* src1_h8_ptr = (const __hip_bfloat168*)(&src1[i]);
    const __hip_bfloat168* src2_h8_ptr = (const __hip_bfloat168*)(&src2[i]);
    const __hip_bfloat168* src3_h8_ptr = (const __hip_bfloat168*)(&src3[i]);
    const __hip_bfloat168* src4_h8_ptr = (const __hip_bfloat168*)(&src4[i]);
    const __hip_bfloat168* src5_h8_ptr = (const __hip_bfloat168*)(&src5[i]);
    const __hip_bfloat168* src6_h8_ptr = (const __hip_bfloat168*)(&src6[i]);
    const __hip_bfloat168* src7_h8_ptr = (const __hip_bfloat168*)(&src7[i]);

    __hip_bfloat168 v0 = *src0_h8_ptr;
    __hip_bfloat168 v1 = *src1_h8_ptr;
    __hip_bfloat168 v2 = *src2_h8_ptr;
    __hip_bfloat168 v3 = *src3_h8_ptr;
    __hip_bfloat168 v4 = *src4_h8_ptr;
    __hip_bfloat168 v5 = *src5_h8_ptr;
    __hip_bfloat168 v6 = *src6_h8_ptr;
    __hip_bfloat168 v7 = *src7_h8_ptr;

    __hip_bfloat168 r;
    r.x = __hadd(v0.x, __hadd(v1.x, __hadd(v2.x, __hadd(v3.x, __hadd(v4.x, __hadd(v5.x, __hadd(v6.x, v7.x)))))));
    r.y = __hadd(v0.y, __hadd(v1.y, __hadd(v2.y, __hadd(v3.y, __hadd(v4.y, __hadd(v5.y, __hadd(v6.y, v7.y)))))));
    r.z = __hadd(v0.z, __hadd(v1.z, __hadd(v2.z, __hadd(v3.z, __hadd(v4.z, __hadd(v5.z, __hadd(v6.z, v7.z)))))));
    r.w = __hadd(v0.w, __hadd(v1.w, __hadd(v2.w, __hadd(v3.w, __hadd(v4.w, __hadd(v5.w, __hadd(v6.w, v7.w)))))));
    r.a = __hadd(v0.a, __hadd(v1.a, __hadd(v2.a, __hadd(v3.a, __hadd(v4.a, __hadd(v5.a, __hadd(v6.a, v7.a)))))));
    r.b = __hadd(v0.b, __hadd(v1.b, __hadd(v2.b, __hadd(v3.b, __hadd(v4.b, __hadd(v5.b, __hadd(v6.b, v7.b)))))));
    r.c = __hadd(v0.c, __hadd(v1.c, __hadd(v2.c, __hadd(v3.c, __hadd(v4.c, __hadd(v5.c, __hadd(v6.c, v7.c)))))));
    r.d = __hadd(v0.d, __hadd(v1.d, __hadd(v2.d, __hadd(v3.d, __hadd(v4.d, __hadd(v5.d, __hadd(v6.d, v7.d)))))));

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        __hip_bfloat16 v0 = src0[i];
        __hip_bfloat16 v1 = src1[i];
        __hip_bfloat16 v2 = src2[i];
        __hip_bfloat16 v3 = src3[i];
        __hip_bfloat16 v4 = src4[i];
        __hip_bfloat16 v5 = src5[i];
        __hip_bfloat16 v6 = src6[i];
        __hip_bfloat16 v7 = src7[i];

        __hip_bfloat16 r = __hadd(v0, __hadd(v1, __hadd(v2, __hadd(v3, __hadd(v4, __hadd(v5, __hadd(v6, v7)))))));

        dst[i] = r;
        i++;
      }
    }
  }
}

void __global__ reduce_x4_bf16_kernel(
  const __hip_bfloat16* __restrict__ src0, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src1, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src2, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src3, // shape [scattered_M, N]
  __hip_bfloat16* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    __hip_bfloat168* dst_h8_ptr = (__hip_bfloat168*)(&dst[i]);

    const __hip_bfloat168* src0_h8_ptr = (const __hip_bfloat168*)(&src0[i]);
    const __hip_bfloat168* src1_h8_ptr = (const __hip_bfloat168*)(&src1[i]);
    const __hip_bfloat168* src2_h8_ptr = (const __hip_bfloat168*)(&src2[i]);
    const __hip_bfloat168* src3_h8_ptr = (const __hip_bfloat168*)(&src3[i]);

    __hip_bfloat168 v0 = *src0_h8_ptr;
    __hip_bfloat168 v1 = *src1_h8_ptr;
    __hip_bfloat168 v2 = *src2_h8_ptr;
    __hip_bfloat168 v3 = *src3_h8_ptr;

    __hip_bfloat168 r;
    r.x = __hadd(v0.x, __hadd(v1.x, __hadd(v2.x, v3.x)));
    r.y = __hadd(v0.y, __hadd(v1.y, __hadd(v2.y, v3.y)));
    r.z = __hadd(v0.z, __hadd(v1.z, __hadd(v2.z, v3.z)));
    r.w = __hadd(v0.w, __hadd(v1.w, __hadd(v2.w, v3.w)));
    r.a = __hadd(v0.a, __hadd(v1.a, __hadd(v2.a, v3.a)));
    r.b = __hadd(v0.b, __hadd(v1.b, __hadd(v2.b, v3.b)));
    r.c = __hadd(v0.c, __hadd(v1.c, __hadd(v2.c, v3.c)));
    r.d = __hadd(v0.d, __hadd(v1.d, __hadd(v2.d, v3.d)));

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        __hip_bfloat16 v0 = src0[i];
        __hip_bfloat16 v1 = src1[i];
        __hip_bfloat16 v2 = src2[i];
        __hip_bfloat16 v3 = src3[i];

        __hip_bfloat16 r = __hadd(v0, __hadd(v1, __hadd(v2, v3)));

        dst[i] = r;
        i++;
      }
    }
  }
}


void __global__ reduce_x2_bf16_kernel(
  const __hip_bfloat16* __restrict__ src0, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src1, // shape [scattered_M, N]
  __hip_bfloat16* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    __hip_bfloat168* dst_h8_ptr = (__hip_bfloat168*)(&dst[i]);

    const __hip_bfloat168* src0_h8_ptr = (const __hip_bfloat168*)(&src0[i]);
    const __hip_bfloat168* src1_h8_ptr = (const __hip_bfloat168*)(&src1[i]);

    __hip_bfloat168 v0 = *src0_h8_ptr;
    __hip_bfloat168 v1 = *src1_h8_ptr;

    __hip_bfloat168 r;
    r.x = __hadd(v0.x, v1.x);
    r.y = __hadd(v0.y, v1.y);
    r.z = __hadd(v0.z, v1.z);
    r.w = __hadd(v0.w, v1.w);
    r.a = __hadd(v0.a, v1.a);
    r.b = __hadd(v0.b, v1.b);
    r.c = __hadd(v0.c, v1.c);
    r.d = __hadd(v0.d, v1.d);

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        __hip_bfloat16 v0 = src0[i];
        __hip_bfloat16 v1 = src1[i];

        __hip_bfloat16 r = __hadd(v0, v1);

        dst[i] = r;
        i++;
      }
    }
  }
}

muillm_comm_error_t __muillm_reduce_bf16(
  hipStream_t stream,
  // inputs
  const __hip_bfloat16* src, // shape [local_size, scattered_M, N]
  int scattered_count,
  int local_size,
  // outputs
  __hip_bfloat16* dst
) {

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(scattered_count, REDUCE_PER_BLOCK);

  if (local_size == 8) {
    // compute the src pointers by applying the offsets
    const __hip_bfloat16* src0 = src + (0 * scattered_count);
    const __hip_bfloat16* src1 = src + (1 * scattered_count);
    const __hip_bfloat16* src2 = src + (2 * scattered_count);
    const __hip_bfloat16* src3 = src + (3 * scattered_count);
    const __hip_bfloat16* src4 = src + (4 * scattered_count);
    const __hip_bfloat16* src5 = src + (5 * scattered_count);
    const __hip_bfloat16* src6 = src + (6 * scattered_count);
    const __hip_bfloat16* src7 = src + (7 * scattered_count);

    reduce_x8_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      src4,
      src5,
      src6,
      src7,
      dst,
      scattered_count
    );
  } else if (local_size == 4) {
    // compute the src pointers by applying the offsets
    const __hip_bfloat16* src0 = src + (0 * scattered_count);
    const __hip_bfloat16* src1 = src + (1 * scattered_count);
    const __hip_bfloat16* src2 = src + (2 * scattered_count);
    const __hip_bfloat16* src3 = src + (3 * scattered_count);

    reduce_x4_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      dst,
      scattered_count
    );
  } else if (local_size == 2) {
    // compute the src pointers by applying the offsets
    const __hip_bfloat16* src0 = src + (0 * scattered_count);
    const __hip_bfloat16* src1 = src + (1 * scattered_count);

    reduce_x2_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      dst,
      scattered_count
    );
  } else {
    return MUILLM_COMM_UNSUPPORTED_SIZE;
  }

  return MUILLM_COMM_SUCCESS;
}


muillm_comm_error_t __muillm_reduce(
  hipStream_t stream,
  // inputs
  const void* src, // shape [local_size, scattered_M, N]
  int scattered_count,
  int local_size,
  muillm_comm_datatype_t datatype,
  // outputs
  void* dst
) {
  if (datatype == MUILLM_COMM_FP16) {
    return __muillm_reduce_fp16(
      stream,
      (const half*) src,
      scattered_count,
      local_size,
      (half*) dst
    );
  } else if (datatype == MUILLM_COMM_BF16) {
    return __muillm_reduce_bf16(
      stream,
      (const __hip_bfloat16*) src,
      scattered_count,
      local_size,
      (__hip_bfloat16*) dst
    );
  } else {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
}