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

#define META_DIM 4

//
// All2All dispatch kernels
//

void __global__ all2all_dispatch_compute_send_counts_kernel(
  const int32_t* __restrict__ indices,
  uint32_t* __restrict__ send_offsets,
  // counters for the different ranks
  uint32_t* counters,
  // local counter to clear for next use
  uint32_t* next_counters,
  int num_local_experts,
  int total_send,
  int local_size,
  int local_rank
) {
  __shared__ uint32_t send_counts_shared[MUILLM_MAX_GPUS];

  if (threadIdx.x < local_size) {
    send_counts_shared[threadIdx.x] = 0;
  }

  __syncthreads();

  if (local_rank == 0 && threadIdx.x < local_size) {
    // clear the counters for the next use
    next_counters[threadIdx.x] = 0;
  }

  // each thread handles one token expert
  for (int i = threadIdx.x; i < total_send; i += THREADS_PER_BLOCK) {
    // TODO: avoid division if num_local_experts is power of 2
    int32_t dst_expert = indices[i];
    int32_t dst_rank = dst_expert / num_local_experts;
    atomicAdd((uint32_t*)&send_counts_shared[dst_rank], 1);
  }

  __syncthreads();

  // do a single thread prefix sum to compute send offsets
  if (threadIdx.x < local_size) {
    int rank = threadIdx.x;
    uint32_t send_count = send_counts_shared[rank];
    uint32_t send_offset = atomicAdd_system((uint32_t*) &counters[rank], send_count);
    send_offsets[rank] = send_offset;
  }
}

void all2all_dispatch_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ indices,
    uint32_t* __restrict__ send_offsets,
    // counters for the different ranks
    uint32_t* counters,
    // local counter to clear for next use
    uint32_t* next_counters,
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int local_size,
    int local_rank
) {
  // call kernel to compute send_counts using a single block to do the send offset reduction as well
  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = 1;

  int total_send = num_tokens * num_experts_per_token;

  all2all_dispatch_compute_send_counts_kernel<<<blocks, threads_per_block, 0, stream>>>(
    indices,
    send_offsets,
    counters,
    next_counters,
    num_local_experts,
    total_send,
    local_size,
    local_rank
  );
}

void __global__ all2all_dispatch_pack_send_buffers_fp32_kernel(
  const float* __restrict__ x,
  const int32_t* __restrict__ indices,
  uint32_t* __restrict__ send_offsets,
  float* __restrict__ send_buf0,
  float* __restrict__ send_buf1,
  float* __restrict__ send_buf2,
  float* __restrict__ send_buf3,
  float* __restrict__ send_buf4,
  float* __restrict__ send_buf5,
  float* __restrict__ send_buf6,
  float* __restrict__ send_buf7,
  int num_local_experts,
  int num_tokens,
  int num_experts_per_token,
  int hidden_dim,
  int buff_meta_offset,
  int rank
) {
  int expert_idx = blockIdx.x;
  int token_idx = blockIdx.y;

  x = &x[token_idx * hidden_dim];

  // determine where to write out
  // TODO: avoid division if num_local_experts is power of 2
  int32_t dst_expert = indices[token_idx * num_experts_per_token + expert_idx];
  int32_t dst_rank = dst_expert / num_local_experts;

  int32_t send_pos = -1;
  if (threadIdx.x == 0) {
    send_pos = (int32_t)atomicAdd((uint32_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos);

  float* send_buffs[8] = {send_buf0, send_buf1, send_buf2, send_buf3, send_buf4, send_buf5, send_buf6, send_buf7};
  float* send_buf = send_buffs[dst_rank];

  // write to send_buf
  float* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  for (int i = threadIdx.x; i < hidden_dim; i += THREADS_PER_BLOCK) {
    send_buf_ptr[i] = x[i];
  }

  // write to send_meta
  int32_t* send_meta = (int32_t*)(((uint8_t*)send_buf) + buff_meta_offset);
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  if (threadIdx.x == 0) {
    send_meta_ptr[0] = dst_expert;
    send_meta_ptr[1] = rank;
    send_meta_ptr[2] = token_idx;
    send_meta_ptr[3] = expert_idx;
  }
}

void all2all_dispatch_pack_send_buffers_fp32(
    hipStream_t stream,
    const float* __restrict__ x,
    const int32_t* __restrict__ indices,
    uint32_t* __restrict__ send_offsets,
    float* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int hidden_dim,
    int buff_meta_offset,
    int local_size,
    int local_rank
) {
  // call kernel to do the packing, with a 2D grid of size (num_experts_per_token, num_tokens)
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(num_experts_per_token, num_tokens);

  all2all_dispatch_pack_send_buffers_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    x,
    indices,
    send_offsets,
    send_buf0,
    send_buf1,
    send_buf2,
    send_buf3,
    send_buf4,
    send_buf5,
    send_buf6,
    send_buf7,
    num_local_experts,
    num_tokens,
    num_experts_per_token,
    hidden_dim,
    buff_meta_offset,
    local_rank
  );
}

void __global__ all2all_dispatch_pack_send_buffers_fp16_kernel(
  const half* __restrict__ x,
  const int32_t* __restrict__ indices,
  uint32_t* __restrict__ send_offsets,
  half* __restrict__ send_buf0,
  half* __restrict__ send_buf1,
  half* __restrict__ send_buf2,
  half* __restrict__ send_buf3,
  half* __restrict__ send_buf4,
  half* __restrict__ send_buf5,
  half* __restrict__ send_buf6,
  half* __restrict__ send_buf7,
  int num_local_experts,
  int num_tokens,
  int num_experts_per_token,
  int hidden_dim,
  int buff_meta_offset,
  int rank
) {
  int expert_idx = blockIdx.x;
  int token_idx = blockIdx.y;

  x = &x[token_idx * hidden_dim];

  // determine where to write out
  // TODO: avoid division if num_local_experts is power of 2
  int32_t dst_expert = indices[token_idx * num_experts_per_token + expert_idx];
  int32_t dst_rank = dst_expert / num_local_experts;

  int32_t send_pos = -1;
  if (threadIdx.x == 0) {
    send_pos = (int32_t)atomicAdd((uint32_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos);

  half* send_buffs[8] = {send_buf0, send_buf1, send_buf2, send_buf3, send_buf4, send_buf5, send_buf6, send_buf7};
  half* send_buf = send_buffs[dst_rank];

  // write to send_buf
  half* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  for (int i = threadIdx.x; i < hidden_dim; i += THREADS_PER_BLOCK) {
    send_buf_ptr[i] = x[i];
  }

  // write to send_meta
  int32_t* send_meta = (int32_t*)(((uint8_t*)send_buf) + buff_meta_offset);
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  if (threadIdx.x == 0) {
    send_meta_ptr[0] = dst_expert;
    send_meta_ptr[1] = rank;
    send_meta_ptr[2] = token_idx;
    send_meta_ptr[3] = expert_idx;
  }
}

void all2all_dispatch_pack_send_buffers_fp16(
    hipStream_t stream,
    const half* __restrict__ x,
    const int32_t* __restrict__ indices,
    uint32_t* __restrict__ send_offsets,
    half* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int hidden_dim,
    int buff_meta_offset,
    int local_size,
    int local_rank
) {
  // call kernel to do the packing, with a 2D grid of size (num_experts_per_token, num_tokens)
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(num_experts_per_token, num_tokens);

  all2all_dispatch_pack_send_buffers_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    x,
    indices,
    send_offsets,
    send_buf0,
    send_buf1,
    send_buf2,
    send_buf3,
    send_buf4,
    send_buf5,
    send_buf6,
    send_buf7,
    num_local_experts,
    num_tokens,
    num_experts_per_token,
    hidden_dim,
    buff_meta_offset,
    local_rank
  );
}

// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_dispatch_unpack_fp16_kernel(
    const half* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    half* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    // TODO recv_counts is on CPU so uncached
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset) {
  
  int lane_id = threadIdx.x % warpSize;

  int token_idx = blockIdx.x;

  if (token_idx >= recv_counts[0]) { // TODO: remove uncached access somehow
    return;
  }

  int global_expert_idx = recv_meta[token_idx * META_DIM + 0]; // expert_id
  int local_expert_idx = global_expert_idx - local_expert_offset;

  int local_num_expert_tokens = -1; // needs to be initialized to avoid compiler bug?

  //
  // reserve space in expert_x and expert_meta
  //

  // do a single atomic add per block and broadcast the result to the warp
  if (threadIdx.x == 0) {
    local_num_expert_tokens = atomicAdd(&expert_num_tokens[local_expert_idx], 1);
  }
  local_num_expert_tokens = __block_broadcast(local_num_expert_tokens);


  //
  // do the copies
  //

  // realign the pointers based on where to read/write
  recv_buf = &recv_buf[token_idx * hidden_dim];
  recv_meta = &recv_meta[token_idx * META_DIM];

  expert_x = &expert_x[(local_expert_idx * max_recv + local_num_expert_tokens) * hidden_dim];
  expert_meta = &expert_meta[(local_expert_idx * max_recv + local_num_expert_tokens) * META_DIM];

  // copy the token to expert_x
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    expert_x[d] = recv_buf[d];
  }

  // copy the meta to expert_meta
  if (threadIdx.x < META_DIM) {
    expert_meta[threadIdx.x] = recv_meta[threadIdx.x];
  }
}

void all2all_dispatch_unpack_fp16(
    hipStream_t stream,
    const half* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    half* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset,
    int max_total_recv) {

  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = max_total_recv;

  all2all_dispatch_unpack_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    recv_buf,
    recv_meta,
    expert_num_tokens,
    expert_x,
    expert_meta,
    recv_counts,
    hidden_dim,
    max_recv,
    local_expert_offset
  );
}

// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_dispatch_unpack_fp32_kernel(
    const float* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    float* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    // TODO recv_counts is on CPU so uncached
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset) {
 
  int lane_id = threadIdx.x % warpSize;

  int token_idx = blockIdx.x;

  if (token_idx >= recv_counts[0]) { // TODO: remove uncached access somehow
    return;
  }

  int global_expert_idx = recv_meta[token_idx * META_DIM + 0]; // expert_id
  int local_expert_idx = global_expert_idx - local_expert_offset;

  int local_num_expert_tokens = -1; // needs to be initialized to avoid compiler bug?

  //
  // reserve space in expert_x and expert_meta
  //

  // do a single atomic add per block and broadcast the result to the warp
  if (threadIdx.x == 0) {
    local_num_expert_tokens = atomicAdd(&expert_num_tokens[local_expert_idx], 1);
  }
  local_num_expert_tokens = __block_broadcast(local_num_expert_tokens);


  //
  // do the copies
  //

  // realign the pointers based on where to read/write
  recv_buf = &recv_buf[token_idx * hidden_dim];
  recv_meta = &recv_meta[token_idx * META_DIM];

  expert_x = &expert_x[(local_expert_idx * max_recv + local_num_expert_tokens) * hidden_dim];
  expert_meta = &expert_meta[(local_expert_idx * max_recv + local_num_expert_tokens) * META_DIM];

  // copy the token to expert_x
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    expert_x[d] = recv_buf[d];
  }

  // copy the meta to expert_meta
  if (threadIdx.x < META_DIM) {
    expert_meta[threadIdx.x] = recv_meta[threadIdx.x];
  }
}

void all2all_dispatch_unpack_fp32(
    hipStream_t stream,
    const float* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    float* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset,
    int max_total_recv) {

  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = max_total_recv;

  all2all_dispatch_unpack_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    recv_buf,
    recv_meta,
    expert_num_tokens,
    expert_x,
    expert_meta,
    recv_counts,
    hidden_dim,
    max_recv,
    local_expert_offset
  );
}

#define COMPUTE_PER_THREAD_FP32 4

void __global__ all2all_compute_fp32_kernel(
  const int32_t* __restrict__ expert_num_tokens,
  const float* __restrict__ expert_x,
  float* __restrict__ expert_y,
  int max_recv,
  int hidden_dim,
  int rank
) {

  int token_idx = blockIdx.x;
  int local_expert_idx = blockIdx.y;

  int num_local_tokens = expert_num_tokens[local_expert_idx];
  if (token_idx >= num_local_tokens) {
    return;
  }

  int element_idx = threadIdx.x * COMPUTE_PER_THREAD_FP32;

  // realign the pointers
  expert_x = &expert_x[(local_expert_idx * max_recv + token_idx) * hidden_dim + element_idx];
  expert_y = &expert_y[(local_expert_idx * max_recv + token_idx) * hidden_dim + element_idx];

  for (; element_idx < hidden_dim; element_idx += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP32) {
    if (element_idx + (COMPUTE_PER_THREAD_FP32 - 1) < hidden_dim) {
      // all elements are within range
      #pragma unroll
      for (int i = 0; i < COMPUTE_PER_THREAD_FP32; i++) {
        expert_y[i] = expert_x[i] * (1.0f + rank);
      }
    } else {
      // not all elements are within range, use a loop
      for (int i = 0; i < COMPUTE_PER_THREAD_FP32 && element_idx < hidden_dim; i++, element_idx++) {
        expert_y[i] = expert_x[i] * (1.0f + rank);
      }
    }
    // realign the pointers for the next iteration
    expert_x += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP32;
    expert_y += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP32;
  }
}

void all2all_compute_fp32(
  hipStream_t stream,
  const int32_t* __restrict__ expert_num_tokens,
  const float* __restrict__ expert_x,
  float* __restrict__ expert_y,
  int num_local_experts,
  int max_recv,
  int hidden_dim,
  int rank
) {
  // call the kernel to compute expert_y by processing COMPUTE_PER_THREADS_FP16 elements per thread
  // and THREADS_PER_BLOCK threads per block
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_compute_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_x,
    expert_y,
    max_recv,
    hidden_dim,
    rank
  );
}

#define COMPUTE_PER_THREAD_FP16 8

void __global__ all2all_compute_fp16_kernel(
  const int32_t* __restrict__ expert_num_tokens,
  const half* __restrict__ expert_x,
  half* __restrict__ expert_y,
  int max_recv,
  int hidden_dim,
  int rank
) {

  int token_idx = blockIdx.x;
  int local_expert_idx = blockIdx.y;

  int num_local_tokens = expert_num_tokens[local_expert_idx];
  if (token_idx >= num_local_tokens) {
    return;
  }

  int element_idx = threadIdx.x * COMPUTE_PER_THREAD_FP16;

  // realign the pointers
  expert_x = &expert_x[(local_expert_idx * max_recv + token_idx) * hidden_dim + element_idx];
  expert_y = &expert_y[(local_expert_idx * max_recv + token_idx) * hidden_dim + element_idx];

  for (; element_idx < hidden_dim; element_idx += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP16) {
    if (element_idx + (COMPUTE_PER_THREAD_FP16 - 1) < hidden_dim) {
      // all elements are within range
      #pragma unroll
      for (int i = 0; i < COMPUTE_PER_THREAD_FP16; i++) {
        expert_y[i] = __float2half_rn(__half2float(expert_x[i]) * (1.0f + rank));
      }
    } else {
      // not all elements are within range, use a loop
      for (int i = 0; i < COMPUTE_PER_THREAD_FP16 && element_idx < hidden_dim; i++, element_idx++) {
        expert_y[i] = __float2half_rn(__half2float(expert_x[i]) * (1.0f + rank));
      }
    }
    // realign the pointers for the next iteration
    expert_x += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP16;
    expert_y += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP16;
  }
}

void all2all_compute_fp16(
  hipStream_t stream,
  const int32_t* __restrict__ expert_num_tokens,
  const half* __restrict__ expert_x,
  half* __restrict__ expert_y,
  int num_local_experts,
  int max_recv,
  int hidden_dim,
  int rank
) {
  // call the kernel to compute expert_y by processing COMPUTE_PER_THREADS_FP16 elements per thread
  // and THREADS_PER_BLOCK threads per block
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_compute_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_x,
    expert_y,
    max_recv,
    hidden_dim,
    rank
  );
}

//
// All2All combine kernels
//

// Compute send count kernels

#define MAX_LDS_LOADED_EXPERTS 1024

void __global__ all2all_combine_compute_send_counts_kernel(
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    uint32_t* __restrict__ send_offsets, // shape [world_size]
    // counters for the different ranks
    uint32_t* counters,
    // local counter to clear for next use
    uint32_t* next_counters,
    int num_local_experts,
    int max_recv,
    int local_size,
    int local_rank
) {
  int warps_per_block = THREADS_PER_BLOCK / warpSize;
  int warp_id = threadIdx.x / warpSize;

  int lane_id = threadIdx.x % warpSize;

  __shared__ uint32_t send_counts_shared[MUILLM_MAX_GPUS];
  __shared__ uint32_t expert_num_tokens_shared[MAX_LDS_LOADED_EXPERTS];

  if (threadIdx.x < local_size) {
    send_counts_shared[threadIdx.x] = 0;
  }

  if (num_local_experts <= MAX_LDS_LOADED_EXPERTS) {
    // load expert_num_tokens into LDS
    for (int i = threadIdx.x; i < num_local_experts; i += THREADS_PER_BLOCK) {
      expert_num_tokens_shared[i] = (uint32_t) expert_num_tokens[i];
    }
  }

  __syncthreads();

  if (local_rank == 0 && threadIdx.x < local_size) {
    // clear the counters for the next use
    next_counters[threadIdx.x] = 0;
  }

  if (num_local_experts <= MAX_LDS_LOADED_EXPERTS) {
    // each thread handles one token expert
    for (int local_expert_idx = warp_id; local_expert_idx < num_local_experts; local_expert_idx += warps_per_block) {
      int num_local_tokens = expert_num_tokens_shared[local_expert_idx];

      const int32_t* local_expert_meta = &expert_meta[local_expert_idx * max_recv * META_DIM + 1];

      // for each token of this expert, figure out which rank to send to
      for (int i = lane_id; i < num_local_tokens; i += warpSize) {
        int32_t dst_rank = local_expert_meta[i * META_DIM]; // dst_rank
        atomicAdd((uint32_t*)&send_counts_shared[dst_rank], 1);
      }
    }
  } else {
    // each thread handles one token expert
    for (int local_expert_idx = warp_id; local_expert_idx < num_local_experts; local_expert_idx += warps_per_block) {
      int num_local_tokens = expert_num_tokens[local_expert_idx];

      const int32_t* local_expert_meta = &expert_meta[local_expert_idx * max_recv * META_DIM + 1];

      // for each token of this expert, figure out which rank to send to
      for (int i = lane_id; i < num_local_tokens; i += warpSize) {
        int32_t dst_rank = local_expert_meta[i * META_DIM]; // dst_rank
        atomicAdd((uint32_t*)&send_counts_shared[dst_rank], 1);
      }
    }
  }

  __syncthreads();

  // do a single thread prefix sum to compute send offsets
  if (threadIdx.x < local_size) {
    int rank = threadIdx.x;
    uint32_t send_count = send_counts_shared[rank];
    uint32_t send_offset = atomicAdd_system((uint32_t*) &counters[rank], send_count);
    send_offsets[rank] = send_offset;
  }
}

void all2all_combine_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    uint32_t* __restrict__ send_offsets, // shape [world_size]
    // counters for the different ranks
    uint32_t* counters,
    // local counter to clear for next use
    uint32_t* next_counters,
    int num_local_experts,
    int max_recv,
    int local_size,
    int local_rank
) {

  // call kernel to compute send_counts using a single block to do the send offset reduction as well
  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = 1;

  all2all_combine_compute_send_counts_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_meta,
    send_offsets,
    counters,
    next_counters,
    num_local_experts,
    max_recv,
    local_size,
    local_rank
  );
}

// Send kernels

#define THREADS_PER_BLOCK_COMBINE_PACK_SEND 64

void __global__ all2all_combine_pack_send_buffers_fp16_kernel(
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const half* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int32_t* __restrict__ send_offsets, // shape [world_size]
    half* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int max_recv,
    int hidden_dim,
    int buff_meta_offset
) {
  int token_idx = blockIdx.x;
  int local_expert_idx = blockIdx.y;

  int num_local_tokens = expert_num_tokens[local_expert_idx];
  if (token_idx >= num_local_tokens) {
    return;
  }

  const int32_t* local_expert_meta = &expert_meta[(local_expert_idx * max_recv + token_idx) * META_DIM];
  const half* local_expert_y = &expert_y[(local_expert_idx * max_recv + token_idx) * hidden_dim];

  // determine where to write out
  int32_t dst_rank = local_expert_meta[1]; // dst_rank

  int32_t send_pos = -1;
  if (threadIdx.x == 0) {
    send_pos = (int32_t)atomicAdd((uint32_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos, THREADS_PER_BLOCK_COMBINE_PACK_SEND);

  half* send_buffs[8] = {send_buf0, send_buf1, send_buf2, send_buf3, send_buf4, send_buf5, send_buf6, send_buf7};
  half* send_buf = send_buffs[dst_rank];

  // write to send_buf
  half* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  {
    int i = 8 * threadIdx.x;
    // vectorized part
    for (; i + 31 < hidden_dim; i += 4 * 8 * THREADS_PER_BLOCK_COMBINE_PACK_SEND) {
      half8 y0 = *((half8*) &local_expert_y[i + 0 * 8 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]);
      half8 y1 = *((half8*) &local_expert_y[i + 1 * 8 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]);
      half8 y2 = *((half8*) &local_expert_y[i + 2 * 8 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]);
      half8 y3 = *((half8*) &local_expert_y[i + 3 * 8 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]);
      *((half8*)&send_buf_ptr[i + 0 * 8 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]) = y0;
      *((half8*)&send_buf_ptr[i + 1 * 8 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]) = y1;
      *((half8*)&send_buf_ptr[i + 2 * 8 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]) = y2;
      *((half8*)&send_buf_ptr[i + 3 * 8 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]) = y3;
    }
    for (; i + 7 < hidden_dim; i += 8 * THREADS_PER_BLOCK_COMBINE_PACK_SEND) {
      half8 y = *((half8*) &local_expert_y[i]);
      *((half8*)&send_buf_ptr[i]) = y;
    }
    // remainders
    if (i + 3 < hidden_dim) {
      half4 y = *((half4*) &local_expert_y[i]);
      *((half4*)&send_buf_ptr[i]) = y;
      i += 4;
    }
    if (i + 1 < hidden_dim) {
      half2 y = *((half2*) &local_expert_y[i]);
      *((half2*)&send_buf_ptr[i]) = y;
      i += 2;
    }
    if (i < hidden_dim) {
      half y0 = local_expert_y[i + 0];
      send_buf_ptr[i + 0] = y0;
    }
  }

  // write to send_meta
  int32_t* send_meta = (int32_t*)(((uint8_t*)send_buf) + buff_meta_offset);
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  for (int i = threadIdx.x; i < META_DIM; i += THREADS_PER_BLOCK_COMBINE_PACK_SEND) {
    send_meta_ptr[i] = local_expert_meta[i];
  }
}

void all2all_combine_pack_send_buffers_fp16(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const half* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int32_t* __restrict__ send_offsets, // shape [world_size]
    half* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int max_recv,
    int hidden_dim,
    int buff_meta_offset
) {
  // call kernel to do the packing, with a 2D grid of size (max_recv, num_local_experts)
  const int threads_per_block = THREADS_PER_BLOCK_COMBINE_PACK_SEND;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_combine_pack_send_buffers_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_meta,
    expert_y,
    send_offsets,
    send_buf0,
    send_buf1,
    send_buf2,
    send_buf3,
    send_buf4,
    send_buf5,
    send_buf6,
    send_buf7,
    max_recv,
    hidden_dim,
    buff_meta_offset
  );
}

void __global__ all2all_combine_pack_send_buffers_fp32_kernel(
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const float* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int32_t* __restrict__ send_offsets, // shape [world_size]
    float* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int max_recv,
    int hidden_dim,
    int buff_meta_offset
) {
  int token_idx = blockIdx.x;
  int local_expert_idx = blockIdx.y;

  int num_local_tokens = expert_num_tokens[local_expert_idx];
  if (token_idx >= num_local_tokens) {
    return;
  }

  const int32_t* local_expert_meta = &expert_meta[(local_expert_idx * max_recv + token_idx) * META_DIM];
  const float* local_expert_y = &expert_y[(local_expert_idx * max_recv + token_idx) * hidden_dim];

  // determine where to write out
  int32_t dst_rank = local_expert_meta[1]; // dst_rank

  int32_t send_pos = -1;
  if (threadIdx.x == 0) {
    send_pos = (int32_t)atomicAdd((uint32_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos, THREADS_PER_BLOCK_COMBINE_PACK_SEND);

  float* send_buffs[8] = {send_buf0, send_buf1, send_buf2, send_buf3, send_buf4, send_buf5, send_buf6, send_buf7};
  float* send_buf = send_buffs[dst_rank];

  // write to send_buf
  float* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  {
    int i = 4 * threadIdx.x;
    // vectorized part
    for (; i + 15 < hidden_dim; i += 4 * 4 * THREADS_PER_BLOCK_COMBINE_PACK_SEND) {
      float4 y0 = *((float4*) &local_expert_y[i + 0 * 4 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]);
      float4 y1 = *((float4*) &local_expert_y[i + 1 * 4 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]);
      float4 y2 = *((float4*) &local_expert_y[i + 2 * 4 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]);
      float4 y3 = *((float4*) &local_expert_y[i + 3 * 4 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]);
      *((float4*)&send_buf_ptr[i + 0 * 4 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]) = y0;
      *((float4*)&send_buf_ptr[i + 1 * 4 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]) = y1;
      *((float4*)&send_buf_ptr[i + 2 * 4 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]) = y2;
      *((float4*)&send_buf_ptr[i + 3 * 4 * THREADS_PER_BLOCK_COMBINE_PACK_SEND]) = y3;
    }
    for (; i + 3 < hidden_dim; i += 4 * THREADS_PER_BLOCK_COMBINE_PACK_SEND) {
      float4 y = *((float4*) &local_expert_y[i]);
      *((float4*)&send_buf_ptr[i]) = y;
    }
    if (i + 1 < hidden_dim) {
      float2 y = *((float2*) &local_expert_y[i]);
      *((float2*)&send_buf_ptr[i]) = y;
      i += 2;
    }
    if (i < hidden_dim) {
      float y0 = local_expert_y[i + 0];
      send_buf_ptr[i + 0] = y0;
    }
  }

  // write to send_meta
  int32_t* send_meta = (int32_t*)(((uint8_t*)send_buf) + buff_meta_offset);
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  for (int i = threadIdx.x; i < META_DIM; i += THREADS_PER_BLOCK_COMBINE_PACK_SEND) {
    send_meta_ptr[i] = local_expert_meta[i];
  }
}

void all2all_combine_pack_send_buffers_fp32(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const float* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int32_t* __restrict__ send_offsets, // shape [world_size]
    float* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int max_recv,
    int hidden_dim,
    int buff_meta_offset
) {
  // call kernel to do the packing, with a 2D grid of size (max_recv, num_local_experts)
  const int threads_per_block = THREADS_PER_BLOCK_COMBINE_PACK_SEND;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_combine_pack_send_buffers_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_meta,
    expert_y,
    send_offsets,
    send_buf0,
    send_buf1,
    send_buf2,
    send_buf3,
    send_buf4,
    send_buf5,
    send_buf6,
    send_buf7,
    max_recv,
    hidden_dim,
    buff_meta_offset
  );
}

// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_combine_scale_experts_fp32_kernel(
  const float* __restrict__ recv_buf,
  const int32_t* __restrict__ recv_meta,
  const float* __restrict__ weights,
  float* __restrict__ scaled_expert_outputs,
  int hidden_dim,
  int experts_per_token
) {
  int recv_token_idx = blockIdx.x;

  // realign input pointers
  recv_buf = &recv_buf[recv_token_idx * hidden_dim];
  recv_meta = &recv_meta[recv_token_idx * META_DIM];

  int src_token_idx = recv_meta[2];
  int src_k = recv_meta[3]; // src_k

  float w = weights[src_token_idx * experts_per_token + src_k];

  // realign output pointer
  scaled_expert_outputs = &scaled_expert_outputs[(src_token_idx * experts_per_token + src_k) * hidden_dim];

  // scale and write the expert output from the receive buffer
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    scaled_expert_outputs[d] = recv_buf[d] * w;
  }
}

void __global__ all2all_combine_write_back_fp32_kernel(
  const float* __restrict__ scaled_expert_outputs,
  float* __restrict__ output,
  int hidden_dim,
  int experts_per_token
) {
  int token_idx = blockIdx.x;

  // realign input pointer
  scaled_expert_outputs = &scaled_expert_outputs[token_idx * experts_per_token * hidden_dim];

  // realign output pointer
  output = &output[token_idx * hidden_dim];

  // combine the expert outputs into output
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    float sum = 0.0f;
    for (int k = 0; k < experts_per_token; k++) {
      sum += scaled_expert_outputs[k * hidden_dim + d];
    }
    output[d] = sum;
  }
}

void all2all_combine_unpack_fp32(
    hipStream_t stream,
    const float* __restrict__ recv_buf, // shape [total_recv, hidden_sim]
    const int32_t* __restrict__ recv_meta, // shape [total_recv, META_DIM]
    const float* __restrict__ weights, // shape [num_tokens, experts_per_token]
    float* __restrict__ scaled_expert_outputs, // shape [num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ output, // shape [max_num_tokens, hidden_dim]
    int hidden_dim,
    int total_recv,
    int num_tokens,
    int experts_per_token
) {
  // First kernel to scale expert outputs by weights and write to scaled_expert_outputs
  {
    const int threads_per_block = THREADS_PER_BLOCK;
    const int blocks = total_recv;

    all2all_combine_scale_experts_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
      recv_buf,
      recv_meta,
      weights,
      scaled_expert_outputs,
      hidden_dim,
      experts_per_token
    );
  }
  // Second kernel to combine scaled_expert_outputs into output
  {
    const int threads_per_block = THREADS_PER_BLOCK;
    const int blocks = num_tokens;

    all2all_combine_write_back_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
      scaled_expert_outputs,
      output,
      hidden_dim,
      experts_per_token
    );
  }
}


// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_combine_scale_experts_fp16_kernel(
  const half* __restrict__ recv_buf,
  const int32_t* __restrict__ recv_meta,
  const float* __restrict__ weights,
  float* __restrict__ scaled_expert_outputs,
  int hidden_dim,
  int experts_per_token
) {
  int recv_token_idx = blockIdx.x;

  // realign input pointers
  recv_buf = &recv_buf[recv_token_idx * hidden_dim];
  recv_meta = &recv_meta[recv_token_idx * META_DIM];

  // int dst_expert = recv_meta[0];
  // int dst_rank = recv_meta[1];
  int src_token_idx = recv_meta[2];
  int src_k = recv_meta[3]; // src_k

  float w = weights[src_token_idx * experts_per_token + src_k];

  // realign output pointer
  scaled_expert_outputs = &scaled_expert_outputs[(src_token_idx * experts_per_token + src_k) * hidden_dim];

  // scale and write the expert output from the receive buffer
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    scaled_expert_outputs[d] = __half2float(recv_buf[d]) * w;
  }
}

void __global__ all2all_combine_write_back_fp16_kernel(
  const float* __restrict__ scaled_expert_outputs,
  half* __restrict__ output,
  int hidden_dim,
  int experts_per_token
) {
  int token_idx = blockIdx.x;

  // realign input pointer
  scaled_expert_outputs = &scaled_expert_outputs[token_idx * experts_per_token * hidden_dim];

  // realign output pointer
  output = &output[token_idx * hidden_dim];

  // combine the expert outputs into output
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    float sum = 0.0f;
    for (int k = 0; k < experts_per_token; k++) {
      sum += scaled_expert_outputs[k * hidden_dim + d];
    }
    output[d] = __float2half(sum);
  }
}

void all2all_combine_unpack_fp16(
    hipStream_t stream,
    const half* __restrict__ recv_buf, // shape [total_recv, hidden_sim]
    const int32_t* __restrict__ recv_meta, // shape [total_recv, META_DIM]
    const float* __restrict__ weights, // shape [num_tokens, experts_per_token]
    float* __restrict__ scaled_expert_outputs, // shape [num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ output, // shape [max_num_tokens, hidden_dim]
    int hidden_dim,
    int total_recv,
    int num_tokens,
    int experts_per_token
) {
  // First kernel to scale expert outputs by weights and write to scaled_expert_outputs
  {
    const int threads_per_block = THREADS_PER_BLOCK;
    const int blocks = total_recv;

    all2all_combine_scale_experts_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
      recv_buf,
      recv_meta,
      weights,
      scaled_expert_outputs,
      hidden_dim,
      experts_per_token
    );
  }
  // Second kernel to combine scaled_expert_outputs into output
  {
    const int threads_per_block = THREADS_PER_BLOCK;
    const int blocks = num_tokens;

    all2all_combine_write_back_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
      scaled_expert_outputs,
      output,
      hidden_dim,
      experts_per_token
    );
  }
}