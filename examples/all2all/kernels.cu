#include <hip/hip_fp16.h>

#include <stdint.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

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

__inline__ __device__ int __block_broadcast(int val) {
  int lane_id = threadIdx.x % warpSize;

  if (THREADS_PER_BLOCK > warpSize) {
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

#define DIV_ROUND_UP(x, y) (((x) + (y) - 1) / (y))

#define META_DIM 4

void __global__ all2all_dispatch_compute_send_counts_kernel(
  const int32_t* __restrict__ indices,
  int64_t* __restrict__ send_counts,
  int64_t* __restrict__ send_offsets,
  int num_local_experts,
  int total_send,
  int world_size
) {

  // each thread handles one token expert
  for (int i = threadIdx.x; i < total_send; i += THREADS_PER_BLOCK) {
    // TODO: avoid division if num_local_experts is power of 2
    int32_t dst_expert = indices[i];
    int32_t dst_rank = dst_expert / num_local_experts;
    atomicAdd((uint64_t*)&send_counts[dst_rank], 1);
  }

  __syncthreads();

  // do a single thread prefix sum to compute send offsets
  if (threadIdx.x == 0) {
    int offset = 0;
    for (int r = 0; r < world_size; r++) {
      send_offsets[r] = offset;
      offset += send_counts[r];
    }
  }
}

void all2all_dispatch_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ indices,
    int64_t* __restrict__ send_counts,
    int64_t* __restrict__ send_offsets,
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int world_size
) {
  // call kernel to compute send_counts using a single block to do the send offset reduction as well
  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = 1;

  int total_send = num_tokens * num_experts_per_token;

  all2all_dispatch_compute_send_counts_kernel<<<blocks, threads_per_block, 0, stream>>>(
    indices,
    send_counts,
    send_offsets,
    num_local_experts,
    total_send,
    world_size
  );
}

void __global__ all2all_dispatch_pack_send_buffers_fp32_kernel(
  const float* __restrict__ x,
  const int32_t* __restrict__ indices,
  int64_t* __restrict__ send_offsets,
  int32_t* __restrict__ send_meta,
  float* __restrict__ send_buf,
  int num_local_experts,
  int num_tokens,
  int num_experts_per_token,
  int hidden_dim,
  int world_size,
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
    send_pos = (int32_t)atomicAdd((uint64_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos);

  // write to send_meta
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  if (threadIdx.x == 0) {
    send_meta_ptr[0] = dst_expert;
    send_meta_ptr[1] = rank;
    send_meta_ptr[2] = token_idx;
    send_meta_ptr[3] = expert_idx;
  }

  // write to send_buf
  float* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  for (int i = threadIdx.x; i < hidden_dim; i += THREADS_PER_BLOCK) {
    send_buf_ptr[i] = x[i];
  }
}

void all2all_dispatch_pack_send_buffers_fp32(
    hipStream_t stream,
    const float* __restrict__ x,
    const int32_t* __restrict__ indices,
    int64_t* __restrict__ send_offsets,
    int32_t* __restrict__ send_meta,
    float* __restrict__ send_buf,
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int hidden_dim,
    int world_size,
    int rank
) {
  // call kernel to do the packing, with a 2D grid of size (num_experts_per_token, num_tokens)
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(num_experts_per_token, num_tokens);

  all2all_dispatch_pack_send_buffers_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    x,
    indices,
    send_offsets,
    send_meta,
    send_buf,
    num_local_experts,
    num_tokens,
    num_experts_per_token,
    hidden_dim,
    world_size,
    rank
  );
}

void __global__ all2all_dispatch_pack_send_buffers_fp16_kernel(
  const half* __restrict__ x,
  const int32_t* __restrict__ indices,
  int64_t* __restrict__ send_offsets,
  int32_t* __restrict__ send_meta,
  half* __restrict__ send_buf,
  int num_local_experts,
  int num_tokens,
  int num_experts_per_token,
  int hidden_dim,
  int world_size,
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
    send_pos = (int32_t)atomicAdd((uint64_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos);

  // write to send_meta
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  if (threadIdx.x == 0) {
    send_meta_ptr[0] = dst_expert;
    send_meta_ptr[1] = rank;
    send_meta_ptr[2] = token_idx;
    send_meta_ptr[3] = expert_idx;
  }

  // write to send_buf
  half* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  for (int i = threadIdx.x; i < hidden_dim; i += THREADS_PER_BLOCK) {
    send_buf_ptr[i] = x[i];
  }
}

void all2all_dispatch_pack_send_buffers_fp16(
    hipStream_t stream,
    const half* __restrict__ x,
    const int32_t* __restrict__ indices,
    int64_t* __restrict__ send_offsets,
    int32_t* __restrict__ send_meta,
    half* __restrict__ send_buf,
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int hidden_dim,
    int world_size,
    int rank
) {
  // call kernel to do the packing, with a 2D grid of size (num_experts_per_token, num_tokens)
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(num_experts_per_token, num_tokens);

  all2all_dispatch_pack_send_buffers_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    x,
    indices,
    send_offsets,
    send_meta,
    send_buf,
    num_local_experts,
    num_tokens,
    num_experts_per_token,
    hidden_dim,
    world_size,
    rank
  );
}

// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_dispatch_pack_fp16_kernel(
    const half* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    half* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    int hidden_dim,
    int max_recv,
    int local_expert_offset) {
  
  int lane_id = threadIdx.x % warpSize;

  int token_idx = blockIdx.x;

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

void all2all_dispatch_pack_fp16(
    hipStream_t stream,
    const half* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    half* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    int hidden_dim,
    int max_recv,
    int local_expert_offset,
    int total_recv) {

  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = total_recv;

  all2all_dispatch_pack_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    recv_buf,
    recv_meta,
    expert_num_tokens,
    expert_x,
    expert_meta,
    hidden_dim,
    max_recv,
    local_expert_offset
  );
}

// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_dispatch_pack_fp32_kernel(
    const float* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    float* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    int hidden_dim,
    int max_recv,
    int local_expert_offset) {
 
  int lane_id = threadIdx.x % warpSize;

  int token_idx = blockIdx.x;

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

void all2all_dispatch_pack_fp32(
    hipStream_t stream,
    const float* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    float* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    int hidden_dim,
    int max_recv,
    int local_expert_offset,
    int total_recv) {

  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = total_recv;

  all2all_dispatch_pack_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    recv_buf,
    recv_meta,
    expert_num_tokens,
    expert_x,
    expert_meta,
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

void __global__ all2all_combine_compute_send_counts_kernel(
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    int64_t* __restrict__ send_counts, // shape [world_size]
    int64_t* __restrict__ send_offsets, // shape [world_size]
    int num_local_experts,
    int max_recv,
    int world_size
) {

  // each thread handles one token expert
  for (int local_expert_idx = 0; local_expert_idx < num_local_experts; local_expert_idx++) {
    int num_local_tokens = expert_num_tokens[local_expert_idx];

    const int32_t* local_expert_meta = &expert_meta[local_expert_idx * max_recv * META_DIM + 1];

    // for each token of this expert, figure out which rank to send to
    for (int i = threadIdx.x; i < num_local_tokens; i += THREADS_PER_BLOCK) {
      int32_t dst_rank = local_expert_meta[i * META_DIM]; // dst_rank
      atomicAdd((uint64_t*)&send_counts[dst_rank], 1);
    }
  }

  __syncthreads();

  // do a single thread prefix sum to compute send offsets
  if (threadIdx.x == 0) {
    int offset = 0;
    for (int r = 0; r < world_size; r++) {
      send_offsets[r] = offset;
      offset += send_counts[r];
    }
  }
}

void all2all_combine_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    int64_t* __restrict__ send_counts, // shape [world_size]
    int64_t* __restrict__ send_offsets, // shape [world_size]
    int num_local_experts,
    int max_recv,
    int world_size
) {

  // call kernel to compute send_counts using a single block to do the send offset reduction as well
  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = 1;

  all2all_combine_compute_send_counts_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_meta,
    send_counts,
    send_offsets,
    num_local_experts,
    max_recv,
    world_size
  );
}


void __global__ all2all_combine_pack_send_buffers_fp32_kernel(
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const float* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int64_t* __restrict__ send_offsets, // shape [world_size]
    int32_t* __restrict__ send_meta, // shape [total_send, META_DIM]
    float* __restrict__ send_buf, // shape [total_send, hidden_dim]
    int max_recv,
    int hidden_dim
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
    send_pos = (int32_t)atomicAdd((uint64_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos);

  // write to send_meta
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  for (int i = threadIdx.x; i < META_DIM; i += THREADS_PER_BLOCK) {
    send_meta_ptr[i] = local_expert_meta[i];
  }

  // write to send_buf
  float* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  for (int i = threadIdx.x; i < hidden_dim; i += THREADS_PER_BLOCK) {
    send_buf_ptr[i] = local_expert_y[i];
  }
}

void all2all_combine_pack_send_buffers_fp32(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const float* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int64_t* __restrict__ send_offsets, // shape [world_size]
    int32_t* __restrict__ send_meta, // shape [total_send, META_DIM]
    float* __restrict__ send_buf, // shape [total_send, hidden_dim]
    int total_send,
    int num_local_experts,
    int max_recv,
    int hidden_dim
) {
  // call kernel to do the packing, with a 2D grid of size (max_recv, num_local_experts)
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_combine_pack_send_buffers_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_meta,
    expert_y,
    send_offsets,
    send_meta,
    send_buf,
    max_recv,
    hidden_dim
  );
}

void __global__ all2all_combine_pack_send_buffers_fp16_kernel(
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const half* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int64_t* __restrict__ send_offsets, // shape [world_size]
    int32_t* __restrict__ send_meta, // shape [total_send, META_DIM]
    half* __restrict__ send_buf, // shape [total_send, hidden_dim]
    int max_recv,
    int hidden_dim
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
    send_pos = (int32_t)atomicAdd((uint64_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos);

  // write to send_meta
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  for (int i = threadIdx.x; i < META_DIM; i += THREADS_PER_BLOCK) {
    send_meta_ptr[i] = local_expert_meta[i];
  }

  // write to send_buf
  half* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  for (int i = threadIdx.x; i < hidden_dim; i += THREADS_PER_BLOCK) {
    send_buf_ptr[i] = local_expert_y[i];
  }
}

void all2all_combine_pack_send_buffers_fp16(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const half* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int64_t* __restrict__ send_offsets, // shape [world_size]
    int32_t* __restrict__ send_meta, // shape [total_send, META_DIM]
    half* __restrict__ send_buf, // shape [total_send, hidden_dim]
    int total_send,
    int num_local_experts,
    int max_recv,
    int hidden_dim
) {
  // call kernel to do the packing, with a 2D grid of size (max_recv, num_local_experts)
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_combine_pack_send_buffers_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_meta,
    expert_y,
    send_offsets,
    send_meta,
    send_buf,
    max_recv,
    hidden_dim
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

void all2all_combine_pack_fp32(
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

void all2all_combine_pack_fp16(
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