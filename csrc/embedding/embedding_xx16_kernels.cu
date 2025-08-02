#include <hip/hip_fp16.h>

#include <stdint.h>

typedef uint16_t xx16;

#define THREADS_PER_BLOCK 256

template <typename T>
static inline const T* __device__ addr(const T* p, unsigned index) {
  // helps the AMDGPU compiler understand it can use the sgrp pair + single vgpr addressing mode
  unsigned byte_offset = sizeof(T) * index;
  const uint8_t* p8 = (const uint8_t*)p;
  return (const T*) (p8 + byte_offset);
}

template <typename T>
static inline T* __device__ addr(T* p, unsigned index) {
  // helps the AMDGPU compiler understand it can use the sgrp pair + single vgpr addressing mode
  unsigned byte_offset = sizeof(T) * index;
  uint8_t* p8 = (uint8_t*)p;
  return (T*) (p8 + byte_offset);
}

// expected block dimensions: [x=N]
__global__ void embedding_forward_xx16_indices_i32_kernel(
  // inputs
  const uint16_t* embedding_weights,
  const uint32_t* indices,
  // outputs
  uint16_t* out_embeddings,
  // other arguments
  unsigned E
) {
  // one block copies one embedding
  unsigned idx = blockIdx.x;

  uint32_t embedding_idx = indices[idx];

  // re-align the pointers
  embedding_weights = &embedding_weights[embedding_idx * E];
  out_embeddings = &out_embeddings[idx * E];

  // copy the embedding out
  for (unsigned i = threadIdx.x; i < E; i += THREADS_PER_BLOCK) {
    out_embeddings[i] = embedding_weights[i];
  }
}

void muillm_embedding_forward_xx16_indices_i32(
  hipStream_t stream,
  unsigned N,
  unsigned E,
  const uint16_t* weights,
  const uint32_t* x,
  uint16_t* out_embeddings,
  int simd_lanes
) {
  const unsigned threads_per_blocks = THREADS_PER_BLOCK;

  // TODO: max cuda grid dimension is 65k, so need to do something when N>65k
  const unsigned num_blocks = N;

  embedding_forward_xx16_indices_i32_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    weights,
    x,
    out_embeddings,
    E
  );
}


// expected block dimensions: [x=N]
__global__ void embedding_forward_xx16_indices_i64_kernel(
  // inputs
  const uint16_t* embedding_weights,
  const uint64_t* indices,
  // outputs
  uint16_t* out_embeddings,
  // other arguments
  unsigned E
) {
  // one block copies one embedding
  unsigned idx = blockIdx.x;

  uint64_t embedding_idx = indices[idx];

  // re-align the pointers
  embedding_weights = &embedding_weights[embedding_idx * E];
  out_embeddings = &out_embeddings[idx * E];

  // copy the embedding out
  for (unsigned i = threadIdx.x; i < E; i += THREADS_PER_BLOCK) {
    out_embeddings[i] = embedding_weights[i];
  }
}

void muillm_embedding_forward_xx16_indices_i64(
  hipStream_t stream,
  unsigned N,
  unsigned E,
  const uint16_t* weights,
  const uint64_t* x,
  uint16_t* out_embeddings,
  int simd_lanes
) {
  const unsigned threads_per_blocks = THREADS_PER_BLOCK;

  // TODO: max cuda grid dimension is 65k, so need to do something when N>65k
  const unsigned num_blocks = N;

  embedding_forward_xx16_indices_i64_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    weights,
    x,
    out_embeddings,
    E
  );
}