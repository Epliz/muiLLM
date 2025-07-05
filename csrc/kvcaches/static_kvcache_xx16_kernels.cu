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

// expected block dimensions: [x=num_k_heads+num_v_heads, y=T, z=B]
void __global__ static_kvcache_update_xx16_kernel(
  const xx16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const xx16* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  // KV cache
  xx16* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  xx16* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  const uint64_t* __restrict__ cache_position, // [T]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // k strides
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  // v strides
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride
) {
    // one block does one head of a new token
    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = blockIdx.y; // should be launched with T
    unsigned batch_idx = blockIdx.z;
    unsigned pos_idx = batch_idx * T + tok_idx;

    // determine if we are supposed to transform an embedding from q or k,
    // and which head
    const xx16* __restrict__ X;
    xx16* __restrict__ cache_out;
    unsigned num_heads;

    // strides
    unsigned batch_stride;
    unsigned head_stride;
    unsigned tok_stride;

    // determine if we are processing q, k or v
    if (head_idx < num_k_heads) {
        // k
        X = k_in;
        cache_out = k_cache_out;
        num_heads = num_k_heads;

        batch_stride = k_in_batch_stride;
        head_stride = k_in_head_stride;
        tok_stride = k_in_tok_stride;
    } else {
        // v
        X = v_in;
        cache_out = v_cache_out;
        num_heads = num_v_heads;

        batch_stride = v_in_batch_stride;
        head_stride = v_in_head_stride;
        tok_stride = v_in_tok_stride;

        head_idx -= num_k_heads;
    }

    // realign the pointer to where we are supposed to write out if needed


    // realign embeds_in and embeds_out
    // k/v might be strided, but embedding dimension stride needs to be 1
    unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
    X = &X[embed_in_idx];

    // realign cache_out
    unsigned cache_tok_pos = cache_position[tok_idx];
    cache_out = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      xx16 embed = *addr(X, d);
      *addr(cache_out, d) = embed;
    }
}

void static_kvcache_update_xx16(
  hipStream_t stream,
  const xx16* k_in,
  const xx16* v_in,
  xx16* k_cache_out,
  xx16* v_cache_out,
  const uint64_t* cache_position,
  unsigned B,
  unsigned T,
  unsigned MAX_T,
  unsigned num_k_heads,
  unsigned num_v_heads,
  unsigned embed_dim,
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride
) {
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);


  // TODO: max cuda block dimension is 1024, so need to do something when T>1024
  // expected block dimensions: [x=num_k_heads, y=T, z=B]
  const dim3 num_blocks = dim3(num_k_heads + num_v_heads, T, B);

  static_kvcache_update_xx16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    k_in,
    v_in,
    k_cache_out,
    v_cache_out,
    cache_position,
    B,
    T,
    MAX_T,
    num_k_heads,
    num_v_heads,
    embed_dim,
    k_in_batch_stride,
    k_in_head_stride,
    k_in_tok_stride,
    v_in_batch_stride,
    v_in_head_stride,
    v_in_tok_stride
  );
}