#include <stdint.h>
#include <hip/hip_fp16.h>

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

// expected block dimensions: [x=num_k_heads+num_v_heads, y=MIN(T, MAX_T), z=B]
void __global__ sliding_kvcache_prefill_xx16_kernel(
  const xx16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const xx16* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  // KV cache
  xx16* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  xx16* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned START_T, // starting copied token index
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
    unsigned raw_tok_idx = blockIdx.y;
    // one block does one head of a new token
    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = raw_tok_idx + START_T; // should be launched with COPIED_T = MIN(T, MAX_T)
    unsigned batch_idx = blockIdx.z;

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
    unsigned cache_tok_pos = raw_tok_idx;
    cache_out = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      xx16 embed = *addr(X, d);
      *addr(cache_out, d) = embed;
    }
}

void sliding_kvcache_prefill_xx16(
  hipStream_t stream,
  const xx16* k_in, // shape [B, num_k_heads, T, embed_dim]
  const xx16* v_in, // shape [B, num_v_heads, T, embed_dim]
  // KV cache
  xx16* k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  xx16* v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
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
  // Prefill (we incremented seen_tokens before updating the cache)
  const unsigned COPIED_T = std::min(T, MAX_T);
  const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);
  // We return all tokens in that case to avoid catastrophic forgetting
  // but store in the cache the latest ones
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // expected block dimensions: [x=num_k_heads, y=T, z=B]
  // TODO: max cuda block dimension is 1024, so need to do something when T>1024
  const dim3 num_blocks = dim3(num_k_heads + num_v_heads, COPIED_T, B);

  sliding_kvcache_prefill_xx16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const xx16*)k_in,
    (const xx16*)v_in,
    // KV cache
    (xx16*)k_cache_out,
    (xx16*)v_cache_out,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
    START_T,
    num_k_heads,
    num_v_heads,
    embed_dim,
    // k strides
    k_in_batch_stride,
    k_in_head_stride,
    k_in_tok_stride,
    // v strides
    v_in_batch_stride,
    v_in_head_stride,
    v_in_tok_stride
  );
}

// expected block dimensions: [x=num_k_heads+num_v_heads, y=MAX_T, z=B]
// always have to copy an entire new cache
void __global__ sliding_kvcache_update_overwrite_xx16_kernel(
  const xx16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const xx16* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  // KV cache in
  const xx16* __restrict__ k_cache_in, // [B, num_k_heads, MAX_T, embed_dim]
  const xx16* __restrict__ v_cache_in, // [B, num_v_heads, MAX_T, embed_dim]
  // KV cache out
  xx16* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  xx16* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens in k_in (might be bigger than MAX_T)
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned START_T, // starting copied token index
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
    unsigned raw_tok_idx = blockIdx.y;
    // one block does one head of a token
    unsigned num_new_tokens = (T - START_T);
    unsigned num_old_tokens = (MAX_T - num_new_tokens);
    bool copy_new_tokens = raw_tok_idx >= num_old_tokens;

    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = copy_new_tokens ? (raw_tok_idx - num_old_tokens + START_T) : (raw_tok_idx + num_new_tokens);
    unsigned batch_idx = blockIdx.z;

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
        num_heads = num_k_heads;
        cache_out = k_cache_out;
        if (copy_new_tokens) {
          // copy new tokens
          X = k_in;

          batch_stride = k_in_batch_stride;
          head_stride = k_in_head_stride;
          tok_stride = k_in_tok_stride;

        } else {
          // copy old cache
          X = k_cache_in;

          batch_stride = num_heads * MAX_T * embed_dim;
          head_stride = MAX_T * embed_dim;
          tok_stride = embed_dim;
        }
    } else {
        // v
        num_heads = num_v_heads;
        cache_out = v_cache_out;
        head_idx -= num_k_heads;
        if (copy_new_tokens) {
          // copy new tokens
          X = v_in;

          batch_stride = v_in_batch_stride;
          head_stride = v_in_head_stride;
          tok_stride = v_in_tok_stride;

        } else {
          // copy old cache
          X = v_cache_in;

          batch_stride = num_heads * MAX_T * embed_dim;
          head_stride = MAX_T * embed_dim;
          tok_stride = embed_dim;
        }
    }

    // realign the pointer to where we are supposed to write out if needed


    // realign embeds_in and embeds_out
    // k/v might be strided, but embedding dimension stride needs to be 1
    unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
    X = &X[embed_in_idx];

    // realign cache_out
    unsigned cache_tok_pos = raw_tok_idx;
    cache_out = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      xx16 embed = *addr(X, d);
      *addr(cache_out, d) = embed;
    }
}

void sliding_kvcache_update_overwrite_xx16(
  hipStream_t stream,
  const xx16* k_in, // shape [B, num_k_heads, T, embed_dim]
  const xx16* v_in, // shape [B, num_v_heads, T, embed_dim]
  // KV cache in
  const xx16* k_cache_in, // [B, num_k_heads, MAX_T, embed_dim]
  const xx16* v_cache_in, // [B, num_v_heads, MAX_T, embed_dim]
  // KV cache out
  xx16* k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  xx16* v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  unsigned B, // batch size
  unsigned T, // num new tokens in k_in (might be bigger than MAX_T)
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
  // determine what part of the new tokens to copy
  // if we have a lot of input tokens, we actually copy only the MAX_T last ones
  const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);

  // We return all tokens in that case to avoid catastrophic forgetting
  // but store in the cache the latest ones
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // expected block dimensions: [x=num_k_heads, y=T, z=B]
  // TODO: max cuda block dimension is 1024, so need to do something when MAX_T>1024
  const dim3 num_blocks = dim3(num_k_heads + num_v_heads, MAX_T, B);

  sliding_kvcache_update_overwrite_xx16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const xx16*)k_in,
    (const xx16*)v_in,
    // KV cache in
    (const xx16*)k_cache_in,
    (const xx16*)v_cache_in,
    // KV cache out
    (xx16*)k_cache_out,
    (xx16*)v_cache_out,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
    START_T,
    num_k_heads,
    num_v_heads,
    embed_dim,
    // k strides
    k_in_batch_stride,
    k_in_head_stride,
    k_in_tok_stride,
    // v strides
    v_in_batch_stride,
    v_in_head_stride,
    v_in_tok_stride
  );
}

// expected block dimensions: [x=num_k_heads+num_v_heads, y=T, z=B]
void __global__ sliding_kvcache_update_xx16_kernel(
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
    unsigned raw_tok_idx = blockIdx.y;
    // one block does one head of a new token
    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = raw_tok_idx; // should be launched with T
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

void sliding_kvcache_update_xx16(
  hipStream_t stream,
  const xx16* k_in, // shape [B, num_k_heads, T, embed_dim]
  const xx16* v_in, // shape [B, num_v_heads, T, embed_dim]
  // KV cache
  xx16* k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  xx16* v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  const uint64_t* cache_position, // [T]
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
  // not full, not becoming full -> similar to a static cache update
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // expected block dimensions: [x=num_k_heads, y=T, z=B]
  // TODO: max cuda block dimension is 1024, so need to do something when T>1024
  const dim3 num_blocks = dim3(num_k_heads + num_v_heads, T, B);

  sliding_kvcache_update_xx16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const xx16*)k_in,
    (const xx16*)v_in,
    // KV cache
    (xx16*)k_cache_out,
    (xx16*)v_cache_out,
    (const uint64_t*)cache_position,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
    num_k_heads,
    num_v_heads,
    embed_dim,
    // k strides
    k_in_batch_stride,
    k_in_head_stride,
    k_in_tok_stride,
    // v strides
    v_in_batch_stride,
    v_in_head_stride,
    v_in_tok_stride
  );
}