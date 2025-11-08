#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <stdint.h>

#include "../rope/rotary_position_layout.h"

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

// always launch with T; if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
void __global__ sliding_kvcache_rope_forward_prefill_fp16_kernel(
  const half* __restrict__ cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* __restrict__ sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  half* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  half* __restrict__ k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  half* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned START_T, // starting copied token index
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
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
  unsigned batch_idx = blockIdx.z;

  unsigned tok_idx;
  const half* __restrict__ embeds_in;
  half* __restrict__ embeds_out;
  half* __restrict__ cache_out;
  unsigned num_heads;

  // strides
  unsigned batch_stride;
  unsigned head_stride;
  unsigned tok_stride;

  bool transform_q = false;
  bool transform_k = false;
  bool transform_v = false;

  // determine if we are processing q, k or v
  if (head_idx < num_q_heads) {
    // q
    transform_q = true;
    // need to transform all tokens for q, even if T>MAX_T
    tok_idx = raw_tok_idx;
    embeds_in = q_in;
    embeds_out = q_out;
    cache_out = nullptr; // no q cache
    num_heads = num_q_heads;

    batch_stride = q_in_batch_stride;
    head_stride = q_in_head_stride;
    tok_stride = q_in_tok_stride;
  } else if (head_idx < num_q_heads + num_k_heads) {
    // k
    transform_k = true;
    // need to transform all tokens for k, even if T>MAX_T
    tok_idx = raw_tok_idx;
    embeds_in = k_in;
    embeds_out = k_out;
    cache_out = k_cache_out;
    num_heads = num_k_heads;

    batch_stride = k_in_batch_stride;
    head_stride = k_in_head_stride;
    tok_stride = k_in_tok_stride;

    head_idx -= num_q_heads;
  } else {
    // v
    transform_v = true;
    // for v, we only transform the last MAX_T tokens
    tok_idx = raw_tok_idx + START_T;
    embeds_in = v_in;
    embeds_out = nullptr;
    cache_out = v_cache_out;
    num_heads = num_v_heads;

    batch_stride = v_in_batch_stride;
    head_stride = v_in_head_stride;
    tok_stride = v_in_tok_stride;

    head_idx -= (num_q_heads + num_k_heads);
  }

  // realign embeds_in
  unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
  embeds_in = &embeds_in[embed_in_idx];

  // Apply RoPE to q and k, or just copy v
  if (transform_q || transform_k) {
    // Apply RoPE to q or k
    unsigned pos_idx = batch_idx * T + tok_idx;
    
    // ROTARY_CACHE_BTE_LAYOUT
    uint64_t position_id = tok_idx < T ? pos_idx : 0;

    // realign the cos/sin caches to the position
    const half* cos_pos = &cos_cached[position_id * embed_dim];
    const half* sin_pos = &sin_cached[position_id * embed_dim];

    // realign embeds_out
    if (embeds_out != nullptr) {
      unsigned embed_out_idx = ((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0;
      embeds_out = &embeds_out[embed_out_idx];
    }

    if (transform_q) {
      // transform q: write embeds out, but no cache write
      unsigned half_embed_dim = embed_dim / 2;

      // first half
      const half* __restrict__ rot_embeds_in = &embeds_in[half_embed_dim];
      
      unsigned d = threadIdx.x;
      for (; d < half_embed_dim; d += THREADS_PER_BLOCK) {
        half cos_val = *addr(cos_pos, d);
        half sin_val = *addr(sin_pos, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = __hneg(*addr(rot_embeds_in, d));
    
        half r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;
      }

      // second half
      rot_embeds_in = &embeds_in[(int)-half_embed_dim];
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        half cos_val = *addr(cos_pos, d);
        half sin_val = *addr(sin_pos, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = *addr(rot_embeds_in, d);

        half r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;
      }

    } else {
      // transform k: write embeds out, and write to cache

      // realign cache_out
      unsigned cache_tok_pos = raw_tok_idx - START_T;
      half* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

      bool valid_cache_write = (raw_tok_idx >= START_T) && (raw_tok_idx < START_T + MAX_T);

      unsigned half_embed_dim = embed_dim / 2;

      // first half
      const half* __restrict__ rot_embeds_in = &embeds_in[half_embed_dim];

      unsigned d = threadIdx.x;
      for (; d < half_embed_dim; d += THREADS_PER_BLOCK) {
        half cos_val = *addr(cos_pos, d);
        half sin_val = *addr(sin_pos, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = __hneg(*addr(rot_embeds_in, d));
    
        half r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;

        // write the rotated token to cache
        if (valid_cache_write) {
          *addr(cache_out_write, d) = r;
        }
      }

      // second half
      rot_embeds_in = &embeds_in[(int)-half_embed_dim];
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        half cos_val = *addr(cos_pos, d);
        half sin_val = *addr(sin_pos, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = *addr(rot_embeds_in, d);

        half r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;

        // write the rotated token to cache
        if (valid_cache_write) {
          *addr(cache_out_write, d) = r;
        }
      }
    }
    
  } else {
    // For v, just copy without RoPE
    
    unsigned cache_tok_pos = raw_tok_idx;

    if (cache_tok_pos >= MAX_T) {
      // out of range, nothing to do
      return;
    }

    // realign cache_out
    half* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
    
    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      half embed = *addr(embeds_in, d);
      *addr(cache_out_write, d) = embed;
    }
  }
}

void sliding_kvcache_rope_forward_prefill_fp16(
  hipStream_t stream,
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  // k strides
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  // v strides
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride,
  const half* cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* v_in, // shape [B, num_v_heads, T, embed_dim]
  half* q_out, // shape [B, num_q_heads, T, embed_dim]
  half* k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  half* k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* v_cache_out // [B, num_v_heads, MAX_T, embed_dim]
) {
  // Prefill (we incremented seen_tokens before updating the cache)
  const unsigned COPIED_T = std::min(T, MAX_T);
  const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);
  // We return all tokens in that case to avoid catastrophic forgetting
  // but store in the cache the latest ones
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // always launch with T; if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
  // expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, T, B);

  sliding_kvcache_rope_forward_prefill_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    cos_cached,
    sin_cached,
    q_in,
    k_in,
    v_in,
    q_out,
    k_out,
    // KV cache
    k_cache_out,
    v_cache_out,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
    START_T,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    embed_dim,
    // q strides
    q_in_batch_stride,
    q_in_head_stride,
    q_in_tok_stride,
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

// always launch with T; if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=MIN(T, MAX_T), z=B]
void __global__ sliding_kvcache_rope_forward_prefill_bf16_kernel(
  const __hip_bfloat16* __restrict__ cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* __restrict__ sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  __hip_bfloat16* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  __hip_bfloat16* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned START_T, // starting copied token index
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
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
  unsigned head_idx = blockIdx.x;
  unsigned batch_idx = blockIdx.z;

  unsigned tok_idx;
  const __hip_bfloat16* __restrict__ embeds_in;
  __hip_bfloat16* __restrict__ embeds_out;
  __hip_bfloat16* __restrict__ cache_out;
  unsigned num_heads;

  unsigned batch_stride;
  unsigned head_stride;
  unsigned tok_stride;

  bool transform_q = false;
  bool transform_k = false;
  bool transform_v = false;

  if (head_idx < num_q_heads) {
    transform_q = true;
    // need to transform all tokens for q, even if T>MAX_T
    tok_idx = raw_tok_idx;
    embeds_in = q_in;
    embeds_out = q_out;
    cache_out = nullptr; // no q cache
    num_heads = num_q_heads;

    batch_stride = q_in_batch_stride;
    head_stride = q_in_head_stride;
    tok_stride = q_in_tok_stride;
  } else if (head_idx < num_q_heads + num_k_heads) {
    transform_k = true;
    // need to transform all tokens for k, even if T>MAX_T
    tok_idx = raw_tok_idx;
    embeds_in = k_in;
    embeds_out = k_out;
    cache_out = k_cache_out;
    num_heads = num_k_heads;

    batch_stride = k_in_batch_stride;
    head_stride = k_in_head_stride;
    tok_stride = k_in_tok_stride;

    head_idx -= num_q_heads;
  } else {
    transform_v = true;
    // for v, we only transform the last MAX_T tokens
    tok_idx = raw_tok_idx + START_T;
    embeds_in = v_in;
    embeds_out = nullptr;
    cache_out = v_cache_out;
    num_heads = num_v_heads;

    batch_stride = v_in_batch_stride;
    head_stride = v_in_head_stride;
    tok_stride = v_in_tok_stride;

    head_idx -= (num_q_heads + num_k_heads);
  }

  // realign embeds_in
  unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
  embeds_in = &embeds_in[embed_in_idx];

  // Apply RoPE to q and k, or just copy v
  if (transform_q || transform_k) {
    // Apply RoPE to q or k
    unsigned pos_idx = batch_idx * T + tok_idx;
    
    // ROTARY_CACHE_BTE_LAYOUT
    uint64_t position_id = tok_idx < T ? pos_idx : 0;

    // realign the cos/sin caches to the position
    const __hip_bfloat16* cos_pos = &cos_cached[position_id * embed_dim];
    const __hip_bfloat16* sin_pos = &sin_cached[position_id * embed_dim];

    // realign embeds_out
    if (embeds_out != nullptr) {
      unsigned embed_out_idx = ((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0;
      embeds_out = &embeds_out[embed_out_idx];
    }

    if (transform_q) {
      // transform q: write embeds out, but no cache write
      unsigned half_embed_dim = embed_dim / 2;

      // first half
      const __hip_bfloat16* __restrict__ rot_embeds_in = &embeds_in[half_embed_dim];

      unsigned d = threadIdx.x;
      for (; d < half_embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 cos_val = *addr(cos_pos, d);
        __hip_bfloat16 sin_val = *addr(sin_pos, d);

        __hip_bfloat16 embed = *addr(embeds_in, d);
        __hip_bfloat16 rot_embed = __hneg(*addr(rot_embeds_in, d));
      
        __hip_bfloat16 r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;
      }

      // second half
      rot_embeds_in = &embeds_in[(int)-half_embed_dim];
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 cos_val = *addr(cos_pos, d);
        __hip_bfloat16 sin_val = *addr(sin_pos, d);

        __hip_bfloat16 embed = *addr(embeds_in, d);
        __hip_bfloat16 rot_embed = *addr(rot_embeds_in, d);

        __hip_bfloat16 r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;
      }
      
    } else {
      // transform k: write embeds out, and write to cache

      // realign cache_out
      unsigned cache_tok_pos = raw_tok_idx - START_T;
      __hip_bfloat16* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
      
      bool valid_cache_write = (raw_tok_idx >= START_T) && (raw_tok_idx < START_T + MAX_T);

      unsigned half_embed_dim = embed_dim / 2;

      // first half
      const __hip_bfloat16* __restrict__ rot_embeds_in = &embeds_in[half_embed_dim];

      unsigned d = threadIdx.x;
      for (; d < half_embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 cos_val = *addr(cos_pos, d);
        __hip_bfloat16 sin_val = *addr(sin_pos, d);

        __hip_bfloat16 embed = *addr(embeds_in, d);
        __hip_bfloat16 rot_embed = __hneg(*addr(rot_embeds_in, d));
      
        __hip_bfloat16 r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;

        if (valid_cache_write) {
          *addr(cache_out_write, d) = r;
        }
      }

      // second half
      rot_embeds_in = &embeds_in[(int)-half_embed_dim];
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 cos_val = *addr(cos_pos, d);
        __hip_bfloat16 sin_val = *addr(sin_pos, d);

        __hip_bfloat16 embed = *addr(embeds_in, d);
        __hip_bfloat16 rot_embed = *addr(rot_embeds_in, d);

        __hip_bfloat16 r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;

        if (valid_cache_write) {
          *addr(cache_out_write, d) = r;
        }
      }
    }
  } else {
    // transform v: just copy without RoPE
    unsigned cache_tok_pos = raw_tok_idx;

    if (cache_tok_pos >= MAX_T) {
      // out of range, nothing to do
      return;
    }

    // realign cache_out
    __hip_bfloat16* cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      __hip_bfloat16 embed = *addr(embeds_in, d);
      *addr(cache_out_write, d) = embed;
    }
  }
}

void sliding_kvcache_rope_forward_prefill_bf16(
  hipStream_t stream,
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  // k strides
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  // v strides
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride,
  const __hip_bfloat16* cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* q_out, // shape [B, num_q_heads, T, embed_dim]
  __hip_bfloat16* k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  __hip_bfloat16* k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  __hip_bfloat16* v_cache_out // [B, num_v_heads, MAX_T, embed_dim]
) {
  // Prefill (we incremented seen_tokens before updating the cache)
  const unsigned COPIED_T = std::min(T, MAX_T);
  const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);
  // We return all tokens in that case to avoid catastrophic forgetting
  // but store in the cache the latest ones
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // always launch with T; if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
  // expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, T, B);

  sliding_kvcache_rope_forward_prefill_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    cos_cached,
    sin_cached,
    q_in,
    k_in,
    v_in,
    q_out,
    k_out,
    // KV cache
    k_cache_out,
    v_cache_out,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
    START_T,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    embed_dim,
    // q strides
    q_in_batch_stride,
    q_in_head_stride,
    q_in_tok_stride,
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

// always launch with max(T, MAX_T); if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(T, MAX_T), z=B]
// applies RoPE to q,k and copies rotated k,v to sliding cache
void __global__ sliding_kvcache_rope_forward_update_overwrite_fp16_kernel(
  const half* __restrict__ cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* __restrict__ sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  half* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache in
  const half* __restrict__ k_cache_in, // [B, num_k_heads, MAX_T, embed_dim]
  const half* __restrict__ v_cache_in, // [B, num_v_heads, MAX_T, embed_dim]
  // KV cache out
  half* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens in k_in (might be bigger than MAX_T)
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned START_T, // starting copied token index
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
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
  bool copy_new_tokens = raw_tok_idx >= num_old_tokens; // for k and v

  unsigned head_idx = blockIdx.x;
  unsigned batch_idx = blockIdx.z;

  // determine if we are supposed to transform an embedding from q, k, or v
  unsigned tok_idx;
  const half* __restrict__ embeds_in;
  half* __restrict__ embeds_out;
  const half* __restrict__ prev_cache;
  half* __restrict__ cache_out;
  unsigned num_heads;

  // strides
  unsigned batch_stride;
  unsigned head_stride;
  unsigned tok_stride;

  bool transform_q = false;
  bool transform_k = false;
  bool transform_v = false;
  // determine if we are processing q, k or v
  if (head_idx < num_q_heads) {
    // q
    transform_q = true;
    tok_idx = raw_tok_idx; // always copy all tokens for q
    embeds_in = q_in;
    embeds_out = q_out;
    cache_out = nullptr; // no q cache
    num_heads = num_q_heads;
    
    batch_stride = q_in_batch_stride;
    head_stride = q_in_head_stride;
    tok_stride = q_in_tok_stride;
  } else if (head_idx < num_q_heads + num_k_heads) {
    // k
    transform_k = true;
    num_heads = num_k_heads;
    cache_out = k_cache_out;
    head_idx -= num_q_heads;
    
    if (copy_new_tokens) {
      // copy new tokens
      tok_idx = (raw_tok_idx - num_old_tokens + START_T);
      embeds_in = k_in;
      embeds_out = nullptr;
      
      batch_stride = k_in_batch_stride;
      head_stride = k_in_head_stride;
      tok_stride = k_in_tok_stride;
    } else {
      // copy old cache
      tok_idx = (raw_tok_idx + num_new_tokens);
      embeds_in = k_cache_in;
      embeds_out = nullptr;
      
      batch_stride = num_heads * MAX_T * embed_dim;
      head_stride = MAX_T * embed_dim;
      tok_stride = embed_dim;
    }
  } else {
    // v
    transform_v = true;
    num_heads = num_v_heads;
    cache_out = v_cache_out;
    head_idx -= (num_q_heads + num_k_heads);
    
    if (copy_new_tokens) {
      // copy new tokens
      tok_idx = (raw_tok_idx - num_old_tokens + START_T);
      embeds_in = v_in;
      embeds_out = nullptr;
      
      batch_stride = v_in_batch_stride;
      head_stride = v_in_head_stride;
      tok_stride = v_in_tok_stride;
    } else {
      // copy old cache
      tok_idx = (raw_tok_idx + num_new_tokens);
      embeds_in = v_cache_in;
      embeds_out = nullptr;
      
      batch_stride = num_heads * MAX_T * embed_dim;
      head_stride = MAX_T * embed_dim;
      tok_stride = embed_dim;
    }
  }

  // realign embeds_in
  unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
  embeds_in = &embeds_in[embed_in_idx];

  // realign cache_out
  half* __restrict__ cache_out_write = nullptr;
  if (cache_out != nullptr) {
    unsigned cache_tok_pos = raw_tok_idx;
    cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
  }

  // Apply RoPE to q and k, or just copy v
  if (transform_q || (transform_k && copy_new_tokens)) {
    // Apply RoPE to q (all tokens) or k (new tokens only)
    unsigned pos_idx = batch_idx * T + tok_idx;
    
    // ROTARY_CACHE_BTE_LAYOUT
    uint64_t position_id = tok_idx < T ? pos_idx : 0;

    // realign the cos/sin caches to the position
    const half* cos_pos = &cos_cached[position_id * embed_dim];
    const half* sin_pos = &sin_cached[position_id * embed_dim];

    if (transform_q) {
      // transform q: write embeds out, but no cache write
      if (tok_idx >= T) {
        // out of range, nothing to do
        return;
      }

      // realign embeds_out for q
      embeds_out = &embeds_out[((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0];
      
      unsigned half_embed_dim = embed_dim / 2;

      // first half
      const half* __restrict__ rot_embeds_in = &embeds_in[half_embed_dim];
      
      unsigned d = threadIdx.x;
      for (; d < half_embed_dim; d += THREADS_PER_BLOCK) {
        half cos_val = *addr(cos_pos, d);
        half sin_val = *addr(sin_pos, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = __hneg(*addr(rot_embeds_in, d));
    
        half r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;
      }

      // second half
      rot_embeds_in = &embeds_in[(int)-half_embed_dim];
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        half cos_val = *addr(cos_pos, d);
        half sin_val = *addr(sin_pos, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = *addr(rot_embeds_in, d);

        half r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;
      }
    } else {
      // transform k: no write embeds out, and write to cache
      unsigned cache_tok_pos = raw_tok_idx;
      if (cache_tok_pos >= MAX_T) {
        // out of range, nothing to do
        return;
      }

      // realign cache_out
      half* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

      unsigned half_embed_dim = embed_dim / 2;

      // first half
      const half* __restrict__ rot_embeds_in = &embeds_in[half_embed_dim];
      
      unsigned d = threadIdx.x;
      for (; d < half_embed_dim; d += THREADS_PER_BLOCK) {
        half cos_val = *addr(cos_pos, d);
        half sin_val = *addr(sin_pos, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = __hneg(*addr(rot_embeds_in, d));
    
        half r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        // write the rotated token to cache
        *addr(cache_out_write, d) = r;
      }

      // second half
      rot_embeds_in = &embeds_in[(int)-half_embed_dim];
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        half cos_val = *addr(cos_pos, d);
        half sin_val = *addr(sin_pos, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = *addr(rot_embeds_in, d);

        half r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        // write the rotated token to cache
        *addr(cache_out_write, d) = r;
      }
    }
  } else {
    // For v or copying old k cache, just copy without RoPE
    unsigned cache_tok_pos = raw_tok_idx;
    if (cache_tok_pos >= MAX_T) {
      // out of range, nothing to do
      return;
    }

    // realign cache_out
    half* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
    
    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      half embed = *addr(embeds_in, d);
      *addr(cache_out_write, d) = embed;
    }
  }
}

void sliding_kvcache_rope_forward_update_overwrite_fp16(
  hipStream_t stream,
  unsigned B, // batch size
  unsigned T, // num new tokens in k_in (might be bigger than MAX_T)
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned num_q_heads, // number of heads for k
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  // k strides
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  // v strides
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride,
  const half* cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* v_in, // shape [B, num_v_heads, T, embed_dim]
  half* q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache in
  const half* k_cache_in, // [B, num_k_heads, MAX_T, embed_dim]
  const half* v_cache_in, // [B, num_v_heads, MAX_T, embed_dim]
  // KV cache out
  half* k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* v_cache_out // [B, num_v_heads, MAX_T, embed_dim]
) {
  // determine what part of the new tokens to copy
  // if we have a lot of input tokens, we actually copy only the MAX_T last ones
  const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // always launch with max(T, MAX_T); if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
  // expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(T, MAX_T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, max(T, MAX_T), B);

  sliding_kvcache_rope_forward_update_overwrite_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    cos_cached,
    sin_cached,
    q_in,
    k_in,
    v_in,
    q_out,
    // KV cache in
    k_cache_in,
    v_cache_in,
    // KV cache out
    k_cache_out,
    v_cache_out,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
    START_T,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    embed_dim,
    // q strides
    q_in_batch_stride,
    q_in_head_stride,
    q_in_tok_stride,
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

// always launch with max(T, MAX_T); if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(T, MAX_T), z=B]
// applies RoPE to q,k and copies rotated k,v to sliding cache
void __global__ sliding_kvcache_rope_forward_update_overwrite_bf16_kernel(
  const __hip_bfloat16* __restrict__ cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* __restrict__ sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache in
  const __hip_bfloat16* __restrict__ k_cache_in, // [B, num_k_heads, MAX_T, embed_dim]
  const __hip_bfloat16* __restrict__ v_cache_in, // [B, num_v_heads, MAX_T, embed_dim]
  // KV cache out
  __hip_bfloat16* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  __hip_bfloat16* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens in k_in (might be bigger than MAX_T)
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned START_T, // starting copied token index
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
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
  bool copy_new_tokens = raw_tok_idx >= num_old_tokens; // for k and v

  unsigned head_idx = blockIdx.x;
  unsigned batch_idx = blockIdx.z;

  // determine if we are supposed to transform an embedding from q, k, or v
  unsigned tok_idx;
  const __hip_bfloat16* __restrict__ embeds_in;
  __hip_bfloat16* __restrict__ embeds_out;
  const __hip_bfloat16* __restrict__ prev_cache;
  __hip_bfloat16* __restrict__ cache_out;
  unsigned num_heads;

  // strides
  unsigned batch_stride;
  unsigned head_stride;
  unsigned tok_stride;

  bool transform_q = false;
  bool transform_k = false;
  bool transform_v = false;
  // determine if we are processing q, k or v
  if (head_idx < num_q_heads) {
    // q
    transform_q = true;
    tok_idx = raw_tok_idx; // always copy all tokens for q
    embeds_in = q_in;
    embeds_out = q_out;
    cache_out = nullptr; // no q cache
    num_heads = num_q_heads;
    
    batch_stride = q_in_batch_stride;
    head_stride = q_in_head_stride;
    tok_stride = q_in_tok_stride;
  } else if (head_idx < num_q_heads + num_k_heads) {
    // k
    transform_k = true;
    num_heads = num_k_heads;
    cache_out = k_cache_out;
    head_idx -= num_q_heads;
    
    if (copy_new_tokens) {
      // copy new tokens
      tok_idx = (raw_tok_idx - num_old_tokens + START_T);
      embeds_in = k_in;
      embeds_out = nullptr;
      
      batch_stride = k_in_batch_stride;
      head_stride = k_in_head_stride;
      tok_stride = k_in_tok_stride;
    } else {
      // copy old cache
      tok_idx = (raw_tok_idx + num_new_tokens);
      embeds_in = k_cache_in;
      embeds_out = nullptr;
      
      batch_stride = num_heads * MAX_T * embed_dim;
      head_stride = MAX_T * embed_dim;
      tok_stride = embed_dim;
    }
  } else {
    // v
    transform_v = true;
    num_heads = num_v_heads;
    cache_out = v_cache_out;
    head_idx -= (num_q_heads + num_k_heads);
    
    if (copy_new_tokens) {
      // copy new tokens
      tok_idx = (raw_tok_idx - num_old_tokens + START_T);
      embeds_in = v_in;
      embeds_out = nullptr;
      
      batch_stride = v_in_batch_stride;
      head_stride = v_in_head_stride;
      tok_stride = v_in_tok_stride;
    } else {
      // copy old cache
      tok_idx = (raw_tok_idx + num_new_tokens);
      embeds_in = v_cache_in;
      embeds_out = nullptr;
      
      batch_stride = num_heads * MAX_T * embed_dim;
      head_stride = MAX_T * embed_dim;
      tok_stride = embed_dim;
    }
  }

  // realign embeds_in
  unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
  embeds_in = &embeds_in[embed_in_idx];

  // realign cache_out
  __hip_bfloat16* __restrict__ cache_out_write = nullptr;
  if (cache_out != nullptr) {
    unsigned cache_tok_pos = raw_tok_idx;
    cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
  }

  // Apply RoPE to q and k, or just copy v
  if (transform_q || (transform_k && copy_new_tokens)) {
    // Apply RoPE to q (all tokens) or k (new tokens only)
    unsigned pos_idx = batch_idx * T + tok_idx;
    
    // ROTARY_CACHE_BTE_LAYOUT
    uint64_t position_id = tok_idx < T ? pos_idx : 0;

    // realign the cos/sin caches to the position
    const __hip_bfloat16* cos_pos = &cos_cached[position_id * embed_dim];
    const __hip_bfloat16* sin_pos = &sin_cached[position_id * embed_dim];

    if (transform_q) {
      // transform q: write embeds out, but no cache write
      if (tok_idx >= T) {
        // out of range, nothing to do
        return;
      }

      // realign embeds_out for q
      embeds_out = &embeds_out[((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0];
      
      unsigned half_embed_dim = embed_dim / 2;

      // first half
      const __hip_bfloat16* __restrict__ rot_embeds_in = &embeds_in[half_embed_dim];
      
      unsigned d = threadIdx.x;
      for (; d < half_embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 cos_val = *addr(cos_pos, d);
        __hip_bfloat16 sin_val = *addr(sin_pos, d);

        __hip_bfloat16 embed = *addr(embeds_in, d);
        __hip_bfloat16 rot_embed = __hneg(*addr(rot_embeds_in, d));

        __hip_bfloat16 r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;
      }

      // second half
      rot_embeds_in = &embeds_in[(int)-half_embed_dim];
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 cos_val = *addr(cos_pos, d);
        __hip_bfloat16 sin_val = *addr(sin_pos, d);

        __hip_bfloat16 embed = *addr(embeds_in, d);
        __hip_bfloat16 rot_embed = *addr(rot_embeds_in, d);

        __hip_bfloat16 r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        *addr(embeds_out, d) = r;
      }
    } else {
      // transform k: no write embeds out, and write to cache
      unsigned cache_tok_pos = raw_tok_idx;
      if (cache_tok_pos >= MAX_T) {
        // out of range, nothing to do
        return;
      }

      // realign cache_out
      __hip_bfloat16* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

      unsigned half_embed_dim = embed_dim / 2;

      // first half
      const __hip_bfloat16* __restrict__ rot_embeds_in = &embeds_in[half_embed_dim];
      
      unsigned d = threadIdx.x;
      for (; d < half_embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 cos_val = *addr(cos_pos, d);
        __hip_bfloat16 sin_val = *addr(sin_pos, d);

        __hip_bfloat16 embed = *addr(embeds_in, d);
        __hip_bfloat16 rot_embed = __hneg(*addr(rot_embeds_in, d));

        __hip_bfloat16 r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        // write the rotated token to cache
        *addr(cache_out_write, d) = r;
      }

      // second half
      rot_embeds_in = &embeds_in[(int)-half_embed_dim];
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 cos_val = *addr(cos_pos, d);
        __hip_bfloat16 sin_val = *addr(sin_pos, d);

        __hip_bfloat16 embed = *addr(embeds_in, d);
        __hip_bfloat16 rot_embed = *addr(rot_embeds_in, d);

        __hip_bfloat16 r = __hfma(rot_embed, sin_val, __hmul(embed, cos_val));

        // write the rotated token to cache
        *addr(cache_out_write, d) = r;
      }
    }
  } else {
    // For v or copying old k cache, just copy without RoPE
    unsigned cache_tok_pos = raw_tok_idx;
    if (cache_tok_pos >= MAX_T) {
      // out of range, nothing to do
      return;
    }

    // realign cache_out
    __hip_bfloat16* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
    
    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      __hip_bfloat16 embed = *addr(embeds_in, d);

      *addr(cache_out_write, d) = embed;
    }
  }
}

void sliding_kvcache_rope_forward_update_overwrite_bf16(
  hipStream_t stream,
  unsigned B, // batch size
  unsigned T, // num new tokens in k_in (might be bigger than MAX_T)
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned num_q_heads, // number of heads for k
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  // k strides
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  // v strides
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride,
  const __hip_bfloat16* cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache in
  const __hip_bfloat16* k_cache_in, // [B, num_k_heads, MAX_T, embed_dim]
  const __hip_bfloat16* v_cache_in, // [B, num_v_heads, MAX_T, embed_dim]
  // KV cache out
  __hip_bfloat16* k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  __hip_bfloat16* v_cache_out // [B, num_v_heads, MAX_T, embed_dim]
) {
  // determine what part of the new tokens to copy
  // if we have a lot of input tokens, we actually copy only the MAX_T last ones
  const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // always launch with max(T, MAX_T); if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
  // expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(T, MAX_T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, max(T, MAX_T), B);

  sliding_kvcache_rope_forward_update_overwrite_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    cos_cached,
    sin_cached,
    q_in,
    k_in,
    v_in,
    q_out,
    // KV cache in
    k_cache_in,
    v_cache_in,
    // KV cache out
    k_cache_out,
    v_cache_out,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
    START_T,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    embed_dim,
    // q strides
    q_in_batch_stride,
    q_in_head_stride,
    q_in_tok_stride,
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

// always launch with T; if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
void __global__ sliding_kvcache_complex_rope_forward_prefill_fp16_kernel(
  const float* __restrict__ position_embeds, // shape [B, T, embed_dim // 2, 2]
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  half* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  half* __restrict__ k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  half* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned START_T, // starting copied token index
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
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
  unsigned batch_idx = blockIdx.z;

  unsigned tok_idx;
  const half* __restrict__ embeds_in;
  half* __restrict__ embeds_out;
  half* __restrict__ cache_out;
  unsigned num_heads;

  // strides
  unsigned batch_stride;
  unsigned head_stride;
  unsigned tok_stride;

  bool transform_q = false;
  bool transform_k = false;
  bool transform_v = false;

  // determine if we are processing q, k or v
  if (head_idx < num_q_heads) {
    // q
    transform_q = true;
    // need to transform all tokens for q, even if T>MAX_T
    tok_idx = raw_tok_idx;
    embeds_in = q_in;
    embeds_out = q_out;
    cache_out = nullptr; // no q cache
    num_heads = num_q_heads;

    batch_stride = q_in_batch_stride;
    head_stride = q_in_head_stride;
    tok_stride = q_in_tok_stride;
  } else if (head_idx < num_q_heads + num_k_heads) {
    // k
    transform_k = true;
    // need to transform all tokens for k, even if T>MAX_T
    tok_idx = raw_tok_idx;
    embeds_in = k_in;
    embeds_out = k_out;
    cache_out = k_cache_out;
    num_heads = num_k_heads;

    batch_stride = k_in_batch_stride;
    head_stride = k_in_head_stride;
    tok_stride = k_in_tok_stride;

    head_idx -= num_q_heads;
  } else {
    // v
    transform_v = true;
    // for v, we only transform the last MAX_T tokens
    tok_idx = raw_tok_idx + START_T;
    embeds_in = v_in;
    embeds_out = nullptr;
    cache_out = v_cache_out;
    num_heads = num_v_heads;

    batch_stride = v_in_batch_stride;
    head_stride = v_in_head_stride;
    tok_stride = v_in_tok_stride;

    head_idx -= (num_q_heads + num_k_heads);
  }

  // realign embeds_in
  unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
  embeds_in = &embeds_in[embed_in_idx];

  // Apply RoPE to q and k, or just copy v
  if (transform_q || transform_k) {
    // Apply RoPE to q or k
    unsigned pos_idx = batch_idx * T + tok_idx;

    // realign the position_embeddings to the position
    position_embeds = &position_embeds[pos_idx * embed_dim];

    // realign embeds_out
    if (embeds_out != nullptr) {
      unsigned embed_out_idx = ((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0;
      embeds_out = &embeds_out[embed_out_idx];
    }

    if (transform_q) {
      // transform q: write embeds out, but no cache write
      unsigned d = 2 * threadIdx.x;
      for (; d < embed_dim; d += 2 * THREADS_PER_BLOCK) {
        float2 cos_sin = *(const float2*)addr(position_embeds, d);
        float cos = cos_sin.x;
        float sin = cos_sin.y;

        float2 embed = __half22float2(*(const half2*)addr(embeds_in, d));
        float real_embed = embed.x;
        float imag_embed = embed.y;

        float real_rot_embed = fma(real_embed, cos, -imag_embed * sin);
        float imag_rot_embed = fma(real_embed, sin, imag_embed * cos);

        half2 rot_embed = __float22half2_rn(make_float2(real_rot_embed, imag_rot_embed));

        *((half2*)addr(embeds_out, d)) = rot_embed;
      }

    } else {
      // transform k: write embeds out, and write to cache

      // realign cache_out
      unsigned cache_tok_pos = raw_tok_idx - START_T;
      half* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

      bool valid_cache_write = (raw_tok_idx >= START_T) && (raw_tok_idx < START_T + MAX_T);
      unsigned d = 2 * threadIdx.x;
      for (; d < embed_dim; d += 2 * THREADS_PER_BLOCK) {
        float2 cos_sin = *(const float2*)addr(position_embeds, d);
        float cos = cos_sin.x;
        float sin = cos_sin.y;

        float2 embed = __half22float2(*(const half2*)addr(embeds_in, d));
        float real_embed = embed.x;
        float imag_embed = embed.y;

        float real_rot_embed = fma(real_embed, cos, -imag_embed * sin);
        float imag_rot_embed = fma(real_embed, sin, imag_embed * cos);

        half2 rot_embed = __float22half2_rn(make_float2(real_rot_embed, imag_rot_embed));

        *((half2*)addr(embeds_out, d)) = rot_embed;

        // write the new token in cache
        if (valid_cache_write) {
          *((half2*)addr(cache_out_write, d)) = rot_embed;
        }
      }
    }
    
  } else {
    // For v, just copy without RoPE
    
    unsigned cache_tok_pos = raw_tok_idx;

    if (cache_tok_pos >= MAX_T) {
      // out of range, nothing to do
      return;
    }

    // realign cache_out
    half* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
    
    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      half embed = *addr(embeds_in, d);
      *addr(cache_out_write, d) = embed;
    }
  }
}

void sliding_kvcache_complex_rope_forward_prefill_fp16(
  hipStream_t stream,
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  // k strides
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  // v strides
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride,
  const float* position_embeddings, // shape [B, T, embed_dim // 2, 2]
  const half* q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* v_in, // shape [B, num_v_heads, T, embed_dim]
  half* q_out, // shape [B, num_q_heads, T, embed_dim]
  half* k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  half* k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* v_cache_out // [B, num_v_heads, MAX_T, embed_dim]
) {
  // Prefill (we incremented seen_tokens before updating the cache)
  const unsigned COPIED_T = std::min(T, MAX_T);
  const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);
  // We return all tokens in that case to avoid catastrophic forgetting
  // but store in the cache the latest ones
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // always launch with T; if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
  // expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, T, B);

  sliding_kvcache_complex_rope_forward_prefill_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    position_embeddings,
    q_in,
    k_in,
    v_in,
    q_out,
    k_out,
    // KV cache
    k_cache_out,
    v_cache_out,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
    START_T,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    embed_dim,
    // q strides
    q_in_batch_stride,
    q_in_head_stride,
    q_in_tok_stride,
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

// always launch with T; if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
void __global__ sliding_kvcache_complex_rope_forward_prefill_bf16_kernel(
  const float* __restrict__ position_embeds, // shape [B, T, embed_dim // 2, 2]
  const __hip_bfloat16* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  __hip_bfloat16* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  __hip_bfloat16* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned START_T, // starting copied token index
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
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
  unsigned batch_idx = blockIdx.z;

  unsigned tok_idx;
  const __hip_bfloat16* __restrict__ embeds_in;
  __hip_bfloat16* __restrict__ embeds_out;
  __hip_bfloat16* __restrict__ cache_out;
  unsigned num_heads;

  // strides
  unsigned batch_stride;
  unsigned head_stride;
  unsigned tok_stride;

  bool transform_q = false;
  bool transform_k = false;
  bool transform_v = false;

  // determine if we are processing q, k or v
  if (head_idx < num_q_heads) {
    // q
    transform_q = true;
    // need to transform all tokens for q, even if T>MAX_T
    tok_idx = raw_tok_idx;
    embeds_in = q_in;
    embeds_out = q_out;
    cache_out = nullptr; // no q cache
    num_heads = num_q_heads;

    batch_stride = q_in_batch_stride;
    head_stride = q_in_head_stride;
    tok_stride = q_in_tok_stride;
  } else if (head_idx < num_q_heads + num_k_heads) {
    // k
    transform_k = true;
    // need to transform all tokens for k, even if T>MAX_T
    tok_idx = raw_tok_idx;
    embeds_in = k_in;
    embeds_out = k_out;
    cache_out = k_cache_out;
    num_heads = num_k_heads;

    batch_stride = k_in_batch_stride;
    head_stride = k_in_head_stride;
    tok_stride = k_in_tok_stride;

    head_idx -= num_q_heads;
  } else {
    // v
    transform_v = true;
    // for v, we only transform the last MAX_T tokens
    tok_idx = raw_tok_idx + START_T;
    embeds_in = v_in;
    embeds_out = nullptr;
    cache_out = v_cache_out;
    num_heads = num_v_heads;

    batch_stride = v_in_batch_stride;
    head_stride = v_in_head_stride;
    tok_stride = v_in_tok_stride;

    head_idx -= (num_q_heads + num_k_heads);
  }

  // realign embeds_in
  unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
  embeds_in = &embeds_in[embed_in_idx];

  // Apply RoPE to q and k, or just copy v
  if (transform_q || transform_k) {
    // Apply RoPE to q or k
    unsigned pos_idx = batch_idx * T + tok_idx;

    // realign the position_embeddings to the position
    position_embeds = &position_embeds[pos_idx * embed_dim];

    // realign embeds_out
    if (embeds_out != nullptr) {
      unsigned embed_out_idx = ((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0;
      embeds_out = &embeds_out[embed_out_idx];
    }

    if (transform_q) {
      // transform q: write embeds out, but no cache write
      unsigned d = 2 * threadIdx.x;
      for (; d < embed_dim; d += 2 * THREADS_PER_BLOCK) {
        float2 cos_sin = *(const float2*)addr(position_embeds, d);
        float cos = cos_sin.x;
        float sin = cos_sin.y;

        float2 embed = __bfloat1622float2(*(const __hip_bfloat162*)addr(embeds_in, d));
        float real_embed = embed.x;
        float imag_embed = embed.y;

        float real_rot_embed = fma(real_embed, cos, -imag_embed * sin);
        float imag_rot_embed = fma(real_embed, sin, imag_embed * cos);

        __hip_bfloat162 rot_embed = __float22bfloat162_rn(make_float2(real_rot_embed, imag_rot_embed));

        *((__hip_bfloat162*)addr(embeds_out, d)) = rot_embed;
      }

    } else {
      // transform k: write embeds out, and write to cache

      // realign cache_out
      unsigned cache_tok_pos = raw_tok_idx - START_T;
      __hip_bfloat16* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

      bool valid_cache_write = (raw_tok_idx >= START_T) && (raw_tok_idx < START_T + MAX_T);
      unsigned d = 2 * threadIdx.x;
      for (; d < embed_dim; d += 2 * THREADS_PER_BLOCK) {
        float2 cos_sin = *(const float2*)addr(position_embeds, d);
        float cos = cos_sin.x;
        float sin = cos_sin.y;

        float2 embed = __bfloat1622float2(*(const __hip_bfloat162*)addr(embeds_in, d));
        float real_embed = embed.x;
        float imag_embed = embed.y;

        float real_rot_embed = fma(real_embed, cos, -imag_embed * sin);
        float imag_rot_embed = fma(real_embed, sin, imag_embed * cos);

        __hip_bfloat162 rot_embed = __float22bfloat162_rn(make_float2(real_rot_embed, imag_rot_embed));

        *((__hip_bfloat162*)addr(embeds_out, d)) = rot_embed;

        // write the new token in cache
        if (valid_cache_write) {
          *((__hip_bfloat162*)addr(cache_out_write, d)) = rot_embed;
        }
      }
    }
    
  } else {
    // For v, just copy without RoPE
    
    unsigned cache_tok_pos = raw_tok_idx;

    if (cache_tok_pos >= MAX_T) {
      // out of range, nothing to do
      return;
    }

    // realign cache_out
    __hip_bfloat16* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      __hip_bfloat16 embed = *addr(embeds_in, d);
      *addr(cache_out_write, d) = embed;
    }
  }
}

void sliding_kvcache_complex_rope_forward_prefill_bf16(
  hipStream_t stream,
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  // k strides
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  // v strides
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride,
  const float* position_embeddings, // shape [B, T, embed_dim // 2, 2]
  const __hip_bfloat16* q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* q_out, // shape [B, num_q_heads, T, embed_dim]
  __hip_bfloat16* k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  __hip_bfloat16* k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  __hip_bfloat16* v_cache_out // [B, num_v_heads, MAX_T, embed_dim]
) {
  // Prefill (we incremented seen_tokens before updating the cache)
  const unsigned COPIED_T = std::min(T, MAX_T);
  const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);
  // We return all tokens in that case to avoid catastrophic forgetting
  // but store in the cache the latest ones
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // always launch with T; if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
  // expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, T, B);

  sliding_kvcache_complex_rope_forward_prefill_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    position_embeddings,
    q_in,
    k_in,
    v_in,
    q_out,
    k_out,
    // KV cache
    k_cache_out,
    v_cache_out,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
    START_T,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    embed_dim,
    // q strides
    q_in_batch_stride,
    q_in_head_stride,
    q_in_tok_stride,
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


// always launch with max(T, MAX_T); if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(T, MAX_T), z=B]
// applies RoPE to q,k and copies rotated k,v to sliding cache
void __global__ sliding_kvcache_complex_rope_forward_update_overwrite_fp16_kernel(
  const float* __restrict__ position_embeds, // shape [B, T, embed_dim // 2, 2]
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  half* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache in
  const half* __restrict__ k_cache_in, // [B, num_k_heads, MAX_T, embed_dim]
  const half* __restrict__ v_cache_in, // [B, num_v_heads, MAX_T, embed_dim]
  // KV cache out
  half* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens in k_in (might be bigger than MAX_T)
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned START_T, // starting copied token index
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
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
  bool copy_new_tokens = raw_tok_idx >= num_old_tokens; // for k and v

  unsigned head_idx = blockIdx.x;
  unsigned batch_idx = blockIdx.z;

  // determine if we are supposed to transform an embedding from q, k, or v
  unsigned tok_idx;
  const half* __restrict__ embeds_in;
  half* __restrict__ embeds_out;
  const half* __restrict__ prev_cache;
  half* __restrict__ cache_out;
  unsigned num_heads;

  // strides
  unsigned batch_stride;
  unsigned head_stride;
  unsigned tok_stride;

  bool transform_q = false;
  bool transform_k = false;
  bool transform_v = false;
  // determine if we are processing q, k or v
  if (head_idx < num_q_heads) {
    // q
    transform_q = true;
    tok_idx = raw_tok_idx; // always copy all tokens for q
    embeds_in = q_in;
    embeds_out = q_out;
    cache_out = nullptr; // no q cache
    num_heads = num_q_heads;
    
    batch_stride = q_in_batch_stride;
    head_stride = q_in_head_stride;
    tok_stride = q_in_tok_stride;
  } else if (head_idx < num_q_heads + num_k_heads) {
    // k
    transform_k = true;
    num_heads = num_k_heads;
    cache_out = k_cache_out;
    head_idx -= num_q_heads;
    
    if (copy_new_tokens) {
      // copy new tokens
      tok_idx = (raw_tok_idx - num_old_tokens + START_T);
      embeds_in = k_in;
      embeds_out = nullptr;
      
      batch_stride = k_in_batch_stride;
      head_stride = k_in_head_stride;
      tok_stride = k_in_tok_stride;
    } else {
      // copy old cache
      tok_idx = (raw_tok_idx + num_new_tokens);
      embeds_in = k_cache_in;
      embeds_out = nullptr;
      
      batch_stride = num_heads * MAX_T * embed_dim;
      head_stride = MAX_T * embed_dim;
      tok_stride = embed_dim;
    }
  } else {
    // v
    transform_v = true;
    num_heads = num_v_heads;
    cache_out = v_cache_out;
    head_idx -= (num_q_heads + num_k_heads);
    
    if (copy_new_tokens) {
      // copy new tokens
      tok_idx = (raw_tok_idx - num_old_tokens + START_T);
      embeds_in = v_in;
      embeds_out = nullptr;
      
      batch_stride = v_in_batch_stride;
      head_stride = v_in_head_stride;
      tok_stride = v_in_tok_stride;
    } else {
      // copy old cache
      tok_idx = (raw_tok_idx + num_new_tokens);
      embeds_in = v_cache_in;
      embeds_out = nullptr;
      
      batch_stride = num_heads * MAX_T * embed_dim;
      head_stride = MAX_T * embed_dim;
      tok_stride = embed_dim;
    }
  }

  // realign embeds_in
  unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
  embeds_in = &embeds_in[embed_in_idx];

  // realign cache_out
  half* __restrict__ cache_out_write = nullptr;
  if (cache_out != nullptr) {
    unsigned cache_tok_pos = raw_tok_idx;
    cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
  }

  // Apply RoPE to q and k, or just copy v
  if (transform_q || (transform_k && copy_new_tokens)) {
    // Apply RoPE to q (all tokens) or k (new tokens only)
    unsigned pos_idx = batch_idx * T + tok_idx;

    // realign the position_embeddings to the position
    position_embeds = &position_embeds[pos_idx * embed_dim];

    if (transform_q) {
      // transform q: write embeds out, but no cache write
      if (tok_idx >= T) {
        // out of range, nothing to do
        return;
      }

      // realign embeds_out for q
      embeds_out = &embeds_out[((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0];
      
      unsigned d = 2 * threadIdx.x;
      for (; d < embed_dim; d += 2 * THREADS_PER_BLOCK) {
        float2 cos_sin = *(const float2*)addr(position_embeds, d);
        float cos = cos_sin.x;
        float sin = cos_sin.y;

        float2 embed = __half22float2(*(const half2*)addr(embeds_in, d));
        float real_embed = embed.x;
        float imag_embed = embed.y;

        float real_rot_embed = fma(real_embed, cos, -imag_embed * sin);
        float imag_rot_embed = fma(real_embed, sin, imag_embed * cos);

        half2 rot_embed = __float22half2_rn(make_float2(real_rot_embed, imag_rot_embed));

        *((half2*)addr(embeds_out, d)) = rot_embed;
      }
    } else {
      // transform k: no write embeds out, and write to cache
      unsigned cache_tok_pos = raw_tok_idx;
      if (cache_tok_pos >= MAX_T) {
        // out of range, nothing to do
        return;
      }

      // realign cache_out
      half* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

      unsigned d = 2 * threadIdx.x;
      for (; d < embed_dim; d += 2 * THREADS_PER_BLOCK) {
        float2 cos_sin = *(const float2*)addr(position_embeds, d);
        float cos = cos_sin.x;
        float sin = cos_sin.y;

        float2 embed = __half22float2(*(const half2*)addr(embeds_in, d));
        float real_embed = embed.x;
        float imag_embed = embed.y;

        float real_rot_embed = fma(real_embed, cos, -imag_embed * sin);
        float imag_rot_embed = fma(real_embed, sin, imag_embed * cos);

        half2 rot_embed = __float22half2_rn(make_float2(real_rot_embed, imag_rot_embed));

        // write the new token in cache
        *((half2*)addr(cache_out_write, d)) = rot_embed;
      }
    }
  } else {
    // For v or copying old k cache, just copy without RoPE
    unsigned cache_tok_pos = raw_tok_idx;
    if (cache_tok_pos >= MAX_T) {
      // out of range, nothing to do
      return;
    }

    // realign cache_out
    half* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
    
    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      half embed = *addr(embeds_in, d);
      *addr(cache_out_write, d) = embed;
    }
  }
}

void sliding_kvcache_complex_rope_forward_update_overwrite_fp16(
  hipStream_t stream,
  unsigned B, // batch size
  unsigned T, // num new tokens in k_in (might be bigger than MAX_T)
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned num_q_heads, // number of heads for k
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  // k strides
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  // v strides
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride,
  const float* position_embeddings, // shape [B, T, embed_dim // 2, 2]
  const half* q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* v_in, // shape [B, num_v_heads, T, embed_dim]
  half* q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache in
  const half* k_cache_in, // [B, num_k_heads, MAX_T, embed_dim]
  const half* v_cache_in, // [B, num_v_heads, MAX_T, embed_dim]
  // KV cache out
  half* k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* v_cache_out // [B, num_v_heads, MAX_T, embed_dim]
) {
  // determine what part of the new tokens to copy
  // if we have a lot of input tokens, we actually copy only the MAX_T last ones
  const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // always launch with max(T, MAX_T); if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
  // expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(T, MAX_T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, max(T, MAX_T), B);

  sliding_kvcache_complex_rope_forward_update_overwrite_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    position_embeddings,
    q_in,
    k_in,
    v_in,
    q_out,
    // KV cache in
    k_cache_in,
    v_cache_in,
    // KV cache out
    k_cache_out,
    v_cache_out,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
    START_T,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    embed_dim,
    // q strides
    q_in_batch_stride,
    q_in_head_stride,
    q_in_tok_stride,
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


// always launch with max(T, MAX_T); if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(T, MAX_T), z=B]
// applies RoPE to q,k and copies rotated k,v to sliding cache
void __global__ sliding_kvcache_complex_rope_forward_update_overwrite_bf16_kernel(
  const float* __restrict__ position_embeds, // shape [S, embed_dim] or [B, T, embed_dim // 2, 2]
  const __hip_bfloat16* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache in
  const __hip_bfloat16* __restrict__ k_cache_in, // [B, num_k_heads, MAX_T, embed_dim]
  const __hip_bfloat16* __restrict__ v_cache_in, // [B, num_v_heads, MAX_T, embed_dim]
  // KV cache out
  __hip_bfloat16* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  __hip_bfloat16* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens in k_in (might be bigger than MAX_T)
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned START_T, // starting copied token index
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
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
  bool copy_new_tokens = raw_tok_idx >= num_old_tokens; // for k and v

  unsigned head_idx = blockIdx.x;
  unsigned batch_idx = blockIdx.z;

  // determine if we are supposed to transform an embedding from q, k, or v
  unsigned tok_idx;
  const __hip_bfloat16* __restrict__ embeds_in;
  __hip_bfloat16* __restrict__ embeds_out;
  const __hip_bfloat16* __restrict__ prev_cache;
  __hip_bfloat16* __restrict__ cache_out;
  unsigned num_heads;

  // strides
  unsigned batch_stride;
  unsigned head_stride;
  unsigned tok_stride;

  bool transform_q = false;
  bool transform_k = false;
  bool transform_v = false;
  // determine if we are processing q, k or v
  if (head_idx < num_q_heads) {
    // q
    transform_q = true;
    tok_idx = raw_tok_idx; // always copy all tokens for q
    embeds_in = q_in;
    embeds_out = q_out;
    cache_out = nullptr; // no q cache
    num_heads = num_q_heads;
    
    batch_stride = q_in_batch_stride;
    head_stride = q_in_head_stride;
    tok_stride = q_in_tok_stride;
  } else if (head_idx < num_q_heads + num_k_heads) {
    // k
    transform_k = true;
    num_heads = num_k_heads;
    cache_out = k_cache_out;
    head_idx -= num_q_heads;
    
    if (copy_new_tokens) {
      // copy new tokens
      tok_idx = (raw_tok_idx - num_old_tokens + START_T);
      embeds_in = k_in;
      embeds_out = nullptr;
      
      batch_stride = k_in_batch_stride;
      head_stride = k_in_head_stride;
      tok_stride = k_in_tok_stride;
    } else {
      // copy old cache
      tok_idx = (raw_tok_idx + num_new_tokens);
      embeds_in = k_cache_in;
      embeds_out = nullptr;
      
      batch_stride = num_heads * MAX_T * embed_dim;
      head_stride = MAX_T * embed_dim;
      tok_stride = embed_dim;
    }
  } else {
    // v
    transform_v = true;
    num_heads = num_v_heads;
    cache_out = v_cache_out;
    head_idx -= (num_q_heads + num_k_heads);
    
    if (copy_new_tokens) {
      // copy new tokens
      tok_idx = (raw_tok_idx - num_old_tokens + START_T);
      embeds_in = v_in;
      embeds_out = nullptr;
      
      batch_stride = v_in_batch_stride;
      head_stride = v_in_head_stride;
      tok_stride = v_in_tok_stride;
    } else {
      // copy old cache
      tok_idx = (raw_tok_idx + num_new_tokens);
      embeds_in = v_cache_in;
      embeds_out = nullptr;
      
      batch_stride = num_heads * MAX_T * embed_dim;
      head_stride = MAX_T * embed_dim;
      tok_stride = embed_dim;
    }
  }

  // realign embeds_in
  unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
  embeds_in = &embeds_in[embed_in_idx];

  // realign cache_out
  __hip_bfloat16* __restrict__ cache_out_write = nullptr;
  if (cache_out != nullptr) {
    unsigned cache_tok_pos = raw_tok_idx;
    cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
  }

  // Apply RoPE to q and k, or just copy v
  if (transform_q || (transform_k && copy_new_tokens)) {
    // Apply RoPE to q (all tokens) or k (new tokens only)
    unsigned pos_idx = batch_idx * T + tok_idx;

    // realign the position embeddings to the position
    position_embeds = &position_embeds[pos_idx * embed_dim];

    if (transform_q) {
      // transform q: write embeds out, but no cache write
      if (tok_idx >= T) {
        // out of range, nothing to do
        return;
      }

      // realign embeds_out for q
      embeds_out = &embeds_out[((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0];
      
      unsigned d = 2 * threadIdx.x;
      for (; d < embed_dim; d += 2 * THREADS_PER_BLOCK) {
        float2 cos_sin = *(const float2*)addr(position_embeds, d);
        float cos = cos_sin.x;
        float sin = cos_sin.y;

        float2 embed = __bfloat1622float2(*(const __hip_bfloat162*)addr(embeds_in, d));
        float real_embed = embed.x;
        float imag_embed = embed.y;

        float real_rot_embed = fma(real_embed, cos, -imag_embed * sin);
        float imag_rot_embed = fma(real_embed, sin, imag_embed * cos);

        __hip_bfloat162 rot_embed = __float22bfloat162_rn(make_float2(real_rot_embed, imag_rot_embed));

        *((__hip_bfloat162*)addr(embeds_out, d)) = rot_embed;
      }
    } else {
      // transform k: no write embeds out, and write to cache
      unsigned cache_tok_pos = raw_tok_idx;
      if (cache_tok_pos >= MAX_T) {
        // out of range, nothing to do
        return;
      }

      // realign cache_out
      __hip_bfloat16* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];

      unsigned d = 2 * threadIdx.x;
      for (; d < embed_dim; d += 2 * THREADS_PER_BLOCK) {
        float2 cos_sin = *(const float2*)addr(position_embeds, d);
        float cos = cos_sin.x;
        float sin = cos_sin.y;

        float2 embed = __bfloat1622float2(*(const __hip_bfloat162*)addr(embeds_in, d));
        float real_embed = embed.x;
        float imag_embed = embed.y;

        float real_rot_embed = fma(real_embed, cos, -imag_embed * sin);
        float imag_rot_embed = fma(real_embed, sin, imag_embed * cos);

        __hip_bfloat162 rot_embed = __float22bfloat162_rn(make_float2(real_rot_embed, imag_rot_embed));

        *((__hip_bfloat162*)addr(cache_out_write, d)) = rot_embed;
      }
    }
  } else {
    // For v or copying old k cache, just copy without RoPE
    unsigned cache_tok_pos = raw_tok_idx;
    if (cache_tok_pos >= MAX_T) {
      // out of range, nothing to do
      return;
    }

    // realign cache_out
    __hip_bfloat16* __restrict__ cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
    
    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      __hip_bfloat16 embed = *addr(embeds_in, d);

      *addr(cache_out_write, d) = embed;
    }
  }
}

void sliding_kvcache_complex_rope_forward_update_overwrite_bf16(
  hipStream_t stream,
  unsigned B, // batch size
  unsigned T, // num new tokens in k_in (might be bigger than MAX_T)
  unsigned MAX_T, // number of tokens in the KV cache
  unsigned num_q_heads, // number of heads for k
  unsigned num_k_heads, // number of heads for k
  unsigned num_v_heads, // number of heads for v
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  // k strides
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  // v strides
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride,
  const float* position_embeddings, // shape [B, T, embed_dim // 2, 2]
  const __hip_bfloat16* q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache in
  const __hip_bfloat16* k_cache_in, // [B, num_k_heads, MAX_T, embed_dim]
  const __hip_bfloat16* v_cache_in, // [B, num_v_heads, MAX_T, embed_dim]
  // KV cache out
  __hip_bfloat16* k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  __hip_bfloat16* v_cache_out // [B, num_v_heads, MAX_T, embed_dim]
) {
  // determine what part of the new tokens to copy
  // if we have a lot of input tokens, we actually copy only the MAX_T last ones
  const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
  // always launch with max(T, MAX_T); if T>MAX_T, transform all T tokens for q and k, but only store in cache the last MAX_T
  // expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(T, MAX_T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, max(T, MAX_T), B);

  sliding_kvcache_complex_rope_forward_update_overwrite_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    position_embeddings,
    q_in,
    k_in,
    v_in,
    q_out,
    // KV cache in
    k_cache_in,
    v_cache_in,
    // KV cache out
    k_cache_out,
    v_cache_out,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
    START_T,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    embed_dim,
    // q strides
    q_in_batch_stride,
    q_in_head_stride,
    q_in_tok_stride,
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