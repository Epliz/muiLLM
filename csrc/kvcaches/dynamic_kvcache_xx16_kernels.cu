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

// expected block dimensions: [x=num_k_heads+num_v_heads, y=PREV_T+T, z=B]
void __global__ dynamic_kvcache_update_xx16_kernel(
  const xx16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const xx16* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  const xx16* __restrict__ prev_k_in, // shape [B, num_k_heads, PREV_T, embed_dim]
  const xx16* __restrict__ prev_v_in, // shape [B, num_v_heads, PREV_T, embed_dim]
  // KV cache
  xx16* __restrict__ k_cache_out, // [B, num_k_heads, PREV_T+T, embed_dim]
  xx16* __restrict__ v_cache_out, // [B, num_v_heads, PREV_T+T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned PREV_T, // num previous tokens to copy
  unsigned T, // num new tokens
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
  unsigned v_in_tok_stride,
  // prev k strides
  unsigned prev_k_in_batch_stride,
  unsigned prev_k_in_head_stride,
  unsigned prev_k_in_tok_stride,
  // prev v strides
  unsigned prev_v_in_batch_stride,
  unsigned prev_v_in_head_stride,
  unsigned prev_v_in_tok_stride
) {
    unsigned NEW_T = PREV_T + T;
  
    // one block does one head of a new token
    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = blockIdx.y; // should be launched with PREV_T+T
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
        cache_out = k_cache_out;
        num_heads = num_k_heads;

        // either copy old tokens or new tokens
        if (tok_idx < PREV_T) {
          X = prev_k_in;

          // strides are the ones of the cache
          batch_stride = prev_k_in_batch_stride;
          head_stride = prev_k_in_head_stride;
          tok_stride = prev_k_in_tok_stride;
        } else {
          X = k_in;

          // strides are the ones of K
          batch_stride = k_in_batch_stride;
          head_stride = k_in_head_stride;
          tok_stride = k_in_tok_stride;
        }
    } else {
        // v
        cache_out = v_cache_out;
        num_heads = num_v_heads;
        head_idx -= num_k_heads;
  
        // either copy old tokens or new tokens
        if (tok_idx < PREV_T) {
          X = prev_v_in;

          // strides are the ones of the cache if less than prev_t
          batch_stride = prev_v_in_batch_stride;
          head_stride = prev_v_in_head_stride;
          tok_stride = prev_v_in_tok_stride;
        } else {
          X = v_in;
          // strides are the ones of V
          batch_stride = v_in_batch_stride;
          head_stride = v_in_head_stride;
          tok_stride = v_in_tok_stride;
        }
    }

    // realign the pointer to where we are supposed to write out if needed

    // realign cache_out
    cache_out = &cache_out[(((batch_idx * num_heads) + head_idx) * NEW_T + tok_idx) * embed_dim + 0];

    if (tok_idx < PREV_T) {
      // copy old tokens
      unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
      X = &X[embed_in_idx];
    } else {
      // copy new tokens
      tok_idx -= PREV_T;
      unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
      X = &X[embed_in_idx];
    }

    // realign embeds_in and embeds_out
    // k/v might be strided, but embedding dimension stride needs to be 1
    unsigned d = threadIdx.x;
    for (; d < embed_dim; d += THREADS_PER_BLOCK) {
      xx16 embed = *addr(X, d);
      *addr(cache_out, d) = embed;
    }
}

void dynamic_kvcache_update_xx16(
  hipStream_t stream,
  const xx16* k_in,
  const xx16* v_in,
  const xx16* prev_k_cache_in,
  const xx16* prev_v_cache_in,
  xx16* k_cache_out,
  xx16* v_cache_out,
  unsigned B,
  unsigned PREV_T,
  unsigned T,
  unsigned num_k_heads,
  unsigned num_v_heads,
  unsigned embed_dim,
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride,
  unsigned prev_k_in_batch_stride,
  unsigned prev_k_in_head_stride,
  unsigned prev_k_in_tok_stride,
  unsigned prev_v_in_batch_stride,
  unsigned prev_v_in_head_stride,
  unsigned prev_v_in_tok_stride
) {
  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);


  // TODO: max cuda block dimension is 1024, so need to do something when T>1024
  // expected block dimensions: [x=num_k_heads+num_v_heads, y=PREV_T+T, z=B]
  const dim3 num_blocks = dim3(num_k_heads + num_v_heads, PREV_T + T, B);

  dynamic_kvcache_update_xx16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    k_in,
    v_in,
    prev_k_cache_in,
    prev_v_cache_in,
    k_cache_out,
    v_cache_out,
    B,
    PREV_T,
    T,
    num_k_heads,
    num_v_heads,
    embed_dim,
    k_in_batch_stride,
    k_in_head_stride,
    k_in_tok_stride,
    v_in_batch_stride,
    v_in_head_stride,
    v_in_tok_stride,
    prev_k_in_batch_stride,
    prev_k_in_head_stride,
    prev_k_in_tok_stride,
    prev_v_in_batch_stride,
    prev_v_in_head_stride,
    prev_v_in_tok_stride
  );
}


// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(PREV_T, T), z=B]
void __global__ apply_rope_forward_fp16_kernel_write_dynamic_cache(
  const half* __restrict__ cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* __restrict__ sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  half* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  half* __restrict__ k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  const half* __restrict__ prev_k_cache, // [B, num_k_heads, PREV_T, embed_dim]
  const half* __restrict__ prev_v_cache, // [B, num_v_heads, PREV_T, embed_dim]
  half* __restrict__ k_cache_out, // [B, num_k_heads, PREV_T + T, embed_dim]
  half* __restrict__ v_cache_out, // [B, num_v_heads, PREV_T + T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned PREV_T, // number of tokens previously in the KV cache
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
    // one block does one head of a new token
    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = blockIdx.y; // should be launched with max(PREV_T, T)
    unsigned batch_idx = blockIdx.z;
    unsigned pos_idx = batch_idx * T + tok_idx;

    // determine if we are supposed to transform an embedding from q or k,
    // and which head
    const half* __restrict__ embeds_in;
    half* __restrict__ embeds_out;
    const half* __restrict__ prev_cache;
    half* __restrict__ cache_out;
    unsigned num_heads;

    // strides
    unsigned batch_stride;
    unsigned head_stride;
    unsigned tok_stride;

    // determine if we are processing q, k or v
    if (head_idx < num_q_heads) {
        embeds_in = q_in;
        embeds_out = q_out;
        // no q cache
        prev_cache = nullptr;
        cache_out = nullptr;
        num_heads = num_q_heads;
  
        batch_stride = q_in_batch_stride;
        head_stride = q_in_head_stride;
        tok_stride = q_in_tok_stride;
    } else if (head_idx < num_q_heads + num_k_heads) {
        // k
        embeds_in = k_in;
        embeds_out = k_out;
        prev_cache = prev_k_cache;
        cache_out = k_cache_out;
        num_heads = num_k_heads;

        batch_stride = k_in_batch_stride;
        head_stride = k_in_head_stride;
        tok_stride = k_in_tok_stride;

        head_idx -= num_q_heads;
    } else {
        // v
        embeds_in = v_in;
        embeds_out = nullptr;
        prev_cache = prev_v_cache;
        cache_out = v_cache_out;
        num_heads = num_v_heads;

        batch_stride = v_in_batch_stride;
        head_stride = v_in_head_stride;
        tok_stride = v_in_tok_stride;

        head_idx -= (num_q_heads + num_k_heads);
    }

    // ROTARY_CACHE_BTE_LAYOUT
    uint64_t position_id = tok_idx < T ? pos_idx : 0;

    // realign the pointer to where we are supposed to write out if needed

    // if dynamic cache, copy previous content here
    if (prev_cache != nullptr){
        unsigned NEW_T = PREV_T + T;

        // realign the cache pointers according to where we are supposed to read/write the token
        // we are in charge of
        prev_cache = &prev_cache[(((batch_idx * num_heads) + head_idx) * PREV_T + tok_idx) * embed_dim + 0];
        half* __restrict__ cache_out_copy = &cache_out[(((batch_idx * num_heads) + head_idx) * NEW_T + tok_idx) * embed_dim + 0];

        // vectorized part: NOT WORKING
        unsigned d = threadIdx.x;//2 * threadIdx.x;
        /*
        for (; d + 1 < embed_dim; d += (2 * THREADS_PER_BLOCK)) {
            half2 c = *addr((const half2*) prev_cache, d);
            *addr((half2*) cache_out_copy, d) = c;
        }
        */
        // loop remainder
        for (; d < embed_dim; d += THREADS_PER_BLOCK) {
            *addr(cache_out_copy, d) = *addr(prev_cache, d);
        }
    }

    // realign the cos/sin caches to the position
    cos_cached = &cos_cached[position_id * embed_dim];
    sin_cached = &sin_cached[position_id * embed_dim];

    if (tok_idx >= T) {
        // no tokens to apply the rotary embeddings to, we were just there to do the copy
        return;
    }


    // realign embeds_in and embeds_out
    // q/k/v might be strided, but embedding dimension stride needs to be 1
    unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
    embeds_in = &embeds_in[embed_in_idx];

    if (embeds_out != nullptr) {
      unsigned embed_out_idx = ((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0;
      embeds_out = &embeds_out[embed_out_idx];
    }

    // realign cache_out
    half* __restrict__ cache_out_write = nullptr;
    if (cache_out != nullptr) {
      unsigned NEW_T = PREV_T + T;
      cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * NEW_T + (PREV_T + tok_idx)) * embed_dim + 0];
    }

    if (embeds_out != nullptr) {
      // q or k
      unsigned half_embed_dim = embed_dim / 2;

      // TODO: vectorize
      // first half
      const half* __restrict__ rot_embeds_in;
      rot_embeds_in = &embeds_in[half_embed_dim];

      unsigned d = threadIdx.x;
      for (; d < half_embed_dim; d += THREADS_PER_BLOCK) {
        half cos_pos = *addr(cos_cached, d);
        half sin_pos = *addr(sin_cached, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = __hneg(*addr(rot_embeds_in, d));
    
        half r = __hfma(rot_embed, sin_pos,__hmul(embed, cos_pos));

        *addr(embeds_out, d) = r;

        // TODO: variants with and without cache write?
        // write the new token in cache
        if (cache_out_write != nullptr) {
          *addr(cache_out_write, d) = r;
        }
      }

      // second half
      rot_embeds_in = &embeds_in[(int)-half_embed_dim];
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        half cos_pos = *addr(cos_cached, d);
        half sin_pos = *addr(sin_cached, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = *addr(rot_embeds_in, d);

        half r = __hfma(rot_embed, sin_pos,__hmul(embed, cos_pos));

        *addr(embeds_out, d) = r;

        // TODO: variants with and without cache write?
        // write the new token in cache
        if (cache_out_write != nullptr) {
          *addr(cache_out_write, d) = r;
        }
      }
    } else {
      // v
      unsigned d = threadIdx.x;
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        half embed = *addr(embeds_in, d);
        *addr(cache_out_write, d) = embed;
      }
    }
}

void muillm_apply_rope_forward_fp16_dynamic_cache(
  hipStream_t stream,
  unsigned B,
  unsigned T,
  unsigned PREV_T,
  unsigned num_q_heads,
  unsigned num_k_heads,
  unsigned num_v_heads,
  unsigned embed_dim,
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
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
  const half* prev_k_cache, // [B, num_k_heads, PREV_T, embed_dim]
  const half* prev_v_cache, // [B, num_v_heads, PREV_T, embed_dim]
  half* k_cache_out, // [B, num_k_heads, PREV_T + T, embed_dim]
  half* v_cache_out // [B, num_v_heads, PREV_T + T, embed_dim]
) {

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);


  // expected block dimensions: [x=num_q_heads+num_k_heads, y=max(PREV_T, T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, std::max(PREV_T, T), B);

  apply_rope_forward_fp16_kernel_write_dynamic_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const half*)cos_cached,
    (const half*)sin_cached,
    (const half*)q_in,
    (const half*)k_in,
    (const half*)v_in,
    (half*)q_out,
    (half*)k_out,
    // KV cache
    (const half*)prev_k_cache,
    (const half*)prev_v_cache,
    (half*)k_cache_out,
    (half*)v_cache_out,
    // tensor dimension sizes
    B,
    T,
    PREV_T,
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

// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(PREV_T, T), z=B]
void __global__ apply_rope_forward_bf16_kernel_write_dynamic_cache(
  const __hip_bfloat16* __restrict__ cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* __restrict__ sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  const __hip_bfloat16* __restrict__ prev_k_cache, // [B, num_k_heads, PREV_T, embed_dim]
  const __hip_bfloat16* __restrict__ prev_v_cache, // [B, num_v_heads, PREV_T, embed_dim]
  __hip_bfloat16* __restrict__ k_cache_out, // [B, num_k_heads, PREV_T + T, embed_dim]
  __hip_bfloat16* __restrict__ v_cache_out, // [B, num_v_heads, PREV_T + T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned PREV_T, // number of tokens previously in the KV cache
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
    // one block does one head of a new token
    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = blockIdx.y; // should be launched with max(PREV_T, T)
    unsigned batch_idx = blockIdx.z;
    unsigned pos_idx = batch_idx * T + tok_idx;

    // determine if we are supposed to transform an embedding from q or k,
    // and which head
    const __hip_bfloat16* __restrict__ embeds_in;
    __hip_bfloat16* __restrict__ embeds_out;
    const __hip_bfloat16* __restrict__ prev_cache;
    __hip_bfloat16* __restrict__ cache_out;
    unsigned num_heads;

    // strides
    unsigned batch_stride;
    unsigned head_stride;
    unsigned tok_stride;

    // determine if we are processing q, k or v
    if (head_idx < num_q_heads) {
        embeds_in = q_in;
        embeds_out = q_out;
        // no q cache
        prev_cache = nullptr;
        cache_out = nullptr;
        num_heads = num_q_heads;
  
        batch_stride = q_in_batch_stride;
        head_stride = q_in_head_stride;
        tok_stride = q_in_tok_stride;
    } else if (head_idx < num_q_heads + num_k_heads) {
        // k
        embeds_in = k_in;
        embeds_out = k_out;
        prev_cache = prev_k_cache;
        cache_out = k_cache_out;
        num_heads = num_k_heads;

        batch_stride = k_in_batch_stride;
        head_stride = k_in_head_stride;
        tok_stride = k_in_tok_stride;

        head_idx -= num_q_heads;
    } else {
        // v
        embeds_in = v_in;
        embeds_out = nullptr;
        prev_cache = prev_v_cache;
        cache_out = v_cache_out;
        num_heads = num_v_heads;

        batch_stride = v_in_batch_stride;
        head_stride = v_in_head_stride;
        tok_stride = v_in_tok_stride;

        head_idx -= (num_q_heads + num_k_heads);
    }

    // ROTARY_CACHE_BTE_LAYOUT
    uint64_t position_id = tok_idx < T ? pos_idx : 0;

    // realign the pointer to where we are supposed to write out if needed

    // if dynamic cache, copy previous content here
    if (prev_cache != nullptr){
        unsigned NEW_T = PREV_T + T;

        // realign the cache pointers according to where we are supposed to read/write the token
        // we are in charge of
        prev_cache = &prev_cache[(((batch_idx * num_heads) + head_idx) * PREV_T + tok_idx) * embed_dim + 0];
        __hip_bfloat16* __restrict__ cache_out_copy = &cache_out[(((batch_idx * num_heads) + head_idx) * NEW_T + tok_idx) * embed_dim + 0];

        // vectorized part: NOT WORKING
        unsigned d = threadIdx.x;//2 * threadIdx.x;
        /*
        for (; d + 1 < embed_dim; d += (2 * THREADS_PER_BLOCK)) {
            __hip_bfloat162 c = *addr((const __hip_bfloat162*) prev_cache, d);
            *addr((__hip_bfloat162*) cache_out_copy, d) = c;
        }
        */
        // loop remainder
        for (; d < embed_dim; d += THREADS_PER_BLOCK) {
            *addr(cache_out_copy, d) = *addr(prev_cache, d);
        }
    }

    // realign the cos/sin caches to the position
    cos_cached = &cos_cached[position_id * embed_dim];
    sin_cached = &sin_cached[position_id * embed_dim];

    if (tok_idx >= T) {
        // no tokens to apply the rotary embeddings to, we were just there to do the copy
        return;
    }


    // realign embeds_in and embeds_out
    // q/k/v might be strided, but embedding dimension stride needs to be 1
    unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
    embeds_in = &embeds_in[embed_in_idx];

    if (embeds_out != nullptr) {
      unsigned embed_out_idx = ((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0;
      embeds_out = &embeds_out[embed_out_idx];
    }

    // realign cache_out
    __hip_bfloat16* __restrict__ cache_out_write = nullptr;
    if (cache_out != nullptr) {
      unsigned NEW_T = PREV_T + T;
      cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * NEW_T + (PREV_T + tok_idx)) * embed_dim + 0];
    }

    if (embeds_out != nullptr) {
      // q or k
      unsigned half_embed_dim = embed_dim / 2;

      // TODO: vectorize
      // first half
      const __hip_bfloat16* __restrict__ rot_embeds_in;
      rot_embeds_in = &embeds_in[half_embed_dim];

      unsigned d = threadIdx.x;
      for (; d < half_embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 cos_pos = *addr(cos_cached, d);
        __hip_bfloat16 sin_pos = *addr(sin_cached, d);

        __hip_bfloat16 embed = *addr(embeds_in, d);
        __hip_bfloat16 rot_embed = __hneg(*addr(rot_embeds_in, d));
    
        __hip_bfloat16 r = __hfma(rot_embed, sin_pos,__hmul(embed, cos_pos));

        *addr(embeds_out, d) = r;

        // TODO: variants with and without cache write?
        // write the new token in cache
        if (cache_out_write != nullptr) {
          *addr(cache_out_write, d) = r;
        }
      }

      // second half
      rot_embeds_in = &embeds_in[(int)-half_embed_dim];
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 cos_pos = *addr(cos_cached, d);
        __hip_bfloat16 sin_pos = *addr(sin_cached, d);

        __hip_bfloat16 embed = *addr(embeds_in, d);
        __hip_bfloat16 rot_embed = *addr(rot_embeds_in, d);

        __hip_bfloat16 r = __hfma(rot_embed, sin_pos,__hmul(embed, cos_pos));

        *addr(embeds_out, d) = r;

        // TODO: variants with and without cache write?
        // write the new token in cache
        if (cache_out_write != nullptr) {
          *addr(cache_out_write, d) = r;
        }
      }
    } else {
      // v
      unsigned d = threadIdx.x;
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 embed = *addr(embeds_in, d);
        *addr(cache_out_write, d) = embed;
      }
    }
}

void muillm_apply_rope_forward_bf16_dynamic_cache(
  hipStream_t stream,
  unsigned B,
  unsigned T,
  unsigned PREV_T,
  unsigned num_q_heads,
  unsigned num_k_heads,
  unsigned num_v_heads,
  unsigned embed_dim,
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
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
  const __hip_bfloat16* prev_k_cache, // [B, num_k_heads, PREV_T, embed_dim]
  const __hip_bfloat16* prev_v_cache, // [B, num_v_heads, PREV_T, embed_dim]
  __hip_bfloat16* k_cache_out, // [B, num_k_heads, PREV_T + T, embed_dim]
  __hip_bfloat16* v_cache_out // [B, num_v_heads, PREV_T + T, embed_dim]
) {

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);


  // expected block dimensions: [x=num_q_heads+num_k_heads, y=max(PREV_T, T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, std::max(PREV_T, T), B);

  apply_rope_forward_bf16_kernel_write_dynamic_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const __hip_bfloat16*)cos_cached,
    (const __hip_bfloat16*)sin_cached,
    (const __hip_bfloat16*)q_in,
    (const __hip_bfloat16*)k_in,
    (const __hip_bfloat16*)v_in,
    (__hip_bfloat16*)q_out,
    (__hip_bfloat16*)k_out,
    // KV cache
    (const __hip_bfloat16*)prev_k_cache,
    (const __hip_bfloat16*)prev_v_cache,
    (__hip_bfloat16*)k_cache_out,
    (__hip_bfloat16*)v_cache_out,
    // tensor dimension sizes
    B,
    T,
    PREV_T,
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


// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(PREV_T, T), z=B]
void __global__ apply_complex_rope_forward_fp16_kernel_write_dynamic_cache(
  const float* __restrict__ position_embeds, // shape [B, T, embed_dim // 2, 2]
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  half* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  half* __restrict__ k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  const half* __restrict__ prev_k_cache, // [B, num_k_heads, PREV_T, embed_dim]
  const half* __restrict__ prev_v_cache, // [B, num_v_heads, PREV_T, embed_dim]
  half* __restrict__ k_cache_out, // [B, num_k_heads, PREV_T + T, embed_dim]
  half* __restrict__ v_cache_out, // [B, num_v_heads, PREV_T + T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned PREV_T, // number of tokens previously in the KV cache
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
    // one block does one head of a new token
    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = blockIdx.y; // should be launched with max(PREV_T, T)
    unsigned batch_idx = blockIdx.z;
    unsigned pos_idx = batch_idx * T + tok_idx;

    // determine if we are supposed to transform an embedding from q or k,
    // and which head
    const half* __restrict__ embeds_in;
    half* __restrict__ embeds_out;
    const half* __restrict__ prev_cache;
    half* __restrict__ cache_out;
    unsigned num_heads;

    // strides
    unsigned batch_stride;
    unsigned head_stride;
    unsigned tok_stride;

    // determine if we are processing q, k or v
    if (head_idx < num_q_heads) {
        embeds_in = q_in;
        embeds_out = q_out;
        // no q cache
        prev_cache = nullptr;
        cache_out = nullptr;
        num_heads = num_q_heads;
  
        batch_stride = q_in_batch_stride;
        head_stride = q_in_head_stride;
        tok_stride = q_in_tok_stride;
    } else if (head_idx < num_q_heads + num_k_heads) {
        // k
        embeds_in = k_in;
        embeds_out = k_out;
        prev_cache = prev_k_cache;
        cache_out = k_cache_out;
        num_heads = num_k_heads;

        batch_stride = k_in_batch_stride;
        head_stride = k_in_head_stride;
        tok_stride = k_in_tok_stride;

        head_idx -= num_q_heads;
    } else {
        // v
        embeds_in = v_in;
        embeds_out = nullptr;
        prev_cache = prev_v_cache;
        cache_out = v_cache_out;
        num_heads = num_v_heads;

        batch_stride = v_in_batch_stride;
        head_stride = v_in_head_stride;
        tok_stride = v_in_tok_stride;

        head_idx -= (num_q_heads + num_k_heads);
    }

    // index for where to read into the cos/sin caches, if we need to
    // (try to trigger the read before the cache copy - need to check if done by the compiler)
    uint64_t position_id;

    // ROTARY_CACHE_BTE_LAYOUT
    position_id = tok_idx < T ? pos_idx : 0;

    // realign the pointer to where we are supposed to write out if needed

    // if dynamic cache, copy previous content here
    if (prev_cache != nullptr){
        unsigned NEW_T = PREV_T + T;

        // realign the cache pointers according to where we are supposed to read/write the token
        // we are in charge of
        prev_cache = &prev_cache[(((batch_idx * num_heads) + head_idx) * PREV_T + tok_idx) * embed_dim + 0];
        half* __restrict__ cache_out_copy = &cache_out[(((batch_idx * num_heads) + head_idx) * NEW_T + tok_idx) * embed_dim + 0];

        // vectorized part: NOT WORKING
        unsigned d = threadIdx.x;//2 * threadIdx.x;
        /*
        for (; d + 1 < embed_dim; d += (2 * THREADS_PER_BLOCK)) {
            half2 c = *addr((const half2*) prev_cache, d);
            *addr((half2*) cache_out_copy, d) = c;
        }
        */
        // loop remainder
        for (; d < embed_dim; d += THREADS_PER_BLOCK) {
            *addr(cache_out_copy, d) = *addr(prev_cache, d);
        }
    }

    // realign the position embeds to the position
    position_embeds = &position_embeds[position_id * embed_dim];

    if (tok_idx >= T) {
        // no tokens to apply the rotary embeddings to, we were just there to do the copy
        return;
    }

    // realign embeds_in and embeds_out
    // q/k/v might be strided, but embedding dimension stride needs to be 1
    unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
    embeds_in = &embeds_in[embed_in_idx];

    if (embeds_out != nullptr) {
      unsigned embed_out_idx = ((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0;
      embeds_out = &embeds_out[embed_out_idx];
    }

    // realign cache_out
    half* __restrict__ cache_out_write = nullptr;
    if (cache_out != nullptr) {
      unsigned NEW_T = PREV_T + T;
      cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * NEW_T + (PREV_T + tok_idx)) * embed_dim + 0];
    }

    if (embeds_out != nullptr) {
      // q or k

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

        // TODO: variants with and without cache write?
        // write the new token in cache
        if (cache_out_write != nullptr) {
          *((half2*)addr(cache_out_write, d)) = rot_embed;
        }
      }
    } else {
      // v
      unsigned d = threadIdx.x;
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        half embed = *addr(embeds_in, d);
        *addr(cache_out_write, d) = embed;
      }
    }
}

void muillm_apply_complex_rope_forward_fp16_dynamic_cache(
  hipStream_t stream,
  unsigned B,
  unsigned T,
  unsigned PREV_T,
  unsigned num_q_heads,
  unsigned num_k_heads,
  unsigned num_v_heads,
  unsigned embed_dim,
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride,
  const float* position_embeds, // shape [B, T, embed_dim // 2, 2]
  const half* q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* v_in, // shape [B, num_v_heads, T, embed_dim]
  half* q_out, // shape [B, num_q_heads, T, embed_dim]
  half* k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  const half* prev_k_cache, // [B, num_k_heads, PREV_T, embed_dim]
  const half* prev_v_cache, // [B, num_v_heads, PREV_T, embed_dim]
  half* k_cache_out, // [B, num_k_heads, PREV_T + T, embed_dim]
  half* v_cache_out // [B, num_v_heads, PREV_T + T, embed_dim]
) {

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);


  // expected block dimensions: [x=num_q_heads+num_k_heads, y=max(PREV_T, T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, std::max(PREV_T, T), B);

  apply_complex_rope_forward_fp16_kernel_write_dynamic_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    position_embeds,
    (const half*)q_in,
    (const half*)k_in,
    (const half*)v_in,
    (half*)q_out,
    (half*)k_out,
    // KV cache
    (const half*)prev_k_cache,
    (const half*)prev_v_cache,
    (half*)k_cache_out,
    (half*)v_cache_out,
    // tensor dimension sizes
    B,
    T,
    PREV_T,
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

// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(PREV_T, T), z=B]
void __global__ apply_complex_rope_forward_bf16_kernel_write_dynamic_cache(
  const float* __restrict__ position_embeds, // shape [B, T, embed_dim // 2, 2]
  const __hip_bfloat16* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  const __hip_bfloat16* __restrict__ prev_k_cache, // [B, num_k_heads, PREV_T, embed_dim]
  const __hip_bfloat16* __restrict__ prev_v_cache, // [B, num_v_heads, PREV_T, embed_dim]
  __hip_bfloat16* __restrict__ k_cache_out, // [B, num_k_heads, PREV_T + T, embed_dim]
  __hip_bfloat16* __restrict__ v_cache_out, // [B, num_v_heads, PREV_T + T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned PREV_T, // number of tokens previously in the KV cache
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
    // one block does one head of a new token
    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = blockIdx.y; // should be launched with max(PREV_T, T)
    unsigned batch_idx = blockIdx.z;
    unsigned pos_idx = batch_idx * T + tok_idx;

    // determine if we are supposed to transform an embedding from q or k,
    // and which head
    const __hip_bfloat16* __restrict__ embeds_in;
    __hip_bfloat16* __restrict__ embeds_out;
    const __hip_bfloat16* __restrict__ prev_cache;
    __hip_bfloat16* __restrict__ cache_out;
    unsigned num_heads;

    // strides
    unsigned batch_stride;
    unsigned head_stride;
    unsigned tok_stride;

    // determine if we are processing q, k or v
    if (head_idx < num_q_heads) {
        embeds_in = q_in;
        embeds_out = q_out;
        // no q cache
        prev_cache = nullptr;
        cache_out = nullptr;
        num_heads = num_q_heads;
  
        batch_stride = q_in_batch_stride;
        head_stride = q_in_head_stride;
        tok_stride = q_in_tok_stride;
    } else if (head_idx < num_q_heads + num_k_heads) {
        // k
        embeds_in = k_in;
        embeds_out = k_out;
        prev_cache = prev_k_cache;
        cache_out = k_cache_out;
        num_heads = num_k_heads;

        batch_stride = k_in_batch_stride;
        head_stride = k_in_head_stride;
        tok_stride = k_in_tok_stride;

        head_idx -= num_q_heads;
    } else {
        // v
        embeds_in = v_in;
        embeds_out = nullptr;
        prev_cache = prev_v_cache;
        cache_out = v_cache_out;
        num_heads = num_v_heads;

        batch_stride = v_in_batch_stride;
        head_stride = v_in_head_stride;
        tok_stride = v_in_tok_stride;

        head_idx -= (num_q_heads + num_k_heads);
    }

    // index for where to read into the cos/sin caches, if we need to
    // (try to trigger the read before the cache copy - need to check if done by the compiler)
    uint64_t position_id;

    // ROTARY_CACHE_BTE_LAYOUT
    position_id = tok_idx < T ? pos_idx : 0;

    // realign the pointer to where we are supposed to write out if needed

    // if dynamic cache, copy previous content here
    if (prev_cache != nullptr){
        unsigned NEW_T = PREV_T + T;

        // realign the cache pointers according to where we are supposed to read/write the token
        // we are in charge of
        prev_cache = &prev_cache[(((batch_idx * num_heads) + head_idx) * PREV_T + tok_idx) * embed_dim + 0];
        __hip_bfloat16* __restrict__ cache_out_copy = &cache_out[(((batch_idx * num_heads) + head_idx) * NEW_T + tok_idx) * embed_dim + 0];

        // vectorized part: NOT WORKING
        unsigned d = threadIdx.x;//2 * threadIdx.x;
        /*
        for (; d + 1 < embed_dim; d += (2 * THREADS_PER_BLOCK)) {
            __hip_bfloat162 c = *addr((const __hip_bfloat162*) prev_cache, d);
            *addr((__hip_bfloat162*) cache_out_copy, d) = c;
        }
        */
        // loop remainder
        for (; d < embed_dim; d += THREADS_PER_BLOCK) {
            *addr(cache_out_copy, d) = *addr(prev_cache, d);
        }
    }

    // realign the position embeds to the position
    position_embeds = &position_embeds[position_id * embed_dim];

    if (tok_idx >= T) {
        // no tokens to apply the rotary embeddings to, we were just there to do the copy
        return;
    }


    // realign embeds_in and embeds_out
    // q/k/v might be strided, but embedding dimension stride needs to be 1
    unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
    embeds_in = &embeds_in[embed_in_idx];

    if (embeds_out != nullptr) {
      unsigned embed_out_idx = ((batch_idx * num_heads + head_idx) * T + tok_idx) * embed_dim + 0;
      embeds_out = &embeds_out[embed_out_idx];
    }

    // realign cache_out
    __hip_bfloat16* __restrict__ cache_out_write = nullptr;
    if (cache_out != nullptr) {
      unsigned NEW_T = PREV_T + T;
      cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * NEW_T + (PREV_T + tok_idx)) * embed_dim + 0];
    }

    if (embeds_out != nullptr) {
      // q or k
      unsigned half_embed_dim = embed_dim / 2;

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

        // TODO: variants with and without cache write?
        // write the new token in cache
        if (cache_out_write != nullptr) {
          *((__hip_bfloat162*)addr(cache_out_write, d)) = rot_embed;
        }
      }
    } else {
      // v
      unsigned d = threadIdx.x;
      for (; d < embed_dim; d += THREADS_PER_BLOCK) {
        __hip_bfloat16 embed = *addr(embeds_in, d);
        *addr(cache_out_write, d) = embed;
      }
    }
}

void muillm_apply_complex_rope_forward_bf16_dynamic_cache(
  hipStream_t stream,
  unsigned B,
  unsigned T,
  unsigned PREV_T,
  unsigned num_q_heads,
  unsigned num_k_heads,
  unsigned num_v_heads,
  unsigned embed_dim,
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  unsigned v_in_batch_stride,
  unsigned v_in_head_stride,
  unsigned v_in_tok_stride,
  const float* position_embeds, // shape [B, T, embed_dim // 2, 2]
  const __hip_bfloat16* q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* q_out, // shape [B, num_q_heads, T, embed_dim]
  __hip_bfloat16* k_out, // shape [B, num_k_heads, T, embed_dim]
  // KV cache
  const __hip_bfloat16* prev_k_cache, // [B, num_k_heads, PREV_T, embed_dim]
  const __hip_bfloat16* prev_v_cache, // [B, num_v_heads, PREV_T, embed_dim]
  __hip_bfloat16* k_cache_out, // [B, num_k_heads, PREV_T + T, embed_dim]
  __hip_bfloat16* v_cache_out // [B, num_v_heads, PREV_T + T, embed_dim]
) {

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);


  // expected block dimensions: [x=num_q_heads+num_k_heads, y=max(PREV_T, T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, std::max(PREV_T, T), B);

  apply_complex_rope_forward_bf16_kernel_write_dynamic_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    position_embeds,
    (const __hip_bfloat16*)q_in,
    (const __hip_bfloat16*)k_in,
    (const __hip_bfloat16*)v_in,
    (__hip_bfloat16*)q_out,
    (__hip_bfloat16*)k_out,
    // KV cache
    (const __hip_bfloat16*)prev_k_cache,
    (const __hip_bfloat16*)prev_v_cache,
    (__hip_bfloat16*)k_cache_out,
    (__hip_bfloat16*)v_cache_out,
    // tensor dimension sizes
    B,
    T,
    PREV_T,
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