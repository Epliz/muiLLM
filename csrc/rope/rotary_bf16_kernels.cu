#include "rotary_position_layout.h"

#include <hip/hip_bf16.h>

#include <cuComplex.h>

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

static inline __device__ __hip_bfloat162 __muillm_make_bfloat162(
  __hip_bfloat16 a, __hip_bfloat16 b
) {
  // make a __hip_bfloat162 from two __hip_bfloat16 values
  return __hip_bfloat162(a, b);
}
// TODOs:
// 1) check layouts
// 2) optimize array addressing in loops

// expected block dimensions: [x=num_q_heads+num_k_heads, y=T, z=B]
void __global__ apply_rope_forward_bf16_kernel_no_cache(
  const uint64_t* __restrict__ position_ids, // shape [B, T]
  const __hip_bfloat16* __restrict__ cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* __restrict__ sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ k_out, // shape [B, num_k_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  // k strides
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  //
  muillm_rotary_cache_layout_t cache_layout
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
    unsigned num_heads;

    // strides
    unsigned batch_stride;
    unsigned head_stride;
    unsigned tok_stride;

    // TODO: !!!!!!!!!!!!!!!!
    // q and v might not have the same number of heads, which causes issues due to how we launch the kernel
    // if we apply rope to q, we copy the v cache with it
    // launch with num_q_heads + num_k_heads + num_v_heads?
    if (head_idx < num_q_heads) {
        embeds_in = q_in;
        embeds_out = q_out;
        num_heads = num_q_heads;
  
        batch_stride = q_in_batch_stride;
        head_stride = q_in_head_stride;
        tok_stride = q_in_tok_stride;
    } else {
        embeds_in = k_in;
        embeds_out = k_out;
        num_heads = num_k_heads;

        batch_stride = k_in_batch_stride;
        head_stride = k_in_head_stride;
        tok_stride = k_in_tok_stride;

        head_idx -= num_q_heads;
    }

    // index for where to read into the cos/sin caches
    // (try to trigger the read before the cache copy - need to check if done by the compiler)
    uint64_t position_id;

    // there are two possible layouts for the cache
    if (cache_layout == ROTARY_CACHE_SE_LAYOUT) {
      position_id = tok_idx < T ? position_ids[pos_idx] : 0;
    } else {
      // ROTARY_CACHE_BTE_LAYOUT
      position_id = tok_idx < T ? pos_idx : 0;
    }

    // realign the pointer to where we are supposed to write out if needed

    // realign the cos/sin caches to the position
    cos_cached = &cos_cached[position_id * embed_dim];
    sin_cached = &sin_cached[position_id * embed_dim];


    // realign embeds_in and embeds_out
    // q/k might be strided, but embedding dimension stride needs to be 1
    unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
    unsigned embed_out_idx = ((batch_idx * num_heads + head_idx)* T + tok_idx) * embed_dim + 0;
    embeds_in = &embeds_in[embed_in_idx];
    embeds_out = &embeds_out[embed_out_idx];

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
    }
}

void muillm_apply_rope_forward_bf16_no_cache(
  hipStream_t stream,
  unsigned B,
  unsigned T,
  unsigned num_q_heads,
  unsigned num_k_heads,
  unsigned embed_dim,
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  muillm_rotary_cache_layout_t cache_layout,
  const uint64_t* position_ids,
  const __hip_bfloat16* cos_cached,
  const __hip_bfloat16* sin_cached,
  const __hip_bfloat16* q_in,
  const __hip_bfloat16* k_in,
  __hip_bfloat16* q_out,
  __hip_bfloat16* k_out
) {

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);

  // expected block dimensions: [x=num_q_heads+num_k_heads, y=max(PREV_T, T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads, T, B);

  apply_rope_forward_bf16_kernel_no_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint64_t*)position_ids,
    (const __hip_bfloat16*)cos_cached,
    (const __hip_bfloat16*)sin_cached,
    (const __hip_bfloat16*)q_in,
    (const __hip_bfloat16*)k_in,
    (__hip_bfloat16*)q_out,
    (__hip_bfloat16*)k_out,
    // tensor dimension sizes
    B,
    T,
    num_q_heads,
    num_k_heads,
    embed_dim,
    // q strides
    q_in_batch_stride,
    q_in_head_stride,
    q_in_tok_stride,
    // k strides
    k_in_batch_stride,
    k_in_head_stride,
    k_in_tok_stride,
    // cache layout
    cache_layout
  );
}

// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(PREV_T, T), z=B]
void __global__ apply_rope_forward_bf16_kernel_write_dynamic_cache(
  const uint64_t* __restrict__ position_ids, // shape [B, T]
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
  unsigned v_in_tok_stride,
  //
  muillm_rotary_cache_layout_t cache_layout
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

    // there are two possible layouts for the cache
    if (cache_layout == ROTARY_CACHE_SE_LAYOUT) {
      position_id = tok_idx < T ? position_ids[pos_idx] : 0;
    } else {
      // ROTARY_CACHE_BTE_LAYOUT
      position_id = tok_idx < T ? pos_idx : 0;
    }

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
  muillm_rotary_cache_layout_t cache_layout,
  const uint64_t* position_ids, // shape [B, T]
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
    (const uint64_t*)position_ids,
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
    v_in_tok_stride,
    // cache layout
    cache_layout
  );
}

// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
void __global__ apply_rope_forward_bf16_kernel_write_static_cache(
  const uint64_t* __restrict__ position_ids, // shape [B, T]
  const __hip_bfloat16* __restrict__ cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* __restrict__ sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache
  __hip_bfloat16* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  __hip_bfloat16* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
  const uint64_t* __restrict__ cache_position, // [T]
  // tensor dimension sizes
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
  //
  muillm_rotary_cache_layout_t cache_layout
) {
    // one block does one head of a new token
    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = blockIdx.y; // should be launched with T
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

    bool q_or_k = false;
    // determine if we are processing q, k or v
    if (head_idx < num_q_heads) {
        q_or_k = true;
        embeds_in = q_in;
        embeds_out = q_out;
        // no q cache
        cache_out = nullptr;
        num_heads = num_q_heads;
  
        batch_stride = q_in_batch_stride;
        head_stride = q_in_head_stride;
        tok_stride = q_in_tok_stride;
    } else if (head_idx < num_q_heads + num_k_heads) {
        // k
        q_or_k = true;
        embeds_in = k_in;
        embeds_out = nullptr;
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

    // there are two possible layouts for the cache
    if (cache_layout == ROTARY_CACHE_SE_LAYOUT) {
      position_id = tok_idx < T ? position_ids[pos_idx] : 0;
    } else {
      // ROTARY_CACHE_BTE_LAYOUT
      position_id = tok_idx < T ? pos_idx : 0;
    }

    // realign the pointer to where we are supposed to write out if needed

    // realign the cos/sin caches to the position
    cos_cached = &cos_cached[position_id * embed_dim];
    sin_cached = &sin_cached[position_id * embed_dim];


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
      unsigned cache_tok_pos = cache_position[tok_idx];
      cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
    }

    // we need to do rope on both q and k, but don't need to write out k
    if (q_or_k) {
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

        if (embeds_out != nullptr) {
          *addr(embeds_out, d) = r;
        }

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

        if (embeds_out != nullptr) {
          *addr(embeds_out, d) = r;
        }

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

void muillm_apply_rope_forward_bf16_static_cache(
  hipStream_t stream,
  unsigned B,
  unsigned T,
  unsigned MAX_T,
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
  muillm_rotary_cache_layout_t cache_layout,
  const uint64_t* position_ids, // shape [B, T]
  const __hip_bfloat16* cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const __hip_bfloat16* q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache
  __hip_bfloat16* k_cache, // [B, num_k_heads, MAX_T, embed_dim]
  __hip_bfloat16* v_cache, // [B, num_v_heads, MAX_T, embed_dim]
  const uint64_t* cache_position // [T] - positions of the new tokens in
) {

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);


  // expected block dimensions: [x=num_q_heads+num_k_heads, y=T, z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, T, B);

  apply_rope_forward_bf16_kernel_write_static_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint64_t*)position_ids,
    (const __hip_bfloat16*)cos_cached,
    (const __hip_bfloat16*)sin_cached,
    (const __hip_bfloat16*)q_in,
    (const __hip_bfloat16*)k_in,
    (const __hip_bfloat16*)v_in,
    (__hip_bfloat16*)q_out,
    // KV cache
    (__hip_bfloat16*)k_cache,
    (__hip_bfloat16*)v_cache,
    (const uint64_t*)cache_position,
    // tensor dimension sizes
    B,
    T,
    MAX_T,
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
    v_in_tok_stride,
    // cache layout
    cache_layout
  );
}

__global__ void apply_complex_rope_forward_bf16_kernel_no_cache(
  const __hip_bfloat16* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const float* __restrict__ position_embeds, // shape [B, T, embed_dim / 2, 2]
  __hip_bfloat16* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  __hip_bfloat16* __restrict__ k_out, // shape [B, num_k_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned num_q_heads, // number of heads for q
  unsigned num_k_heads, // number of heads for k
  unsigned embed_dim, // half of the size of embeddings in each head
  // q strides
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  // k strides
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride
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
    unsigned num_heads;

    // strides
    unsigned batch_stride;
    unsigned head_stride;
    unsigned tok_stride;

    // TODO: !!!!!!!!!!!!!!!!
    // q and v might not have the same number of heads, which causes issues due to how we launch the kernel
    // if we apply rope to q, we copy the v cache with it
    // launch with num_q_heads + num_k_heads + num_v_heads?
    if (head_idx < num_q_heads) {
        embeds_in = q_in;
        embeds_out = q_out;
        num_heads = num_q_heads;
  
        batch_stride = q_in_batch_stride;
        head_stride = q_in_head_stride;
        tok_stride = q_in_tok_stride;
    } else {
        embeds_in = k_in;
        embeds_out = k_out;
        num_heads = num_k_heads;

        batch_stride = k_in_batch_stride;
        head_stride = k_in_head_stride;
        tok_stride = k_in_tok_stride;

        head_idx -= num_q_heads;
    }

    // index for where to read into the cos/sin caches
    // (try to trigger the read before the cache copy - need to check if done by the compiler)
    uint64_t position_id;


    // ROTARY_CACHE_BTE_LAYOUT
    position_id = tok_idx < T ? pos_idx : 0;

    // realign the pointer to where we are supposed to write out if needed

    // realign the cos/sin caches to the position
    position_embeds = &position_embeds[position_id * embed_dim];


    // realign embeds_in and embeds_out
    // q/k might be strided, but embedding dimension stride needs to be 1
    unsigned embed_in_idx = batch_idx * batch_stride + head_idx * head_stride + tok_idx * tok_stride;
    unsigned embed_out_idx = ((batch_idx * num_heads + head_idx)* T + tok_idx) * embed_dim + 0;
    embeds_in = &embeds_in[embed_in_idx];
    embeds_out = &embeds_out[embed_out_idx];

    unsigned d = 2 * threadIdx.x;
    for (; d + 1 < embed_dim; d += 2 * THREADS_PER_BLOCK) {
        float2 cos_sin = *(const float2*)addr(position_embeds, d);
        __hip_bfloat16 cos = __float2bfloat16(cos_sin.x);
        __hip_bfloat16 sin = __float2bfloat16(cos_sin.y);

        __hip_bfloat162 embed = *(const __hip_bfloat162*)addr(embeds_in, d);
        __hip_bfloat16 real_embed = embed.x;
        __hip_bfloat16 imag_embed = embed.y;

        __hip_bfloat16 real_rot_embed = __hfma(real_embed, cos, __hneg(__hmul(imag_embed, sin)));
        __hip_bfloat16 imag_rot_embed = __hfma(real_embed, sin, __hmul(imag_embed, cos));
    
        __hip_bfloat162 rot_embed = __muillm_make_bfloat162(real_rot_embed, imag_rot_embed);

        *(__hip_bfloat162*)addr(embeds_out, d) = rot_embed;
    }
}

void muillm_apply_complex_rope_forward_bf16_no_cache(
  hipStream_t stream,
  unsigned B,
  unsigned T,
  unsigned num_q_heads,
  unsigned num_k_heads,
  unsigned embed_dim,
  unsigned q_in_batch_stride,
  unsigned q_in_head_stride,
  unsigned q_in_tok_stride,
  unsigned k_in_batch_stride,
  unsigned k_in_head_stride,
  unsigned k_in_tok_stride,
  const float* position_embeds, // shape [B, T, embed_dim / 2, 2]
  const __hip_bfloat16* q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* k_in, // shape [B, num_k_heads, T, embed_dim]
  __hip_bfloat16* q_out, // shape [B, num_q_heads, T, embed_dim]
  __hip_bfloat16* k_out // shape [B, num_k_heads, T, embed_dim]
) {

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);

  // expected block dimensions: [x=num_q_heads+num_k_heads, y=max(PREV_T, T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads, T, B);

  apply_complex_rope_forward_bf16_kernel_no_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const __hip_bfloat16*)q_in,
    (const __hip_bfloat16*)k_in,
    (const float*)position_embeds,
    (__hip_bfloat16*)q_out,
    (__hip_bfloat16*)k_out,
    // tensor dimension sizes
    B,
    T,
    num_q_heads,
    num_k_heads,
    embed_dim,
    // q strides
    q_in_batch_stride,
    q_in_head_stride,
    q_in_tok_stride,
    // k strides
    k_in_batch_stride,
    k_in_head_stride,
    k_in_tok_stride
  );
}
