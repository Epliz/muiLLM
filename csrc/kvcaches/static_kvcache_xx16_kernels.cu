
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


// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
void __global__ apply_rope_forward_fp16_kernel_write_static_cache(
  const half* __restrict__ cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* __restrict__ sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  half* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache
  half* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
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
  unsigned v_in_tok_stride
) {
    // one block does one head of a new token
    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = blockIdx.y; // should be launched with T
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

    // ROTARY_CACHE_BTE_LAYOUT
    uint64_t position_id = position_id = tok_idx < T ? pos_idx : 0;
  
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
    half* __restrict__ cache_out_write = nullptr;
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
      const half* __restrict__ rot_embeds_in;
      rot_embeds_in = &embeds_in[half_embed_dim];

      unsigned d = threadIdx.x;
      for (; d < half_embed_dim; d += THREADS_PER_BLOCK) {
        half cos_pos = *addr(cos_cached, d);
        half sin_pos = *addr(sin_cached, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = __hneg(*addr(rot_embeds_in, d));
    
        half r = __hfma(rot_embed, sin_pos,__hmul(embed, cos_pos));

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
        half cos_pos = *addr(cos_cached, d);
        half sin_pos = *addr(sin_cached, d);

        half embed = *addr(embeds_in, d);
        half rot_embed = *addr(rot_embeds_in, d);

        half r = __hfma(rot_embed, sin_pos,__hmul(embed, cos_pos));

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
        half embed = *addr(embeds_in, d);
        *addr(cache_out_write, d) = embed;
      }
    }
}

void muillm_apply_rope_forward_fp16_static_cache(
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
  const half* cos_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* sin_cached, // shape [S, embed_dim] or [B, T, embed_dim]
  const half* q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* v_in, // shape [B, num_v_heads, T, embed_dim]
  half* q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache
  half* k_cache, // [B, num_k_heads, MAX_T, embed_dim]
  half* v_cache, // [B, num_v_heads, MAX_T, embed_dim]
  const uint64_t* cache_position // [T] - positions of the new tokens in
) {

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);


  // expected block dimensions: [x=num_q_heads+num_k_heads, y=T, z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, T, B);

  apply_rope_forward_fp16_kernel_write_static_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const half*)cos_cached,
    (const half*)sin_cached,
    (const half*)q_in,
    (const half*)k_in,
    (const half*)v_in,
    (half*)q_out,
    // KV cache
    (half*)k_cache,
    (half*)v_cache,
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
    v_in_tok_stride
  );
}

// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
void __global__ apply_rope_forward_bf16_kernel_write_static_cache(
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
  unsigned v_in_tok_stride
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

    // ROTARY_CACHE_BTE_LAYOUT
    uint64_t position_id = tok_idx < T ? pos_idx : 0;

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
    v_in_tok_stride
  );
}


// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
void __global__ apply_complex_rope_forward_fp16_kernel_write_static_cache(
  const float* __restrict__ position_embeds, // shape [B, T, embed_dim // 2, 2]
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  half* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache
  half* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
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
  unsigned v_in_tok_stride
) {
    // one block does one head of a new token
    unsigned head_idx = blockIdx.x;
    unsigned tok_idx = blockIdx.y; // should be launched with T
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

    // realign the pointer to where we are supposed to write out if needed

    // realign the position_embeddings to the position
    position_embeds = &position_embeds[pos_idx * embed_dim];


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
      unsigned cache_tok_pos = cache_position[tok_idx];
      cache_out_write = &cache_out[(((batch_idx * num_heads) + head_idx) * MAX_T + cache_tok_pos) * embed_dim + 0];
    }

    // we need to do rope on both q and k, but don't need to write out k
    if (q_or_k) {
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

        if (embeds_out != nullptr) {
          *((half2*)addr(embeds_out, d)) = rot_embed;
        }

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

void muillm_apply_complex_rope_forward_fp16_static_cache(
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
  const float* position_embeds, // shape [B, T, embed_dim // 2, 2]
  const half* q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* v_in, // shape [B, num_v_heads, T, embed_dim]
  half* q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache
  half* k_cache, // [B, num_k_heads, MAX_T, embed_dim]
  half* v_cache, // [B, num_v_heads, MAX_T, embed_dim]
  const uint64_t* cache_position // [T] - positions of the new tokens in
) {

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);


  // expected block dimensions: [x=num_q_heads+num_k_heads, y=T, z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, T, B);

  apply_complex_rope_forward_fp16_kernel_write_static_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const float*)position_embeds,
    (const half*)q_in,
    (const half*)k_in,
    (const half*)v_in,
    (half*)q_out,
    // KV cache
    (half*)k_cache,
    (half*)v_cache,
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
    v_in_tok_stride
  );
}

// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=T, z=B]
void __global__ apply_complex_rope_forward_bf16_kernel_write_static_cache(
  const float* __restrict__ position_embeds, // shape [B, T, embed_dim // 2, 2]
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
  unsigned v_in_tok_stride
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

    // realign the pointer to where we are supposed to write out if needed

    // realign the position embeddings to the position
    position_embeds = &position_embeds[pos_idx * embed_dim];


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

        if (embeds_out != nullptr) {
          *((__hip_bfloat162*)addr(embeds_out, d)) = rot_embed;
        }

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

void muillm_apply_complex_rope_forward_bf16_static_cache(
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
  const float* position_embeds, // shape [B, T, embed_dim // 2, 2]
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

  apply_complex_rope_forward_bf16_kernel_write_static_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const float*)position_embeds,
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
    v_in_tok_stride
  );
}