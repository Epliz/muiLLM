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

void __global__ compute_rotary_position_embeddings_bf16_kernel(
  // inputs
  const uint64_t* position_ids, // shape [B, T]
  const __hip_bfloat16* cos_cached, // shape [S, E]
  const __hip_bfloat16* sin_cached, // shape [S, E]
  //outputs
  __hip_bfloat16* embeds_cos, // shape [B, T, E]
  __hip_bfloat16* embeds_sin, // shape [B, T, E]
  // tensor dimension sizes
  unsigned B,
  unsigned T,
  unsigned E
) {

  unsigned batch_idx = blockIdx.y;
  unsigned tok_idx = blockIdx.x;
  unsigned pos_idx = batch_idx * T + tok_idx;

  unsigned position_id = position_ids[pos_idx];

  // realign the cos/sin caches to the position
  cos_cached = &cos_cached[position_id * E];
  sin_cached = &sin_cached[position_id * E];

  // compute the embeddings
  embeds_cos = &embeds_cos[pos_idx * E];
  embeds_sin = &embeds_sin[pos_idx * E];

  // copy the cos/sin cached values to the output
  for (unsigned d = threadIdx.x; d < E; d += THREADS_PER_BLOCK) {
    *addr(embeds_cos, d) = *addr(cos_cached, d);
    *addr(embeds_sin, d) = *addr(sin_cached, d);
  }
}

void muillm_compute_rotary_embed_positions_bf16(
  hipStream_t stream,
  unsigned B,
  unsigned T,
  unsigned E,
  const uint64_t* position_ids,
  const __hip_bfloat16* cos_cached,
  const __hip_bfloat16* sin_cached,
  __hip_bfloat16* embeds_cos,
  __hip_bfloat16* embeds_sin
) {

  const unsigned threads_per_block = THREADS_PER_BLOCK;
  // TODO: grid size cannot be bigger than 65535, so we would need to make a persistent kernel
  // if above
  const dim3 num_blocks = dim3(T, B);

  compute_rotary_position_embeddings_bf16_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
    (const uint64_t*)position_ids,
    (const __hip_bfloat16*)cos_cached,
    (const __hip_bfloat16*)sin_cached,
    (__hip_bfloat16*)embeds_cos,
    (__hip_bfloat16*)embeds_sin,
    // tensor dimension sizes
    B,
    T,
    E
  );
}

// TODOs:
// 1) check layouts
// 2) optimize array addressing in loops

// expected block dimensions: [x=num_q_heads+num_k_heads, y=T, z=B]
void __global__ apply_rope_forward_bf16_kernel_no_cache(
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

    // ROTARY_CACHE_BTE_LAYOUT
    uint64_t position_id = position_id = tok_idx < T ? pos_idx : 0;

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
    k_in_tok_stride
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
        float cos = cos_sin.x;
        float sin = cos_sin.y;

        float2 embed = __bfloat1622float2(*(const __hip_bfloat162*)addr(embeds_in, d));
        float real_embed = embed.x;
        float imag_embed = embed.y;

        float real_rot_embed = fma(real_embed, cos, -imag_embed * sin);
        float imag_rot_embed = fma(real_embed, sin, imag_embed * cos);

        float2 rot_embed = make_float2(real_rot_embed, imag_rot_embed);

        *(__hip_bfloat162*)addr(embeds_out, d) =  __float22bfloat162_rn(rot_embed);
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
