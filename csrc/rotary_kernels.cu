#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>


#include <stdint.h>
#include <vector>
#include <algorithm>

#include <iostream>

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

// TODOs:
// 1) check layouts
// 2) optimize array addressing in loops


// expected block dimensions: [x=num_q_heads+num_k_heads, y=T, z=B]
void __global__ apply_rope_kernel_no_cache(
  const uint64_t* __restrict__ position_ids, // shape [B, T]
  const half* __restrict__ cos_cached, // shape [S, embed_dim]
  const half* __restrict__ sin_cached, // shape [S, embed_dim]
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  half* __restrict__ q_out, // shape [B, num_q_heads, T, embed_dim]
  half* __restrict__ k_out, // shape [B, num_k_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned B, // batch size
  unsigned T, // num new tokens
  unsigned S, // number of rotary embeddings in the rotary cache
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
  // v strides?
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
    // TODO: v cache case

    // index for where to read into the cos/sin caches, if we need to
    // (try to trigger the read before the cache copy - need to check if done by the compiler)
    uint64_t position_id = tok_idx < T ? position_ids[pos_idx] : 0;

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
    }
}


static inline std::vector<at::Tensor> muillm_rope_forward_no_cache_cuda(
    torch::Tensor& position_ids,
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in
) {

  auto device = q_in.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = torch::kFloat16;
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto cache_sizes = cos_cached.sizes().vec();

  auto q_sizes = q_in.sizes().vec();
  auto q_strides = q_in.strides().vec();
  auto q_out = torch::empty(q_sizes, output_options);

  auto k_sizes = k_in.sizes().vec();
  auto k_strides = k_in.strides().vec();
  auto k_out = torch::empty(k_sizes, output_options);

  unsigned B = q_sizes[0];
  unsigned T = q_sizes[2];
  unsigned S = cache_sizes[0];
  unsigned num_q_heads = q_sizes[1];
  unsigned num_k_heads = k_sizes[1];
  unsigned embed_dim = q_sizes[3];

  // q strides
  unsigned q_in_batch_stride = q_strides[0];
  unsigned q_in_head_stride = q_strides[1];
  unsigned q_in_tok_stride = q_strides[2];
  // k strides
  unsigned k_in_batch_stride = k_strides[0];
  unsigned k_in_head_stride = k_strides[1];
  unsigned k_in_tok_stride = k_strides[2];

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);


  // expected block dimensions: [x=num_q_heads+num_k_heads, y=max(PREV_T, T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads, T, B);

  apply_rope_kernel_no_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint64_t*)position_ids.data_ptr(),
    (const half*)cos_cached.data_ptr(),
    (const half*)sin_cached.data_ptr(),
    (const half*)q_in.data_ptr(),
    (const half*)k_in.data_ptr(),
    (half*)q_out.data_ptr(),
    (half*)k_out.data_ptr(),
    // tensor dimension sizes
    B,
    T,
    S,
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

  return {q_out, k_out};
}

// expected block dimensions: [x=num_q_heads+num_k_heads+num_v_heads, y=max(PREV_T, T), z=B]
void __global__ apply_rope_kernel_write_cache(
  const uint64_t* __restrict__ position_ids, // shape [B, T]
  const half* __restrict__ cos_cached, // shape [S, embed_dim]
  const half* __restrict__ sin_cached, // shape [S, embed_dim]
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
  unsigned S, // number of rotary embeddings in the rotary cache
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
    uint64_t position_id = tok_idx < T ? position_ids[pos_idx] : 0;

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

static inline std::vector<at::Tensor> muillm_rope_forward_write_dynamic_cache_cuda(
    torch::Tensor& position_ids,
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& prev_k_cache,
    torch::Tensor& prev_v_cache
) {

  auto device = q_in.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = torch::kFloat16;
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto cache_sizes = cos_cached.sizes().vec();

  auto q_sizes = q_in.sizes().vec();
  auto q_strides = q_in.strides().vec();
  auto q_out = torch::empty(q_sizes, output_options);

  auto k_sizes = k_in.sizes().vec();
  auto k_strides = k_in.strides().vec();
  auto k_out = torch::empty(k_sizes, output_options);

  auto v_sizes = v_in.sizes().vec();
  auto v_strides = v_in.strides().vec();

  auto prev_k_cache_sizes = prev_k_cache.sizes().vec();
  auto prev_v_cache_sizes = prev_v_cache.sizes().vec();

  unsigned B = q_sizes[0];
  unsigned T = q_sizes[2];
  unsigned S = cache_sizes[0];
  unsigned PREV_T = prev_k_cache_sizes[2];
  unsigned num_q_heads = q_sizes[1];
  unsigned num_k_heads = k_sizes[1];
  unsigned num_v_heads = v_sizes[1];
  unsigned embed_dim = q_sizes[3];

  auto new_k_cache_sizes = prev_k_cache_sizes;
  new_k_cache_sizes[2] += T;
  auto new_v_cache_sizes = prev_v_cache_sizes;
  new_v_cache_sizes[2] += T;

  auto k_cache_out = torch::empty(new_k_cache_sizes, output_options);;
  auto v_cache_out = torch::empty(new_v_cache_sizes, output_options);;

  // q strides
  unsigned q_in_batch_stride = q_strides[0];
  unsigned q_in_head_stride = q_strides[1];
  unsigned q_in_tok_stride = q_strides[2];
  // k strides
  unsigned k_in_batch_stride = k_strides[0];
  unsigned k_in_head_stride = k_strides[1];
  unsigned k_in_tok_stride = k_strides[2];
  // v strides
  unsigned v_in_batch_stride = v_strides[0];
  unsigned v_in_head_stride = v_strides[1];
  unsigned v_in_tok_stride = v_strides[2];

  const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);


  // expected block dimensions: [x=num_q_heads+num_k_heads, y=max(PREV_T, T), z=B]
  const dim3 num_blocks = dim3(num_q_heads + num_k_heads + num_v_heads, std::max(PREV_T, T), B);

  apply_rope_kernel_write_cache<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint64_t*)position_ids.data_ptr(),
    (const half*)cos_cached.data_ptr(),
    (const half*)sin_cached.data_ptr(),
    (const half*)q_in.data_ptr(),
    (const half*)k_in.data_ptr(),
    (const half*)v_in.data_ptr(),
    (half*)q_out.data_ptr(),
    (half*)k_out.data_ptr(),
    // KV cache
    (const half*)prev_k_cache.data_ptr(),
    (const half*)prev_v_cache.data_ptr(),
    (half*)k_cache_out.data_ptr(),
    (half*)v_cache_out.data_ptr(),
    // tensor dimension sizes
    B,
    T,
    S,
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

  return {q_out, k_out, k_cache_out, v_cache_out};
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> muillm_rope_forward_no_cache(
    torch::Tensor& position_ids,
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in
) {
  CHECK_INPUT(position_ids);
  CHECK_INPUT(cos_cached);
  CHECK_INPUT(sin_cached);
  // q, k are expected to not be contiguous
  // (due to the fact that we compute them packed as qkv and transposition afterwards)
  CHECK_CUDA(q_in);
  CHECK_CUDA(k_in);

  return muillm_rope_forward_no_cache_cuda(
    position_ids,
    cos_cached,
    sin_cached,
    q_in,
    k_in
  );
}

std::vector<at::Tensor> muillm_rope_forward_dynamic_cache(
    torch::Tensor& position_ids,
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& prev_k_cache,
    torch::Tensor& prev_v_cache
) {
  CHECK_INPUT(position_ids);
  CHECK_INPUT(cos_cached);
  CHECK_INPUT(sin_cached);
  // q, k, v are expected to not be contiguous
  // (due to the fact that we compute them packed as qkv and transposition afterwards)
  CHECK_CUDA(q_in);
  CHECK_CUDA(k_in);
  CHECK_CUDA(v_in);
  CHECK_INPUT(prev_k_cache);
  CHECK_INPUT(prev_v_cache);

  return muillm_rope_forward_write_dynamic_cache_cuda(
    position_ids,
    cos_cached,
    sin_cached,
    q_in,
    k_in,
    v_in,
    prev_k_cache,
    prev_v_cache
  );
}