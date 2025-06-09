#include "static_kvcache_kernels.cuh"

#include <ATen/cuda/CUDAContext.h>

#include <cuda_fp16.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
void __global__ sliding_kvcache_prefill(
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  // KV cache
  half* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
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
    const half* __restrict__ X;
    half* __restrict__ cache_out;
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
      half embed = *addr(X, d);
      *addr(cache_out, d) = embed;
    }
}

// expected block dimensions: [x=num_k_heads+num_v_heads, y=MAX_T, z=B]
// always have to copy an entire new cache
void __global__ sliding_kvcache_update_overwrite(
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
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
    const half* __restrict__ X;
    half* __restrict__ cache_out;
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
      half embed = *addr(X, d);
      *addr(cache_out, d) = embed;
    }
}

// expected block dimensions: [x=num_k_heads+num_v_heads, y=T, z=B]
void __global__ sliding_kvcache_update(
  const half* __restrict__ k_in, // shape [B, num_k_heads, T, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, T, embed_dim]
  // KV cache
  half* __restrict__ k_cache_out, // [B, num_k_heads, MAX_T, embed_dim]
  half* __restrict__ v_cache_out, // [B, num_v_heads, MAX_T, embed_dim]
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
    const half* __restrict__ X;
    half* __restrict__ cache_out;
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
      half embed = *addr(X, d);
      *addr(cache_out, d) = embed;
    }
}

std::tuple<at::Tensor, at::Tensor> muillm_sliding_kvcache_update(
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& cache_position,
    uint64_t seen_tokens
) {
  // q, k, v are expected to not be contiguous
  // (due to the fact that we compute them packed as qkv and transposition afterwards)
  CHECK_CUDA(k_in);
  CHECK_CUDA(v_in);
  CHECK_INPUT(k_cache);
  CHECK_INPUT(v_cache);
  CHECK_INPUT(cache_position);

  auto device = k_in.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = torch::kFloat16;
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto k_sizes = k_in.sizes().vec();
  auto k_strides = k_in.strides().vec();

  auto v_sizes = v_in.sizes().vec();
  auto v_strides = v_in.strides().vec();

  auto k_cache_sizes = k_cache.sizes().vec();
  auto v_cache_sizes = v_cache.sizes().vec();

  unsigned B = k_sizes[0];
  unsigned T = k_sizes[2];
  unsigned MAX_T = k_cache_sizes[2];
  unsigned num_k_heads = k_sizes[1];
  unsigned num_v_heads = v_sizes[1];
  unsigned embed_dim = k_sizes[3];

  // k strides
  unsigned k_in_batch_stride = k_strides[0];
  unsigned k_in_head_stride = k_strides[1];
  unsigned k_in_tok_stride = k_strides[2];
  // v strides
  unsigned v_in_batch_stride = v_strides[0];
  unsigned v_in_head_stride = v_strides[1];
  unsigned v_in_tok_stride = v_strides[2];

  bool is_full = seen_tokens > MAX_T;

  if (seen_tokens == T) {
    // Prefill (we incremented seen_tokens before updating the cache)
    const unsigned COPIED_T = std::min(T, MAX_T);
    const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);
    // We return all tokens in that case to avoid catastrophic forgetting
    // but store in the cache the latest ones
    const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
    // expected block dimensions: [x=num_k_heads, y=T, z=B]
    // TODO: max cuda block dimension is 1024, so need to do something when T>1024
    const dim3 num_blocks = dim3(num_k_heads + num_v_heads, COPIED_T, B);

    sliding_kvcache_prefill<<<num_blocks, threads_per_blocks, 0, stream>>>(
      (const half*)k_in.data_ptr(),
      (const half*)v_in.data_ptr(),
      // KV cache
      (half*)k_cache.data_ptr(),
      (half*)v_cache.data_ptr(),
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

    return std::make_tuple(k_in, v_in);
  } else if (is_full) {
    // previously full or getting full

    // determine what part of the new tokens to copy
    // if we have a lot of input tokens, we actually copy only the MAX_T last ones
    const unsigned START_T = (T < MAX_T) ? 0 : (T - MAX_T);

    // allocate new tensors to store the latest k and v
    auto k_cache_out = torch::empty(k_cache_sizes, output_options);
    auto v_cache_out = torch::empty(v_cache_sizes, output_options);

    // We return all tokens in that case to avoid catastrophic forgetting
    // but store in the cache the latest ones
    const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
    // expected block dimensions: [x=num_k_heads, y=T, z=B]
    // TODO: max cuda block dimension is 1024, so need to do something when MAX_T>1024
    const dim3 num_blocks = dim3(num_k_heads + num_v_heads, MAX_T, B);

    sliding_kvcache_update_overwrite<<<num_blocks, threads_per_blocks, 0, stream>>>(
      (const half*)k_in.data_ptr(),
      (const half*)v_in.data_ptr(),
      // KV cache in
      (const half*)k_cache.data_ptr(),
      (const half*)v_cache.data_ptr(),
      // KV cache out
      (half*)k_cache_out.data_ptr(),
      (half*)v_cache_out.data_ptr(),
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

    // caller needs to use these as new caches
    return std::make_tuple(k_cache_out, v_cache_out);
  } else {
    // not full, not becoming full -> similar to a static cache update
    const dim3 threads_per_blocks = dim3(THREADS_PER_BLOCK);
    // expected block dimensions: [x=num_k_heads, y=T, z=B]
    // TODO: max cuda block dimension is 1024, so need to do something when T>1024
    const dim3 num_blocks = dim3(num_k_heads + num_v_heads, T, B);

    sliding_kvcache_update<<<num_blocks, threads_per_blocks, 0, stream>>>(
      (const half*)k_in.data_ptr(),
      (const half*)v_in.data_ptr(),
      // KV cache
      (half*)k_cache.data_ptr(),
      (half*)v_cache.data_ptr(),
      (const uint64_t*)cache_position.data_ptr(),
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

    // restrict to as many tokens as seen by the cache
    auto key_states = k_cache.narrow(/* dim */ 2, /* start */0, /* length */ seen_tokens);
    auto value_states = v_cache.narrow(/* dim */ 2, /* start */ 0, /* length */ seen_tokens);
    return std::make_tuple(key_states, value_states);
  }
}