#include "sliding_kvcache.hpp"

#include <ATen/cuda/CUDAContext.h>

#include <stdint.h>

typedef uint16_t xx16;

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
);

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
);

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
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


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

  auto dtype = k_in.dtype();
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
    if (dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
      // We can use the sliding_kvcache_prefill kernel
      sliding_kvcache_prefill_xx16(
        stream,
        (const xx16*)k_in.data_ptr(),
        (const xx16*)v_in.data_ptr(),
        // KV cache out
        (xx16*)k_cache.data_ptr(),
        (xx16*)v_cache.data_ptr(),
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
    } else {
      TORCH_CHECK(false, "Unsupported dtype for sliding_kvcache_prefill");
    }

    return std::make_tuple(k_in, v_in);
  } else if (is_full) {
    // previously full or getting full -> overwrite

    // allocate new tensors to store the latest k and v
    auto k_cache_out = torch::empty(k_cache_sizes, output_options);
    auto v_cache_out = torch::empty(v_cache_sizes, output_options);
  
    if (dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
      // We can use the sliding_kvcache_update_overwrite kernel
      sliding_kvcache_update_overwrite_xx16(
        stream,
        (const xx16*)k_in.data_ptr(),
        (const xx16*)v_in.data_ptr(),
        // KV cache in
        (const xx16*)k_cache.data_ptr(),
        (const xx16*)v_cache.data_ptr(),
        // KV cache out
        (xx16*)k_cache_out.data_ptr(),
        (xx16*)v_cache_out.data_ptr(),
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
    } else {
      TORCH_CHECK(false, "Unsupported dtype for sliding_kvcache_update_overwrite");
    }
  
    // caller needs to use these as new caches
    return std::make_tuple(k_cache_out, v_cache_out);
  } else {
    // not full, not becoming full -> normal update, similar to a static cache update
    
    if (dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
      // We can use the sliding_kvcache_update kernel
      sliding_kvcache_update_xx16(
        stream,
        (const xx16*)k_in.data_ptr(),
        (const xx16*)v_in.data_ptr(),
        // KV cache out
        (xx16*)k_cache.data_ptr(),
        (xx16*)v_cache.data_ptr(),
        (const uint64_t*)cache_position.data_ptr(),
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
    } else {
      TORCH_CHECK(false, "Unsupported dtype for sliding_kvcache_update");
    }

    // restrict to as many tokens as seen by the cache
    auto key_states = k_cache.narrow(/* dim */ 2, /* start */0, /* length */ seen_tokens);
    auto value_states = v_cache.narrow(/* dim */ 2, /* start */ 0, /* length */ seen_tokens);
    return std::make_tuple(key_states, value_states);
  }
}