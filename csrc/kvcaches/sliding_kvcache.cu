#include "sliding_kvcache.hpp"

#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

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

void static_kvcache_update_xx16(
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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
  const float* position_embeds, // shape [B, T, embed_dim// 2, 2]
  const __hip_bfloat16* q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* k_in, // shape [B, num_k_heads, T, embed_dim]
  const __hip_bfloat16* v_in, // shape [B, num_v_heads, T, embed_dim]
  __hip_bfloat16* q_out, // shape [B, num_q_heads, T, embed_dim]
  // KV cache
  __hip_bfloat16* k_cache, // [B, num_k_heads, MAX_T, embed_dim]
  __hip_bfloat16* v_cache, // [B, num_v_heads, MAX_T, embed_dim]
  const uint64_t* cache_position // [T] - positions of the new tokens in
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
  // k, v are expected to not be contiguous
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

  auto k_sizes = k_in.sizes();
  auto k_strides = k_in.strides();

  auto v_sizes = v_in.sizes();
  auto v_strides = v_in.strides();

  auto k_cache_sizes = k_cache.sizes();
  auto v_cache_sizes = v_cache.sizes();

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
      // We can use the static_kvcache_update_xx16 kernel
      static_kvcache_update_xx16(
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> muillm_rope_forward_sliding_cache(
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& cache_position,
    uint64_t seen_tokens
) {
  CHECK_INPUT(cos_cached);
  CHECK_INPUT(sin_cached);
  // q, k, v are expected to not be contiguous
  // (due to the fact that we compute them packed as qkv and transposition afterwards)
  CHECK_CUDA(q_in);
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

  auto cache_sizes = cos_cached.sizes();

  auto q_sizes = q_in.sizes();
  auto q_strides = q_in.strides();
  auto q_out = torch::empty(q_sizes, output_options);

  auto k_sizes = k_in.sizes();
  auto k_strides = k_in.strides();

  auto v_sizes = v_in.sizes();
  auto v_strides = v_in.strides();

  auto k_cache_sizes = k_cache.sizes();
  auto v_cache_sizes = v_cache.sizes();

  unsigned B = k_sizes[0];
  unsigned T = k_sizes[2];
  unsigned MAX_T = k_cache_sizes[2];
  unsigned num_q_heads = q_sizes[1];
  unsigned num_k_heads = k_sizes[1];
  unsigned num_v_heads = v_sizes[1];
  unsigned embed_dim = k_sizes[3];

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

  muillm_rotary_cache_layout_t cache_layout;

  auto cache_dim = cache_sizes.size();
  if (cache_dim == 2) {
    cache_layout = ROTARY_CACHE_SE_LAYOUT;
  } else if (cache_dim == 3) {
    cache_layout = ROTARY_CACHE_BTE_LAYOUT;
  } else {
    TORCH_CHECK(false, "Unknown rotary cache layout");
  }

  TORCH_CHECK(cache_layout == ROTARY_CACHE_BTE_LAYOUT, "the cache layout must be BTE");

  TORCH_CHECK(cos_cached.dtype() == dtype, "cos_cached must be the same as q,k,v");
  TORCH_CHECK(sin_cached.dtype() == dtype, "sin_cached must be the same as q,k,v");

  bool is_full = seen_tokens > MAX_T;

  if (seen_tokens == T) {
    // Prefill (we incremented seen_tokens before updating the cache)

    // k_out is of different size than k_cache (potentially bigger)
    auto k_out = torch::empty(k_sizes, output_options);

    if (dtype == torch::kFloat16) {
      // We can use the sliding_kvcache_prefill kernel
      sliding_kvcache_rope_forward_prefill_fp16(
        stream,
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
        (const half*)cos_cached.data_ptr(),
        (const half*)sin_cached.data_ptr(),
        (const half*)q_in.data_ptr(),
        (const half*)k_in.data_ptr(),
        (const half*)v_in.data_ptr(),
        (half*)q_out.data_ptr(),
        (half*)k_out.data_ptr(),
        // KV cache out
        (half*)k_cache.data_ptr(),
        (half*)v_cache.data_ptr()
      );
    } else if (dtype == torch::kBFloat16) {
      // We can use the sliding_kvcache_prefill kernel
      sliding_kvcache_rope_forward_prefill_bf16(
        stream,
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
        (const __hip_bfloat16*)cos_cached.data_ptr(),
        (const __hip_bfloat16*)sin_cached.data_ptr(),
        (const __hip_bfloat16*)q_in.data_ptr(),
        (const __hip_bfloat16*)k_in.data_ptr(),
        (const __hip_bfloat16*)v_in.data_ptr(),
        (__hip_bfloat16*)q_out.data_ptr(),
        (__hip_bfloat16*)k_out.data_ptr(),
        // KV cache out
        (__hip_bfloat16*)k_cache.data_ptr(),
        (__hip_bfloat16*)v_cache.data_ptr()
      );
    } else {
      TORCH_CHECK(false, "Unsupported dtype for sliding_kvcache_prefill");
    }

    return std::make_tuple(q_out, k_out, v_in);
  } else if (is_full) {

    // previously full or getting full -> overwrite

    // The case T > MAX_T is not supported for overwriting the cache as in that case
    // we would need to also output a k_out which we don't support
    TORCH_CHECK(T <= MAX_T, "T must be <= MAX_T when overwriting the cache");

    // allocate new tensors to store the latest k and v
    auto k_cache_out = torch::empty(k_cache_sizes, output_options);
    auto v_cache_out = torch::empty(v_cache_sizes, output_options);
  
    if (dtype == torch::kFloat16) {
      // We can use the sliding_kvcache_update_overwrite kernel
      sliding_kvcache_rope_forward_update_overwrite_fp16(
        stream,
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
        (const half*)cos_cached.data_ptr(),
        (const half*)sin_cached.data_ptr(),
        (const half*)q_in.data_ptr(),
        (const half*)k_in.data_ptr(),
        (const half*)v_in.data_ptr(),
        (half*)q_out.data_ptr(),
        // KV cache in
        (const half*)k_cache.data_ptr(),
        (const half*)v_cache.data_ptr(),
        // KV cache out
        (half*)k_cache_out.data_ptr(),
        (half*)v_cache_out.data_ptr()
      );
    } else if (dtype == torch::kBFloat16) {
      // We can use the sliding_kvcache_update_overwrite kernel
      sliding_kvcache_rope_forward_update_overwrite_bf16(
        stream,
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
        (const __hip_bfloat16*)cos_cached.data_ptr(),
        (const __hip_bfloat16*)sin_cached.data_ptr(),
        (const __hip_bfloat16*)q_in.data_ptr(),
        (const __hip_bfloat16*)k_in.data_ptr(),
        (const __hip_bfloat16*)v_in.data_ptr(),
        (__hip_bfloat16*)q_out.data_ptr(),
        // KV cache in
        (const __hip_bfloat16*)k_cache.data_ptr(),
        (const __hip_bfloat16*)v_cache.data_ptr(),
        // KV cache out
        (__hip_bfloat16*)k_cache_out.data_ptr(),
        (__hip_bfloat16*)v_cache_out.data_ptr()
      );
    } else {
      TORCH_CHECK(false, "Unsupported dtype for sliding_kvcache_update_overwrite");
    }
  
    // caller needs to use these as new caches
    return std::make_tuple(q_out, k_cache_out, v_cache_out);
  } else {
    // not full, not becoming full -> normal update, similar to a static cache update
    
    if (dtype == torch::kFloat16) {
      // We can use the static kernel
      muillm_apply_rope_forward_fp16_static_cache(
        stream,
        B,
        T,
        MAX_T,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        embed_dim,
        q_in_batch_stride,
        q_in_head_stride,
        q_in_tok_stride,
        k_in_batch_stride,
        k_in_head_stride,
        k_in_tok_stride,
        v_in_batch_stride,
        v_in_head_stride,
        v_in_tok_stride,
        (const half*)cos_cached.data_ptr(),
        (const half*)sin_cached.data_ptr(),
        (const half*)q_in.data_ptr(),
        (const half*)k_in.data_ptr(),
        (const half*)v_in.data_ptr(),
        (half*)q_out.data_ptr(),
        // KV cache
        (half*)k_cache.data_ptr(),
        (half*)v_cache.data_ptr(),
        (const uint64_t*)cache_position.data_ptr()
      );
    } else if (dtype == torch::kBFloat16) {
      // We can use the static kernel
      muillm_apply_rope_forward_bf16_static_cache(
        stream,
        B,
        T,
        MAX_T,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        embed_dim,
        q_in_batch_stride,
        q_in_head_stride,
        q_in_tok_stride,
        k_in_batch_stride,
        k_in_head_stride,
        k_in_tok_stride,
        v_in_batch_stride,
        v_in_head_stride,
        v_in_tok_stride,
        (const __hip_bfloat16*)cos_cached.data_ptr(),
        (const __hip_bfloat16*)sin_cached.data_ptr(),
        (const __hip_bfloat16*)q_in.data_ptr(),
        (const __hip_bfloat16*)k_in.data_ptr(),
        (const __hip_bfloat16*)v_in.data_ptr(),
        (__hip_bfloat16*)q_out.data_ptr(),
        // KV cache
        (__hip_bfloat16*)k_cache.data_ptr(),
        (__hip_bfloat16*)v_cache.data_ptr(),
        (const uint64_t*)cache_position.data_ptr()
      );
    } else {
      TORCH_CHECK(false, "Unsupported dtype for sliding_kvcache_update");
    }

    // restrict to as many tokens as seen by the cache
    auto key_states = k_cache.narrow(/* dim */ 2, /* start */0, /* length */ seen_tokens);
    auto value_states = v_cache.narrow(/* dim */ 2, /* start */ 0, /* length */ seen_tokens);
    return std::make_tuple(q_out, key_states, value_states);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> muillm_complex_rope_forward_sliding_cache(
    torch::Tensor& position_embeddings,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& cache_position,
    uint64_t seen_tokens
) {
  CHECK_INPUT(position_embeddings);
  // q, k, v are expected to not be contiguous
  // (due to the fact that we compute them packed as qkv and transposition afterwards)
  CHECK_CUDA(q_in);
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

  auto cache_sizes = position_embeddings.sizes();

  auto q_sizes = q_in.sizes();
  auto q_strides = q_in.strides();
  auto q_out = torch::empty(q_sizes, output_options);

  auto k_sizes = k_in.sizes();
  auto k_strides = k_in.strides();

  auto v_sizes = v_in.sizes();
  auto v_strides = v_in.strides();

  auto k_cache_sizes = k_cache.sizes();
  auto v_cache_sizes = v_cache.sizes();

  unsigned B = k_sizes[0];
  unsigned T = k_sizes[2];
  unsigned MAX_T = k_cache_sizes[2];
  unsigned num_q_heads = q_sizes[1];
  unsigned num_k_heads = k_sizes[1];
  unsigned num_v_heads = v_sizes[1];
  unsigned embed_dim = k_sizes[3];

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

  muillm_rotary_cache_layout_t cache_layout;

  auto cache_dim = cache_sizes.size();
  if (cache_dim == 2) {
    cache_layout = ROTARY_CACHE_SE_LAYOUT;
  } else if (cache_dim == 3) {
    cache_layout = ROTARY_CACHE_BTE_LAYOUT;
  } else {
    TORCH_CHECK(false, "Unknown rotary cache layout");
  }

  if (position_embeddings.dtype() != torch::kComplexFloat) {
    TORCH_CHECK(false, "position_embeddings must be of type complex64");
  }

  TORCH_CHECK(cache_layout == ROTARY_CACHE_BTE_LAYOUT, "the cache layout must be BTE");

  bool is_full = seen_tokens > MAX_T;

  if (seen_tokens == T) {
    // Prefill (we incremented seen_tokens before updating the cache)

    // k_out is of different size than k_cache (potentially bigger)
    auto k_out = torch::empty(k_sizes, output_options);

    if (dtype == torch::kFloat16) {
      // We can use the sliding_kvcache_prefill kernel
      sliding_kvcache_complex_rope_forward_prefill_fp16(
        stream,
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
        (const float*)position_embeddings.data_ptr(),
        (const half*)q_in.data_ptr(),
        (const half*)k_in.data_ptr(),
        (const half*)v_in.data_ptr(),
        (half*)q_out.data_ptr(),
        (half*)k_out.data_ptr(),
        // KV cache out
        (half*)k_cache.data_ptr(),
        (half*)v_cache.data_ptr()
      );
    } else if (dtype == torch::kBFloat16) {
      // We can use the sliding_kvcache_prefill kernel
      sliding_kvcache_complex_rope_forward_prefill_bf16(
        stream,
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
        (const float*)position_embeddings.data_ptr(),
        (const __hip_bfloat16*)q_in.data_ptr(),
        (const __hip_bfloat16*)k_in.data_ptr(),
        (const __hip_bfloat16*)v_in.data_ptr(),
        (__hip_bfloat16*)q_out.data_ptr(),
        (__hip_bfloat16*)k_out.data_ptr(),
        // KV cache out
        (__hip_bfloat16*)k_cache.data_ptr(),
        (__hip_bfloat16*)v_cache.data_ptr()
      );
    } else {
      TORCH_CHECK(false, "Unsupported dtype for sliding_kvcache_prefill");
    }

    return std::make_tuple(q_out, k_out, v_in);
  } else if (is_full) {

    // previously full or getting full -> overwrite

    // The case T > MAX_T is not supported for overwriting the cache as in that case
    // we would need to also output a k_out which we don't support
    TORCH_CHECK(T <= MAX_T, "T must be <= MAX_T when overwriting the cache");

    // allocate new tensors to store the latest k and v
    auto k_cache_out = torch::empty(k_cache_sizes, output_options);
    auto v_cache_out = torch::empty(v_cache_sizes, output_options);
  
    if (dtype == torch::kFloat16) {
      // We can use the sliding_kvcache_update_overwrite kernel
      sliding_kvcache_complex_rope_forward_update_overwrite_fp16(
        stream,
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
        (const float*)position_embeddings.data_ptr(),
        (const half*)q_in.data_ptr(),
        (const half*)k_in.data_ptr(),
        (const half*)v_in.data_ptr(),
        (half*)q_out.data_ptr(),
        // KV cache in
        (const half*)k_cache.data_ptr(),
        (const half*)v_cache.data_ptr(),
        // KV cache out
        (half*)k_cache_out.data_ptr(),
        (half*)v_cache_out.data_ptr()
      );
    } else if (dtype == torch::kBFloat16) {
      // We can use the sliding_kvcache_update_overwrite kernel
      sliding_kvcache_complex_rope_forward_update_overwrite_bf16(
        stream,
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
        (const float*)position_embeddings.data_ptr(),
        (const __hip_bfloat16*)q_in.data_ptr(),
        (const __hip_bfloat16*)k_in.data_ptr(),
        (const __hip_bfloat16*)v_in.data_ptr(),
        (__hip_bfloat16*)q_out.data_ptr(),
        // KV cache in
        (const __hip_bfloat16*)k_cache.data_ptr(),
        (const __hip_bfloat16*)v_cache.data_ptr(),
        // KV cache out
        (__hip_bfloat16*)k_cache_out.data_ptr(),
        (__hip_bfloat16*)v_cache_out.data_ptr()
      );
    } else {
      TORCH_CHECK(false, "Unsupported dtype for sliding_kvcache_update_overwrite");
    }
  
    // caller needs to use these as new caches
    return std::make_tuple(q_out, k_cache_out, v_cache_out);
  } else {
    // not full, not becoming full -> normal update, similar to a static cache update
    
    if (dtype == torch::kFloat16) {
      // We can use the static kernel
      muillm_apply_complex_rope_forward_fp16_static_cache(
        stream,
        B,
        T,
        MAX_T,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        embed_dim,
        q_in_batch_stride,
        q_in_head_stride,
        q_in_tok_stride,
        k_in_batch_stride,
        k_in_head_stride,
        k_in_tok_stride,
        v_in_batch_stride,
        v_in_head_stride,
        v_in_tok_stride,
        (const float*)position_embeddings.data_ptr(),
        (const half*)q_in.data_ptr(),
        (const half*)k_in.data_ptr(),
        (const half*)v_in.data_ptr(),
        (half*)q_out.data_ptr(),
        // KV cache
        (half*)k_cache.data_ptr(),
        (half*)v_cache.data_ptr(),
        (const uint64_t*)cache_position.data_ptr()
      );
    } else if (dtype == torch::kBFloat16) {
      // We can use the static kernel
      muillm_apply_complex_rope_forward_bf16_static_cache(
        stream,
        B,
        T,
        MAX_T,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        embed_dim,
        q_in_batch_stride,
        q_in_head_stride,
        q_in_tok_stride,
        k_in_batch_stride,
        k_in_head_stride,
        k_in_tok_stride,
        v_in_batch_stride,
        v_in_head_stride,
        v_in_tok_stride,
        (const float*)position_embeddings.data_ptr(),
        (const __hip_bfloat16*)q_in.data_ptr(),
        (const __hip_bfloat16*)k_in.data_ptr(),
        (const __hip_bfloat16*)v_in.data_ptr(),
        (__hip_bfloat16*)q_out.data_ptr(),
        // KV cache
        (__hip_bfloat16*)k_cache.data_ptr(),
        (__hip_bfloat16*)v_cache.data_ptr(),
        (const uint64_t*)cache_position.data_ptr()
      );
    } else {
      TORCH_CHECK(false, "Unsupported dtype for sliding_kvcache_update");
    }

    // restrict to as many tokens as seen by the cache
    auto key_states = k_cache.narrow(/* dim */ 2, /* start */0, /* length */ seen_tokens);
    auto value_states = v_cache.narrow(/* dim */ 2, /* start */ 0, /* length */ seen_tokens);
    return std::make_tuple(q_out, key_states, value_states);
  }
}