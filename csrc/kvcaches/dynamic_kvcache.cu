#include "dynamic_kvcache.hpp"

#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <stdint.h>

typedef uint16_t xx16;

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
);

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
);

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
);


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
  const float* position_embeds, // shape [B, T, embed_dim // 2, 2], dtype complex float
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
);

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
  const float* position_embeds, // shape [B, T, embed_dim // 2, 2], dtype complex float
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
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<at::Tensor, at::Tensor> muillm_dynamic_kvcache_update(
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& prev_k_cache,
    torch::Tensor& prev_v_cache
) {
  // q, k, v are expected to not be contiguous
  // (due to the fact that we compute them packed as qkv and transposition afterwards)
  CHECK_CUDA(k_in);
  CHECK_CUDA(v_in);
  CHECK_INPUT(prev_k_cache);
  CHECK_INPUT(prev_v_cache);

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

  auto prev_k_cache_sizes = prev_k_cache.sizes().vec();
  auto prev_v_cache_sizes = prev_v_cache.sizes().vec();

  unsigned B = k_sizes[0];
  unsigned T = k_sizes[2];
  unsigned PREV_T = prev_k_cache_sizes[2];
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

  // prev k strides
  auto prev_k_strides = prev_k_cache.strides();
  unsigned prev_k_in_batch_stride = prev_k_strides[0];
  unsigned prev_k_in_head_stride = prev_k_strides[1];
  unsigned prev_k_in_tok_stride = prev_k_strides[2];
  // prev v strides
  auto prev_v_strides = prev_v_cache.strides();
  unsigned prev_v_in_batch_stride = prev_v_strides[0];
  unsigned prev_v_in_head_stride = prev_v_strides[1];
  unsigned prev_v_in_tok_stride = prev_v_strides[2];

  auto new_k_cache_sizes = prev_k_cache_sizes;
  new_k_cache_sizes[2] += T;
  auto new_v_cache_sizes = prev_v_cache_sizes;
  new_v_cache_sizes[2] += T;

  auto k_cache_out = torch::empty(new_k_cache_sizes, output_options);
  auto v_cache_out = torch::empty(new_v_cache_sizes, output_options);

  if (dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
    // We can use the static_kvcache_update_xx16 kernel
    dynamic_kvcache_update_xx16(
      stream,
      (const xx16*)k_in.data_ptr(),
      (const xx16*)v_in.data_ptr(),
      (const xx16*)prev_k_cache.data_ptr(),
      (const xx16*)prev_v_cache.data_ptr(),

      // KV cache out
      (xx16*)k_cache_out.data_ptr(),
      (xx16*)v_cache_out.data_ptr(),
      B,
      PREV_T,
      T,
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
      v_in_tok_stride,
      // prev k strides
      prev_k_in_batch_stride,
      prev_k_in_head_stride,
      prev_k_in_tok_stride,
      // prev v strides
      prev_v_in_batch_stride,
      prev_v_in_head_stride,
      prev_v_in_tok_stride
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for dynamic_kvcache_update");
  }

  return std::make_tuple(k_cache_out, v_cache_out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> muillm_rope_forward_dynamic_cache(
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& prev_k_cache,
    torch::Tensor& prev_v_cache
) {
  CHECK_INPUT(cos_cached);
  CHECK_INPUT(sin_cached);
  // q, k, v are expected to not be contiguous
  // (due to the fact that we compute them packed as qkv and transposition afterwards)
  CHECK_CUDA(q_in);
  CHECK_CUDA(k_in);
  CHECK_CUDA(v_in);
  CHECK_INPUT(prev_k_cache);
  CHECK_INPUT(prev_v_cache);


  auto device = q_in.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = q_in.dtype();
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
  // TODO: remove this, it is not needed
  auto k_out = torch::empty(k_sizes, output_options);

  auto v_sizes = v_in.sizes();
  auto v_strides = v_in.strides();

  auto prev_k_cache_sizes = prev_k_cache.sizes().vec();
  auto prev_v_cache_sizes = prev_v_cache.sizes().vec();

  unsigned B = q_sizes[0];
  unsigned T = q_sizes[2];
  unsigned PREV_T = prev_k_cache_sizes[2];
  unsigned num_q_heads = q_sizes[1];
  unsigned num_k_heads = k_sizes[1];
  unsigned num_v_heads = v_sizes[1];
  unsigned embed_dim = q_sizes[3];

  auto new_k_cache_sizes = prev_k_cache_sizes;
  new_k_cache_sizes[2] += T;
  auto new_v_cache_sizes = prev_v_cache_sizes;
  new_v_cache_sizes[2] += T;

  auto k_cache_out = torch::empty(new_k_cache_sizes, output_options);
  auto v_cache_out = torch::empty(new_v_cache_sizes, output_options);

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

  TORCH_CHECK(cache_layout != ROTARY_CACHE_SE_LAYOUT, "the cache layout must be BTE");

  if (dtype == torch::kFloat16) {
    muillm_apply_rope_forward_fp16_dynamic_cache(
      stream,
      B,
      T,
      PREV_T,
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
      (half*)k_out.data_ptr(),
      // KV cache
      (const half*)prev_k_cache.data_ptr(),
      (const half*)prev_v_cache.data_ptr(),
      (half*)k_cache_out.data_ptr(),
      (half*)v_cache_out.data_ptr()
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_apply_rope_forward_bf16_dynamic_cache(
      stream,
      B,
      T,
      PREV_T,
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
      (__hip_bfloat16*)k_out.data_ptr(),
       // KV cache
       (const __hip_bfloat16*)prev_k_cache.data_ptr(),
       (const __hip_bfloat16*)prev_v_cache.data_ptr(),
       (__hip_bfloat16*)k_cache_out.data_ptr(),
       (__hip_bfloat16*)v_cache_out.data_ptr()
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for rotary embedding dynamic cache");
  }
  return std::make_tuple(q_out, k_cache_out, v_cache_out);
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> muillm_complex_rope_forward_dynamic_cache(
    torch::Tensor& position_embeds,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& prev_k_cache,
    torch::Tensor& prev_v_cache
) {
  CHECK_INPUT(position_embeds);
  // q, k, v are expected to not be contiguous
  // (due to the fact that we compute them packed as qkv and transposition afterwards)
  CHECK_CUDA(q_in);
  CHECK_CUDA(k_in);
  CHECK_CUDA(v_in);
  CHECK_INPUT(prev_k_cache);
  CHECK_INPUT(prev_v_cache);


  auto device = q_in.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = q_in.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto cache_sizes = position_embeds.sizes();

  auto q_sizes = q_in.sizes();
  auto q_strides = q_in.strides();
  auto q_out = torch::empty(q_sizes, output_options);

  auto k_sizes = k_in.sizes();
  auto k_strides = k_in.strides();
  // TODO: remove this, it is not needed
  auto k_out = torch::empty(k_sizes, output_options);

  auto v_sizes = v_in.sizes();
  auto v_strides = v_in.strides();

  auto prev_k_cache_sizes = prev_k_cache.sizes().vec();
  auto prev_v_cache_sizes = prev_v_cache.sizes().vec();

  unsigned B = q_sizes[0];
  unsigned T = q_sizes[2];
  unsigned PREV_T = prev_k_cache_sizes[2];
  unsigned num_q_heads = q_sizes[1];
  unsigned num_k_heads = k_sizes[1];
  unsigned num_v_heads = v_sizes[1];
  unsigned embed_dim = q_sizes[3];

  auto new_k_cache_sizes = prev_k_cache_sizes;
  new_k_cache_sizes[2] += T;
  auto new_v_cache_sizes = prev_v_cache_sizes;
  new_v_cache_sizes[2] += T;

  auto k_cache_out = torch::empty(new_k_cache_sizes, output_options);
  auto v_cache_out = torch::empty(new_v_cache_sizes, output_options);

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

  if (position_embeds.dtype() != torch::kComplexFloat) {
    TORCH_CHECK(false, "position_embeds must be of type complex64");
  }

  TORCH_CHECK(cache_layout == ROTARY_CACHE_BTE_LAYOUT, "The cache layout must be BTE");

  if (dtype == torch::kFloat16) {
    muillm_apply_complex_rope_forward_fp16_dynamic_cache(
      stream,
      B,
      T,
      PREV_T,
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
      (const float*)position_embeds.data_ptr(),
      (const half*)q_in.data_ptr(),
      (const half*)k_in.data_ptr(),
      (const half*)v_in.data_ptr(),
      (half*)q_out.data_ptr(),
      (half*)k_out.data_ptr(),
      // KV cache
      (const half*)prev_k_cache.data_ptr(),
      (const half*)prev_v_cache.data_ptr(),
      (half*)k_cache_out.data_ptr(),
      (half*)v_cache_out.data_ptr()
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_apply_complex_rope_forward_bf16_dynamic_cache(
      stream,
      B,
      T,
      PREV_T,
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
      (const float*)position_embeds.data_ptr(),
      (const __hip_bfloat16*)q_in.data_ptr(),
      (const __hip_bfloat16*)k_in.data_ptr(),
      (const __hip_bfloat16*)v_in.data_ptr(),
      (__hip_bfloat16*)q_out.data_ptr(),
      (__hip_bfloat16*)k_out.data_ptr(),
       // KV cache
       (const __hip_bfloat16*)prev_k_cache.data_ptr(),
       (const __hip_bfloat16*)prev_v_cache.data_ptr(),
       (__hip_bfloat16*)k_cache_out.data_ptr(),
       (__hip_bfloat16*)v_cache_out.data_ptr()
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for complex rotary embedding dynamic cache");
  }

  return std::make_tuple(q_out, k_cache_out, v_cache_out);
}