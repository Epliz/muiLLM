#include "rotary.h"

#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <cuComplex.h>

#include <algorithm>

#include <iostream>


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


void muillm_compute_rotary_embed_positions_fp16(
  hipStream_t stream,
  unsigned B,
  unsigned T,
  unsigned E,
  const uint64_t* position_ids,
  const half* cos_cached,
  const half* sin_cached,
  half* embeds_cos,
  half* embeds_sin
);

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
);

std::tuple<at::Tensor, at::Tensor> muillm_compute_rotary_embed_positions(
    torch::Tensor& x,
    torch::Tensor& position_ids, // shape [batch_size, seq_len]
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached
) {
  CHECK_INPUT(x);
  CHECK_INPUT(position_ids);
  CHECK_INPUT(cos_cached);
  CHECK_INPUT(sin_cached);

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  // check that position_ids is 2D
  if (position_ids.dim() != 2) {
    TORCH_CHECK(false, "position_ids must be a 2D tensor");
  }
  // check that cos_cached and sin_cached are 2D tensors
  if (cos_cached.dim() != 2 || sin_cached.dim() != 2) {
    TORCH_CHECK(false, "cos_cached and sin_cached must be 2D tensors");
  }

  unsigned B = position_ids.size(0);
  unsigned T = position_ids.size(1);
  unsigned E = cos_cached.size(1);

  auto dtype = x.dtype();

  // check that cos_cached and sin_cached have the same dtype as x
  if (cos_cached.dtype() != dtype || sin_cached.dtype() != dtype) {
    TORCH_CHECK(false, "cos_cached and sin_cached must have the same dtype as x");
  }

  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // check the dtype of position_ids
  if (position_ids.dtype() != torch::kLong) {
    TORCH_CHECK(false, "position_ids must be of type Long");
  }

  auto embed_cos = torch::empty({B, T, E}, output_options);
  auto embed_sin = torch::empty({B, T, E}, output_options);

  if (dtype == torch::kFloat16) {
    muillm_compute_rotary_embed_positions_fp16(
      stream,
      B,
      T,
      E,
      (const uint64_t*)position_ids.data_ptr(),
      (const half*)cos_cached.data_ptr(),
      (const half*)sin_cached.data_ptr(),
      (half*)embed_cos.data_ptr(),
      (half*)embed_sin.data_ptr()
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_compute_rotary_embed_positions_bf16(
      stream,
      B,
      T,
      E,
      (const uint64_t*)position_ids.data_ptr(),
      (const __hip_bfloat16*)cos_cached.data_ptr(),
      (const __hip_bfloat16*)sin_cached.data_ptr(),
      (__hip_bfloat16*)embed_cos.data_ptr(),
      (__hip_bfloat16*)embed_sin.data_ptr()
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype");
  }

  return std::make_tuple(cos_cached, sin_cached);
}

void muillm_apply_rope_forward_fp16_no_cache(
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
  const half* cos_cached,
  const half* sin_cached,
  const half* q_in,
  const half* k_in,
  half* q_out,
  half* k_out
);

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
);


std::tuple<at::Tensor, at::Tensor> muillm_rope_forward_no_cache(
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

  auto device = q_in.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = q_in.dtype();

  // check that the dtype of cos_cached and sin_cached matches q_in
  if (cos_cached.dtype() != dtype || sin_cached.dtype() != dtype) {
    TORCH_CHECK(false, "cos_cached and sin_cached must have the same dtype as q_in and k_in");
  }

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

  muillm_rotary_cache_layout_t cache_layout;

  auto cache_dim = cache_sizes.size();
  if (cache_dim == 2) {
    cache_layout = ROTARY_CACHE_SE_LAYOUT;
  } else if (cache_dim == 3) {
    cache_layout = ROTARY_CACHE_BTE_LAYOUT;
  } else {
    TORCH_CHECK(false, "Unknown rotary cache layout");
  }

  if (dtype == torch::kFloat16) {
    muillm_apply_rope_forward_fp16_no_cache(
      stream,
      B,
      T,
      num_q_heads,
      num_k_heads,
      embed_dim,
      q_in_batch_stride,
      q_in_head_stride,
      q_in_tok_stride,
      k_in_batch_stride,
      k_in_head_stride,
      k_in_tok_stride,
      cache_layout,
      (const uint64_t*)position_ids.data_ptr(),
      (const half*)cos_cached.data_ptr(),
      (const half*)sin_cached.data_ptr(),
      (const half*)q_in.data_ptr(),
      (const half*)k_in.data_ptr(),
      (half*)q_out.data_ptr(),
      (half*)k_out.data_ptr()
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_apply_rope_forward_bf16_no_cache(
      stream,
      B,
      T,
      num_q_heads,
      num_k_heads,
      embed_dim,
      q_in_batch_stride,
      q_in_head_stride,
      q_in_tok_stride,
      k_in_batch_stride,
      k_in_head_stride,
      k_in_tok_stride,
      cache_layout,
      (const uint64_t*)position_ids.data_ptr(),
      (const __hip_bfloat16*)cos_cached.data_ptr(),
      (const __hip_bfloat16*)sin_cached.data_ptr(),
      (const __hip_bfloat16*)q_in.data_ptr(),
      (const __hip_bfloat16*)k_in.data_ptr(),
      (__hip_bfloat16*)q_out.data_ptr(),
      (__hip_bfloat16*)k_out.data_ptr()
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for rotary embedding no cache");
  }

  return std::make_tuple(q_out, k_out);
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
  muillm_rotary_cache_layout_t cache_layout,
  const uint64_t* position_ids, // shape [B, T]
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
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> muillm_rope_forward_dynamic_cache(
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


  auto device = q_in.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = q_in.dtype();
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
  // TODO: remove this, it is not needed
  auto k_out = torch::empty(k_sizes, output_options);

  auto v_sizes = v_in.sizes().vec();
  auto v_strides = v_in.strides().vec();

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

  muillm_rotary_cache_layout_t cache_layout;

  auto cache_dim = cache_sizes.size();
  if (cache_dim == 2) {
    cache_layout = ROTARY_CACHE_SE_LAYOUT;
  } else if (cache_dim == 3) {
    cache_layout = ROTARY_CACHE_BTE_LAYOUT;
  } else {
    TORCH_CHECK(false, "Unknown rotary cache layout");
  }

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
      cache_layout,
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
      cache_layout,
      (const uint64_t*)position_ids.data_ptr(),
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
  muillm_rotary_cache_layout_t cache_layout,
  const uint64_t* position_ids, // shape [B, T]
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
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> muillm_rope_forward_static_cache(
    torch::Tensor& position_ids,
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
  CHECK_INPUT(position_ids);
  CHECK_INPUT(cos_cached);
  CHECK_INPUT(sin_cached);
  // q, k, v are expected to not be contiguous
  // (due to the fact that we compute them packed as qkv and transposition afterwards)
  CHECK_CUDA(q_in);
  CHECK_CUDA(k_in);
  CHECK_CUDA(v_in);
  CHECK_INPUT(k_cache);
  CHECK_INPUT(v_cache);

  auto device = q_in.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = q_in.dtype();
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

  auto v_sizes = v_in.sizes().vec();
  auto v_strides = v_in.strides().vec();

  auto k_cache_sizes = k_cache.sizes().vec();
  auto v_cache_sizes = v_cache.sizes().vec();

  unsigned B = q_sizes[0];
  unsigned T = q_sizes[2];
  unsigned MAX_T = k_cache_sizes[2];
  unsigned num_q_heads = q_sizes[1];
  unsigned num_k_heads = k_sizes[1];
  unsigned num_v_heads = v_sizes[1];
  unsigned embed_dim = q_sizes[3];

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

  if (dtype == torch::kFloat16) {
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
      cache_layout,
      (const uint64_t*)position_ids.data_ptr(),
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
      cache_layout,
       (const uint64_t*)position_ids.data_ptr(),
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
    TORCH_CHECK(false, "Unsupported dtype for rotary embedding static cache");
  }

  // restrict to as many tokens as seen by the cache
  auto key_states = k_cache.narrow(/* dim */ 2, /* start */0, /* length */ seen_tokens);
  auto value_states = v_cache.narrow(/* dim */ 2, /* start */ 0, /* length */ seen_tokens);

  return std::make_tuple(q_out, key_states, value_states);
}

void muillm_apply_complex_rope_forward_fp16_no_cache(
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
  const half* q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* k_in, // shape [B, num_k_heads, T, embed_dim]
  half* q_out, // shape [B, num_q_heads, T, embed_dim]
  half* k_out // shape [B, num_k_heads, T, embed_dim]
);

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
);

std::tuple<at::Tensor, at::Tensor> muillm_complex_rope_forward_no_cache(
  torch::Tensor& q_in, // shape [B, num_q_heads, T, embed_dim]
  torch::Tensor& k_in, // shape [B, num_k_heads, T, embed_dim]
  torch::Tensor& position_embeds // shape [B, T, embed_dim / 2, (2)]
) {
  CHECK_INPUT(position_embeds);
  // q, k are expected to not be contiguous
  // (due to the fact that we compute them packed as qkv and transposition afterwards)
  CHECK_CUDA(q_in);
  CHECK_CUDA(k_in);

  auto device = q_in.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = q_in.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto cache_sizes = position_embeds.sizes().vec();

  auto q_sizes = q_in.sizes().vec();
  auto q_strides = q_in.strides().vec();
  auto q_out = torch::empty(q_sizes, output_options);

  auto k_sizes = k_in.sizes().vec();
  auto k_strides = k_in.strides().vec();
  auto k_out = torch::empty(k_sizes, output_options);

  unsigned B = q_sizes[0];
  unsigned T = q_sizes[2];
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

  muillm_rotary_cache_layout_t cache_layout;

  auto cache_dim = cache_sizes.size();
  if (cache_dim == 2) {
    cache_layout = ROTARY_CACHE_SE_LAYOUT;
  } else if (cache_dim == 3) {
    cache_layout = ROTARY_CACHE_BTE_LAYOUT;
  } else {
    TORCH_CHECK(false, "Unknown rotary cache layout");
  }

  if (cache_layout != ROTARY_CACHE_BTE_LAYOUT) {
    // we only support complex ROPE with BTE layout
    TORCH_CHECK(false, "Complex ROPE only supported with BTE layout");
  }

  if (dtype == torch::kFloat16) {
    muillm_apply_complex_rope_forward_fp16_no_cache(
      stream,
      B,
      T,
      num_q_heads,
      num_k_heads,
      embed_dim,
      q_in_batch_stride,
      q_in_head_stride,
      q_in_tok_stride,
      k_in_batch_stride,
      k_in_head_stride,
      k_in_tok_stride,
      (const float*)position_embeds.data_ptr(),
      (const half*)q_in.data_ptr(),
      (const half*)k_in.data_ptr(),
      (half*)q_out.data_ptr(),
      (half*)k_out.data_ptr()
    );
  } else if (dtype == torch::kBFloat16) {
    muillm_apply_complex_rope_forward_bf16_no_cache(
      stream,
      B,
      T,
      num_q_heads,
      num_k_heads,
      embed_dim,
      q_in_batch_stride,
      q_in_head_stride,
      q_in_tok_stride,
      k_in_batch_stride,
      k_in_head_stride,
      k_in_tok_stride,
      (const float*)position_embeds.data_ptr(),
      (const __hip_bfloat16*)q_in.data_ptr(),
      (const __hip_bfloat16*)k_in.data_ptr(),
      (__hip_bfloat16*)q_out.data_ptr(),
      (__hip_bfloat16*)k_out.data_ptr()
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for complex rotary embedding no cache");
  }

  return std::make_tuple(q_out, k_out);
}