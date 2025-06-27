#include "flash_decoding.cuh"

#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <stdint.h>
#include <vector>
#include <algorithm>
#include <cmath>

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / ((b)))

#define MIN_TOKENS_PER_GROUP 4

void flash_decoding_partially_aggregate_fp16(
  hipStream_t stream,
  const half* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const half* __restrict__ k_in, // shape [B, num_k_heads, S, embed_dim]
  const half* __restrict__ v_in, // shape [B, num_v_heads, S, embed_dim]
  const half* __restrict__ m_in, // shape [B, 1, T, S]
  float* __restrict__ partial_vectors, // [B, num_q_heads, G, T, embed_dim]
  half* __restrict__ partial_softmax_denoms, // [B, num_q_heads, G, T]
  half* __restrict__ partial_maxes, // [B, num_q_heads, G, T]
  // tensor dimension sizes
  unsigned B,
  unsigned G,
  unsigned T, // num tokens to decode
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_kv_heads,
  float attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
  // kv strides
  unsigned kv_batch_stride,
  unsigned kv_head_stride,
  unsigned kv_tok_stride
);

void flash_decoding_final_reduce_fp16(
  hipStream_t stream,
  const float* __restrict__ partial_vectors, // shape [B, num_q_heads, G, T, embed_dim]
  const half* __restrict__ partial_softmax_denoms, // shape [B, num_q_heads, G, T]
  const half* __restrict__ partial_maxes, // [B, num_q_heads, G, T]
  half* __restrict__ hidden_out, // [B, num_q_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned B,
  unsigned G, // number of groups
  unsigned T, // num tokens to decode
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim
);


void flash_decoding_partially_aggregate_bf16(
  hipStream_t stream,
  const __hip_bfloat16* __restrict__ q_in, // shape [B, num_q_heads, T, embed_dim]
  const __hip_bfloat16* __restrict__ k_in, // shape [B, num_k_heads, S, embed_dim]
  const __hip_bfloat16* __restrict__ v_in, // shape [B, num_v_heads, S, embed_dim]
  const __hip_bfloat16* __restrict__ m_in, // shape [B, 1, T, S]
  float* __restrict__ partial_vectors, // [B, num_q_heads, G, T, embed_dim]
  __hip_bfloat16* __restrict__ partial_softmax_denoms, // [B, num_q_heads, G, T]
  __hip_bfloat16* __restrict__ partial_maxes, // [B, num_q_heads, G, T]
  // tensor dimension sizes
  unsigned B,
  unsigned G,
  unsigned T, // num tokens to decode
  unsigned S, // number of total tokens
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim,
  unsigned q_to_kv_heads,
  float attention_inv_scale, // factor to scale the attention scores, typically 1/sqrt(embed_dim)
  // kv strides
  unsigned kv_batch_stride,
  unsigned kv_head_stride,
  unsigned kv_tok_stride
);

void flash_decoding_final_reduce_bf16(
  hipStream_t stream,
  const float* __restrict__ partial_vectors, // shape [B, num_q_heads, G, T, embed_dim]
  const __hip_bfloat16* __restrict__ partial_softmax_denoms, // shape [B, num_q_heads, G, T]
  const __hip_bfloat16* __restrict__ partial_maxes, // [B, num_q_heads, G, T]
  __hip_bfloat16* __restrict__ hidden_out, // [B, num_q_heads, T, embed_dim]
  // tensor dimension sizes
  unsigned B,
  unsigned G, // number of groups
  unsigned T, // num tokens to decode
  unsigned num_q_heads, // number of heads for q
  unsigned embed_dim
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// returns partial vectors, partial softmax denoms, partial maxes
static inline std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_decoding_partially_aggregate(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v, // [B, num_v_heads, S, embed_dim]
  torch::Tensor& m // [B, 1, S, T]
) {
  // q is expected to be contiguous
  CHECK_INPUT(q);
  // but k and v might not be due to the static cache being
  // allocated for longer sequences
  // it is fine as long as the innermost stride is 1
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  // m must be contiguous
  if (m.defined()) {
    CHECK_INPUT(m);
  }

  auto device = q.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = q.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same device as inputs
                            .requires_grad(false);
  

  auto temp_options_f32 = at::TensorOptions()
                            .dtype(torch::kFloat32)
                            .layout(at::kStrided)
                            .device(device) // same device as inputs
                            .requires_grad(false);

  auto q_sizes = q.sizes().vec();
  auto k_sizes = k.sizes().vec();
  auto k_strides = k.strides().vec();
  auto v_strides = v.strides().vec();

  TORCH_CHECK(k_strides[3] == 1, "k innermost stride must be 1");
  TORCH_CHECK(v_strides[3] == 1, "v innermost stride must be 1");

  unsigned B = q_sizes[0];
  unsigned num_q_heads = q_sizes[1];
  unsigned T = q_sizes[2]; // always 1 for now
  unsigned embed_dim = q_sizes[3];
  unsigned S = k_sizes[2];
  unsigned num_k_heads = k_sizes[1];

  TORCH_CHECK(T == 1, "T > 1 not supported");

  TORCH_CHECK((k_strides[0] == v_strides[0]) && (k_strides[1] == v_strides[1]) && (k_strides[2] == v_strides[2]), "k and v strides must match");

  // TODO: adapt num_groups based on number of compute units
  unsigned G = DIV_ROUND_UP(S, MIN_TOKENS_PER_GROUP);

  auto partial_vectors = torch::empty({B, num_q_heads, G, T, embed_dim}, temp_options_f32);
  auto partial_softmax_denoms = torch::empty({B, num_q_heads, G, T}, output_options);
  auto partial_maxes = torch::empty({B, num_q_heads, G, T}, output_options);

  // kv strides
  unsigned kv_batch_stride = k_strides[0];
  unsigned kv_head_stride = k_strides[1];
  unsigned kv_tok_stride = k_strides[2];

  // to compute what k/v head to target for a given q head
  unsigned q_to_kv_heads = num_q_heads / num_k_heads;

  float attention_inv_scale = 1.0f / sqrtf(embed_dim);

  if (dtype == torch::kFloat16) {
    flash_decoding_partially_aggregate_fp16(
      stream,
      (const half*) q.data_ptr(), // shape [B, num_q_heads, T, embed_dim]
      (const half*) k.data_ptr(), // shape [B, num_k_heads, S, embed_dim]
      (const half*) v.data_ptr(), // shape [B, num_v_heads, S, embed_dim]
      m.defined() ? (const half*) m.data_ptr() : nullptr, // shape [B, 1, S, T]
      (float*) partial_vectors.data_ptr(), // [B, num_q_heads, G, T, embed_dim]
      (half*) partial_softmax_denoms.data_ptr(), // [B, num_q_heads, G, T]
      (half*) partial_maxes.data_ptr(), // [B, num_q_heads, G, T]
      B,
      G,
      T,
      S,
      num_q_heads,
      embed_dim,
      q_to_kv_heads,
      attention_inv_scale,
      kv_batch_stride,
      kv_head_stride,
      kv_tok_stride
    );
  } else if (dtype == torch::kBFloat16) {
    flash_decoding_partially_aggregate_bf16(
      stream,
      (__hip_bfloat16*) q.data_ptr(), // shape [B, num_q_heads, T, embed_dim]
      (__hip_bfloat16*) k.data_ptr(), // shape [B, num_k_heads, S, embed_dim]
      (__hip_bfloat16*) v.data_ptr(), // shape [B, num_v_heads, S, embed_dim]
      m.defined() ? (__hip_bfloat16*) m.data_ptr() : nullptr, // shape [B, 1, S, T]
      (float*) partial_vectors.data_ptr(), // [B, num_q_heads, G, T, embed_dim]
      (__hip_bfloat16*) partial_softmax_denoms.data_ptr(), // [B, num_q_heads, G, T]
      (__hip_bfloat16*) partial_maxes.data_ptr(), // [B, num_q_heads, G, T]
      B,
      G,
      T,
      S,
      num_q_heads,
      embed_dim,
      q_to_kv_heads,
      attention_inv_scale,
      kv_batch_stride,
      kv_head_stride,
      kv_tok_stride
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for flash decoding");
  }

  return std::make_tuple(partial_vectors, partial_softmax_denoms, partial_maxes);
}

// returns transformed vectors
static inline at::Tensor flash_decoding_final_reduce(
  torch::Tensor& partial_vectors, // shape [B, num_q_heads, G, T, embed_dim]
  torch::Tensor& partial_softmax_denoms, // shape [B, num_q_heads, G, T]
  torch::Tensor& partial_maxes  // [B, num_q_heads, G, T]
) {
  CHECK_INPUT(partial_vectors);
  CHECK_INPUT(partial_softmax_denoms);
  CHECK_INPUT(partial_maxes);

  auto device = partial_vectors.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto dtype = partial_softmax_denoms.dtype();

  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto partial_vectors_sizes = partial_vectors.sizes().vec();

  unsigned B = partial_vectors_sizes[0];
  unsigned num_q_heads = partial_vectors_sizes[1];
  unsigned G = partial_vectors_sizes[2];
  unsigned T = partial_vectors_sizes[3];
  unsigned embed_dim = partial_vectors_sizes[4];

  TORCH_CHECK(T == 1, "T > 1 not supported");

  // q_len is 1, so we can avoid the transposition
  auto hidden_out = torch::empty({B, T, num_q_heads * embed_dim}, output_options);

  if (dtype == torch::kFloat16) {
    flash_decoding_final_reduce_fp16(
      stream,
      (const float*) partial_vectors.data_ptr(), // shape [B, num_q_heads, G, T, embed_dim]
      (const half*) partial_softmax_denoms.data_ptr(), // shape [B, num_q_heads, G, T]
      (const half*) partial_maxes.data_ptr(), // [B, num_q_heads, G, T]
      (half*) hidden_out.data_ptr(), // [B, num_q_heads, T, embed_dim]
      B,
      G,
      T,
      num_q_heads,
      embed_dim
    );
  } else if (dtype == torch::kBFloat16) {
    flash_decoding_final_reduce_bf16(
      stream,
      (const float*) partial_vectors.data_ptr(), // shape [B, num_q_heads, G, T, embed_dim]
      (const __hip_bfloat16*) partial_softmax_denoms.data_ptr(), // shape [B, num_q_heads, G, T]
      (const __hip_bfloat16*) partial_maxes.data_ptr(), // [B, num_q_heads, G, T]
      (__hip_bfloat16*) hidden_out.data_ptr(), // [B, num_q_heads, T, embed_dim]
      B,
      G,
      T,
      num_q_heads,
      embed_dim
    );
  } else {
    TORCH_CHECK(false, "Unsupported dtype for flash decoding final reduce");
  }

  return hidden_out;
}

at::Tensor flash_decode_no_mask(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v  // [B, num_v_heads, S, embed_dim]
) {
  auto undef_tensor = torch::Tensor();

  auto partial_aggregation_results = flash_decoding_partially_aggregate(q, k, v, /*m*/ undef_tensor);

  auto partial_vectors = std::get<0>(partial_aggregation_results);
  auto partial_softmax_denoms = std::get<1>(partial_aggregation_results);
  auto partial_maxes = std::get<2>(partial_aggregation_results);

  return flash_decoding_final_reduce(
    partial_vectors,
    partial_softmax_denoms,
    partial_maxes
  );
}

at::Tensor flash_decode_masked(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v,  // [B, num_v_heads, S, embed_dim]
  torch::Tensor& m  // [B, 1, S, T]
) {
  auto partial_aggregation_results = flash_decoding_partially_aggregate(q, k, v, m);

  auto partial_vectors = std::get<0>(partial_aggregation_results);
  auto partial_softmax_denoms = std::get<1>(partial_aggregation_results);
  auto partial_maxes = std::get<2>(partial_aggregation_results);

  return flash_decoding_final_reduce(
    partial_vectors,
    partial_softmax_denoms,
    partial_maxes
  );
}