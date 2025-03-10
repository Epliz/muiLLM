#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>


#include <stdint.h>
#include <vector>
#include <algorithm>
#include <cmath>

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / ((b)))

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor muillm_causal_transformer_compute_softmax_scores_no_mask(
    torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
    torch::Tensor& k // [B, num_k_heads, S, embed_dim]
) {
  // q is expected to be contiguous
  CHECK_INPUT(q);
  // but k might not be due to the static cache being
  // allocated for longer sequences
  // it is fine as long as the innermost stride is 1
  CHECK_CUDA(k);

  auto device = q.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto output_options_f16 = at::TensorOptions()
                            .dtype(torch::kFloat16)
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

  TORCH_CHECK(k_strides[3] == 1, "k innermost stride must be 1");

  unsigned B = q_sizes[0];
  unsigned T = q_sizes[2];
  unsigned S = k_sizes[2];
  unsigned num_q_heads = q_sizes[1];
  unsigned num_k_heads = k_sizes[1];
  unsigned embed_dim = q_sizes[3];

  // k strides
  unsigned k_batch_stride = k_strides[0];
  unsigned k_head_stride = k_strides[1];
  unsigned k_tok_stride = k_strides[2];

  // output scores have size [B, num_q_heads, T, S]
  auto scores_sizes = q.sizes().vec();
  scores_sizes[3] = S;

  // to compute what k/v head to target for a given q head
  unsigned q_to_k_heads = num_q_heads / num_k_heads;

  float attention_inv_scale = 1.0f / sqrtf(embed_dim);

  auto temp_scores_f32 = torch::empty(scores_sizes, temp_options_f32);
  auto attention_scores = torch::empty(scores_sizes, output_options_f16);

  causally_compute_transformer_softmax_scores(
    stream,
    (const half*)q.data_ptr(),
    (const half*)k.data_ptr(),
    (float*)temp_scores_f32.data_ptr(),
    (half*)attention_scores.data_ptr(),
    // tensor dimension sizes
    B,
    T,
    S,
    num_q_heads,
    embed_dim,
    q_to_k_heads,
    attention_inv_scale,
    k_batch_stride,
    k_head_stride,
    k_tok_stride
  );

  return attention_scores;
}

at::Tensor muillm_causal_transformer_compute_softmax_scores_masked(
    torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
    torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
    torch::Tensor& m // [B, 1, S, T]
) {
  // q is expected to be contiguous
  CHECK_INPUT(q);
  // but k might not be due to the static cache being
  // allocated for longer sequences
  // it is fine as long as the innermost stride is 1
  CHECK_CUDA(k);
  // m must be contiguous
  CHECK_INPUT(m);

  auto device = q.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto output_options_f16 = at::TensorOptions()
                            .dtype(torch::kFloat16)
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

  TORCH_CHECK(k_strides[3] == 1, "k innermost stride must be 1");

  unsigned B = q_sizes[0];
  unsigned T = q_sizes[2]; // always 1 for now
  unsigned S = k_sizes[2];
  unsigned num_q_heads = q_sizes[1];
  unsigned num_k_heads = k_sizes[1];
  unsigned embed_dim = q_sizes[3];

  // k strides
  unsigned k_batch_stride = k_strides[0];
  unsigned k_head_stride = k_strides[1];
  unsigned k_tok_stride = k_strides[2];

  // output scores have size [B, num_q_heads, T, S]
  auto scores_sizes = q.sizes().vec();
  scores_sizes[3] = S;

  // to compute what k/v head to target for a given q head
  unsigned q_to_k_heads = num_q_heads / num_k_heads;

  float attention_inv_scale = 1.0f / sqrtf(embed_dim);

  auto temp_scores_f32 = torch::empty(scores_sizes, temp_options_f32);
  auto attention_scores = torch::empty(scores_sizes, output_options_f16);

  causally_compute_transformer_softmax_scores_masked(
    stream,
    (const half*)q.data_ptr(),
    (const half*)k.data_ptr(),
    (const half*)m.data_ptr(),
    (float*)temp_scores_f32.data_ptr(),
    (half*)attention_scores.data_ptr(),
    // tensor dimension sizes
    B,
    T,
    S,
    num_q_heads,
    embed_dim,
    q_to_k_heads,
    attention_inv_scale,
    k_batch_stride,
    k_head_stride,
    k_tok_stride
  );

  return attention_scores;
}


at::Tensor muillm_causal_transformer_apply_softmax_scores(
  torch::Tensor& attention_weights, // [B, num_q_heads, T, S]
  torch::Tensor& v // [B, num_v_heads, S, embed_dim]
) {
  // attention weights must be contiguous
  CHECK_INPUT(attention_weights);
  // but v might not be due to the static cache
  // it is fine as long as the innermost stride is 1
  CHECK_CUDA(v);

  auto device = v.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  auto output_options_f16 = at::TensorOptions()
                            .dtype(torch::kFloat16)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto attention_weights_sizes = attention_weights.sizes().vec();
  auto v_sizes = v.sizes().vec();
  auto v_strides = v.strides().vec();

  TORCH_CHECK(v_strides[3] == 1, "v innermost stride must be 1");

  unsigned B = attention_weights_sizes[0];
  unsigned T = attention_weights_sizes[2];
  unsigned S = attention_weights_sizes[3];
  unsigned num_q_heads = attention_weights_sizes[1];
  unsigned num_v_heads = v_sizes[1];
  unsigned embed_dim = v_sizes[3];

  // v strides
  unsigned v_batch_stride = v_strides[0];
  unsigned v_head_stride = v_strides[1];
  unsigned v_tok_stride = v_strides[2];

  // to compute what k/v head to target for a given q head
  unsigned q_to_v_heads = num_q_heads / num_v_heads;

  // q_len is 1, so we can avoid the transposition
  auto hidden_out = torch::empty({B, T, num_q_heads * embed_dim}, output_options_f16);

  causally_apply_transformer_softmax_scores(
    sstream,
    (const half*)attention_weights.data_ptr(),
    (const half*)v.data_ptr(),
    (half*)hidden_out.data_ptr(),
    // tensor dimension sizes
    B,
    S,
    num_q_heads,
    embed_dim,
    q_to_v_heads,
    v_batch_stride,
    v_head_stride,
    v_tok_stride
  );

  return hidden_out;
}

at::Tensor muillm_causal_transformer_decoding_no_mask(
  torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
  torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
  torch::Tensor& v  // [B, num_v_heads, S, embed_dim]
) {
  auto attention_scores = muillm_causal_transformer_compute_softmax_scores_no_mask(
    q,
    k
  );

  return muillm_causal_transformer_apply_softmax_scores(
    attention_scores,
    v
  );
}

at::Tensor muillm_causal_transformer_decoding_masked(
    torch::Tensor& q, // [B, num_q_heads, T, embed_dim]
    torch::Tensor& k, // [B, num_k_heads, S, embed_dim]
    torch::Tensor& v,  // [B, num_v_heads, S, embed_dim]
    torch::Tensor& m  // [B, 1, S, T]
) {
  auto attention_scores = muillm_causal_transformer_compute_softmax_scores_masked(
    q,
    k,
    m
  );

  return muillm_causal_transformer_apply_softmax_scores(
    attention_scores,
    v
  );
}