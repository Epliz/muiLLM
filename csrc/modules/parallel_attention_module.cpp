#include "parallel_attention_module.h"
#include "../attention/causal_transformer_decoding.cuh"

//
// Actual code
//

MuiLLMParallelAttention::MuiLLMParallelAttention(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  MuillmRotaryEmbedding* rotary,
  MuiLLMParallelLinear* o_proj,
  int num_tp_heads,
  int num_tp_key_value_heads,
  int head_dim
) {
  this->engine = engine;
  this->comm = comm;
  this->rotary = rotary;
  this->o_proj = o_proj;

  this->num_tp_heads = num_tp_heads;
  this->num_tp_key_value_heads = num_tp_key_value_heads;
  this->head_dim = head_dim;
}

torch::Tensor MuiLLMParallelAttention::forward(
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  torch::Tensor& m,
  torch::Tensor& residual
) {
  bool masked = m.defined();
  auto attn_output = masked ?
      muillm_causal_transformer_decoding_masked(q, k, v, m)
    : muillm_causal_transformer_decoding_no_mask(q, k, v);


  // o proj
  auto proj_attn_output = this->o_proj->forward(attn_output, residual, /*collect_outputs*/ true);
  return proj_attn_output;
}

torch::Tensor MuiLLMParallelAttention::rope_forward(
  MuillmKVCache* cache,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  torch::Tensor& m,
  torch::Tensor& residual,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
) {

  int bsz = q.size(0);
  int q_len = q.size(1);

  torch::Tensor q_res;
  torch::Tensor k_res;
  torch::Tensor v_res;

  if (q_len == 1) {
    // as q_len is 1, we can avoid the transpose
    q_res = q.view({bsz, this->num_tp_heads, q_len, this->head_dim});

    k_res = k.view({bsz, this->num_tp_key_value_heads, q_len, this->head_dim});
    v_res = v.view({bsz, this->num_tp_key_value_heads, q_len, this->head_dim});
  } else {
    q_res = q.view({bsz, q_len, this->num_tp_heads, this->head_dim}).transpose(1, 2);

    k_res = k.view({bsz, q_len, this->num_tp_key_value_heads, this->head_dim}).transpose(1, 2);
    v_res = v.view({bsz, q_len, this->num_tp_key_value_heads, this->head_dim}).transpose(1, 2);
  }

  // rotary
  auto [q_rot, k_rot, v_rot] = this->rotary->forward(
    cache,
    q_res,
    k_res,
    v_res,
    position_ids,
    cos_sin,
    cache_positions
  );

  // attention
  return this->forward(q_rot, k_rot, v_rot, m, residual);
}

//
// Python trampolines
//

// init
muillm_parallel_attention_module_ptr muillm_parallel_attention_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_rotary_embedding_module_ptr_t rotary,
  muillm_parallel_linear_module_ptr_t o_proj,
  int num_tp_heads,
  int num_tp_key_value_heads,
  int head_dim
) {

  MuiLLMParallelAttention* m = new MuiLLMParallelAttention(
    engine.engine_ptr,
    comm.comm_ptr,
    rotary.ptr,
    o_proj.ptr,
    num_tp_heads,
    num_tp_key_value_heads,
    head_dim
  );

  muillm_parallel_attention_module_ptr_t ret;
  ret.ptr = m;
  return ret;
}

// deinit
void muillm_parallel_attention_module_deinit_trampoline(
  muillm_parallel_attention_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

// forward
at::Tensor muillm_parallel_attention_module_forward_trampoline(
  muillm_parallel_attention_module_ptr_t module_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& mask_,
  std::optional<torch::Tensor>& residual_) {

  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;
  torch::Tensor& residual = residual_.has_value() ? residual_.value() : undef_tensor;
  
  MuiLLMParallelAttention* m = module_ptr.ptr;

  return m->forward(q, k, v, mask, residual);
}

at::Tensor muillm_parallel_attention_module_rope_forward_trampoline(
  muillm_parallel_attention_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& mask_,
  torch::Tensor& residual,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
) {
  MuiLLMParallelAttention* attention_module = module_ptr.ptr;
  MuillmKVCache* cache = cache_ptr.ptr;

  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;
  return attention_module->rope_forward(cache, q, k, v, mask, residual, position_ids, cos_sin, cache_positions);
}