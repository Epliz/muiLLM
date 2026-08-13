#include "gemma3_attention_module.h"

#include "../norm/qkrmsnorm.cuh"
#include "../temperaturetuning/temperature_tuning.cuh"
#include "hybrid_chunked_kvcache.h"
#include "../attention/causal_transformer_decoding.cuh"

//
// Actual code
//

MuiLLMGemma3Attention::MuiLLMGemma3Attention(
  muillm_engine_t* engine,
  MuiLLMLinear* o_proj,
  int num_heads,
  int num_key_value_heads,
  int head_dim,
  torch::Tensor& q_norm_weight,
  torch::Tensor& k_norm_weight,
  float norm_epsilon,
  float norm_weights_offset,
  int layer_index
) {
  this->engine = engine;
  this->o_proj = o_proj;

  this->num_heads = num_heads;
  this->num_key_value_heads = num_key_value_heads;
  this->head_dim = head_dim;

  this->layer_index = layer_index;

  this->q_norm_weight = q_norm_weight;
  this->k_norm_weight = k_norm_weight;
  this->norm_epsilon = norm_epsilon;
  this->norm_weights_offset = norm_weights_offset;
}

torch::Tensor MuiLLMGemma3Attention::forward(
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  torch::Tensor& m
) {
  bool masked = m.defined();
  auto attn_output = masked ?
      muillm_causal_transformer_decoding_masked(q, k, v, m)
    : muillm_causal_transformer_decoding_no_mask(q, k, v);


  // o proj
  auto undef_tensor = torch::Tensor();
  auto proj_attn_output = this->o_proj->forward(
    attn_output,
    /* mul_residual */ undef_tensor,
    /* residual */ undef_tensor
  );
  return proj_attn_output;
}

torch::Tensor MuiLLMGemma3Attention::rope_forward(
  MuillmKVCache* cache,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  torch::Tensor& m,
  torch::Tensor& cos,
  torch::Tensor& sin,
  torch::Tensor& cache_positions
) {

  auto undef_tensor = torch::Tensor();

  int bsz = q.size(0);
  int q_len = q.size(1);

  torch::Tensor q_res;
  torch::Tensor k_res;
  torch::Tensor v_res;

  if (q_len == 1) {
    // as q_len is 1, we can avoid the transpose
    q_res = q.view({bsz, this->num_heads, q_len, this->head_dim});

    k_res = k.view({bsz, this->num_key_value_heads, q_len, this->head_dim});
    v_res = v.view({bsz, this->num_key_value_heads, q_len, this->head_dim});
  } else {
    q_res = q.view({bsz, q_len, this->num_heads, this->head_dim}).transpose(1, 2);

    k_res = k.view({bsz, q_len, this->num_key_value_heads, this->head_dim}).transpose(1, 2);
    v_res = v.view({bsz, q_len, this->num_key_value_heads, this->head_dim}).transpose(1, 2);
  }

  {
    auto [q_normalized, k_normalized] = muillm_qkrmsnorm_forward(
      this->q_norm_weight,
      this->k_norm_weight,
      q_res,
      k_res,
      this->norm_epsilon,
      this->norm_weights_offset
    );
    
    q_res = q_normalized;
    k_res = k_normalized;
  }

  auto cos_sin_tuple = std::make_tuple(cos, sin);

  auto qkv_tuple = cache->rope_update(
    q_res,
    k_res,
    v_res,
    cos_sin_tuple,
    cache_positions,
    this->layer_index
  );
  q_res = std::get<0>(qkv_tuple);
  k_res = std::get<1>(qkv_tuple);
  v_res = std::get<2>(qkv_tuple);

  // attention
  return this->forward(q_res, k_res, v_res, m);
}

//
// Python trampolines
//

// init
muillm_gemma3_attention_module_ptr muillm_gemma3_attention_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_linear_module_ptr_t o_proj,
  int num_heads,
  int num_key_value_heads,
  int head_dim,
  torch::Tensor& q_norm_weight,
  torch::Tensor& k_norm_weight,
  float norm_epsilon,
  float norm_weights_offset,
  int layer_index
) {

  MuiLLMGemma3Attention* m = new MuiLLMGemma3Attention(
    engine.engine_ptr,
    o_proj.ptr,
    num_heads,
    num_key_value_heads,
    head_dim,
    q_norm_weight,
    k_norm_weight,
    norm_epsilon,
    norm_weights_offset,
    layer_index
  );

  muillm_gemma3_attention_module_ptr_t ret;
  ret.ptr = m;
  return ret;
}

// deinit
void muillm_gemma3_attention_module_deinit_trampoline(
  muillm_gemma3_attention_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

// forward
at::Tensor muillm_gemma3_attention_module_forward_trampoline(
  muillm_gemma3_attention_module_ptr_t module_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& mask_) {

  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;
  
  MuiLLMGemma3Attention* m = module_ptr.ptr;

  return m->forward(q, k, v, mask);
}

at::Tensor muillm_gemma3_attention_module_rope_forward_trampoline(
  muillm_gemma3_attention_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& mask_,
  torch::Tensor& cos,
  torch::Tensor& sin,
  torch::Tensor& cache_positions
) {
  MuiLLMGemma3Attention* attention_module = module_ptr.ptr;
  MuillmKVCache* cache = cache_ptr.ptr;

  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;

  return attention_module->rope_forward(cache, q, k, v, mask, cos, sin, cache_positions);
}