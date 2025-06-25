#include "parallel_llama4_attention_module.h"

#include "../rope/rotary.h"
#include "../norm/qkl2norm.cuh"
#include "../temperaturetuning/temperature_tuning.cuh"
#include "hybrid_chunked_kvcache.h"
#include "../causal_transformer_decoding.cuh"

//
// Actual code
//

MuiLLMParallelLlama4Attention::MuiLLMParallelLlama4Attention(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  MuiLLMParallelLinear* o_proj,
  int num_tp_heads,
  int num_tp_key_value_heads,
  int head_dim,
  bool use_rope,
  bool use_qk_norm,
  float norm_epsilon,
  bool use_temperature_tuning,
  float attention_scale,
  float floor_scale,
  int layer_index
) {
  this->engine = engine;
  this->comm = comm;
  this->o_proj = o_proj;

  this->num_tp_heads = num_tp_heads;
  this->num_tp_key_value_heads = num_tp_key_value_heads;
  this->head_dim = head_dim;

  this->layer_index = layer_index;

  this->use_rope = use_rope;

  this->use_qk_norm = use_qk_norm;
  this->norm_epsilon = norm_epsilon;

  this->use_temperature_tuning = use_temperature_tuning;
  this->attention_scale = attention_scale;
  this->floor_scale = floor_scale;
}

torch::Tensor MuiLLMParallelLlama4Attention::forward(
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

torch::Tensor MuiLLMParallelLlama4Attention::rope_forward(
  MuillmKVCache* cache,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  torch::Tensor& m,
  torch::Tensor& residual,
  torch::Tensor& position_embeds,
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

  if (this->use_rope) {
    auto [q_rot, k_rot] = muillm_complex_rope_forward_no_cache(
      q_res,
      k_res,
      position_embeds
    );

    q_res = q_rot;
    k_res = k_rot;
  }

  if (this->use_qk_norm) {
    auto [q_normalized, k_normalized] = muillm_qkl2norm_forward(
      q_res,
      k_res,
      this->norm_epsilon
    );
    
    q_res = q_normalized;
    k_res = k_normalized;
  }

  if (this->use_temperature_tuning) {
    q_res = muillm_apply_temperature_tuning(
      q_res,
      cache_positions,
      this->attention_scale,
      this->floor_scale
    );
  }

  // store in cache
  if (cache->type != MUILLM_HYBRID_CHUNKED_KVCACHE) {
    TORCH_CHECK(false, "expected a hybrid chunked cache");
  }

  MuillmHybridChunkedKVCache* hybrid_cache = (MuillmHybridChunkedKVCache*) cache;
  auto [k_out, v_out] = hybrid_cache->update(
    k_res,
    v_res,
    cache_positions,
    this->layer_index
  );

  k_res = k_out;
  v_res = v_out;


  // attention
  return this->forward(q_res, k_res, v_res, m, residual);
}

//
// Python trampolines
//

// init
muillm_parallel_llama4_attention_module_ptr muillm_parallel_llama4_attention_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_linear_module_ptr_t o_proj,
  int num_tp_heads,
  int num_tp_key_value_heads,
  int head_dim,
  bool use_rope,
  bool use_qk_norm,
  float norm_epsilon,
  bool use_temperature_tuning,
  float attention_scale,
  float floor_scale,
  int layer_index
) {

  MuiLLMParallelLlama4Attention* m = new MuiLLMParallelLlama4Attention(
    engine.engine_ptr,
    comm.comm_ptr,
    o_proj.ptr,
    num_tp_heads,
    num_tp_key_value_heads,
    head_dim,
    use_rope,
    use_qk_norm,
    norm_epsilon,
    use_temperature_tuning,
    attention_scale,
    floor_scale,
    layer_index
  );

  muillm_parallel_llama4_attention_module_ptr_t ret;
  ret.ptr = m;
  return ret;
}

// deinit
void muillm_parallel_llama4_attention_module_deinit_trampoline(
  muillm_parallel_llama4_attention_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

// forward
at::Tensor muillm_parallel_llama4_attention_module_forward_trampoline(
  muillm_parallel_llama4_attention_module_ptr_t module_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& mask_,
  std::optional<torch::Tensor>& residual_) {

  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;
  torch::Tensor& residual = residual_.has_value() ? residual_.value() : undef_tensor;
  
  MuiLLMParallelLlama4Attention* m = module_ptr.ptr;

  return m->forward(q, k, v, mask, residual);
}

at::Tensor muillm_parallel_llama4_attention_module_rope_forward_trampoline(
  muillm_parallel_llama4_attention_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& mask_,
  torch::Tensor& residual,
  torch::Tensor& position_embeds,
  torch::Tensor& cache_positions
) {
  MuiLLMParallelLlama4Attention* attention_module = module_ptr.ptr;
  MuillmKVCache* cache = cache_ptr.ptr;

  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;
  return attention_module->rope_forward(cache, q, k, v, mask, residual, position_embeds, cache_positions);
}