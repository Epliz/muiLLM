#include "parallel_gemma3_attention_module.h"

#include "../rope/rotary.h"
#include "../norm/qkrmsnorm.cuh"
#include "../temperaturetuning/temperature_tuning.cuh"
#include "hybrid_chunked_kvcache.h"
#include "../attention/causal_transformer_decoding.cuh"

//
// Actual code
//

MuiLLMParallelGemma3Attention::MuiLLMParallelGemma3Attention(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  MuiLLMParallelLinear* o_proj,
  int num_tp_heads,
  int num_tp_key_value_heads,
  int head_dim,
  torch::Tensor& q_norm_weight,
  torch::Tensor& k_norm_weight,
  float norm_epsilon,
  float norm_weights_offset,
  int layer_index
) {
  this->engine = engine;
  this->comm = comm;
  this->o_proj = o_proj;

  this->num_tp_heads = num_tp_heads;
  this->num_tp_key_value_heads = num_tp_key_value_heads;
  this->head_dim = head_dim;

  this->layer_index = layer_index;

  this->q_norm_weight = q_norm_weight;
  this->k_norm_weight = k_norm_weight;
  this->norm_epsilon = norm_epsilon;
  this->norm_weights_offset = norm_weights_offset;
}

torch::Tensor MuiLLMParallelGemma3Attention::forward(
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
  auto proj_attn_output = this->o_proj->forward(attn_output, /* residual */ undef_tensor, /*collect_outputs*/ true);
  return proj_attn_output;
}

torch::Tensor MuiLLMParallelGemma3Attention::rope_forward(
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
    q_res = q.view({bsz, this->num_tp_heads, q_len, this->head_dim});

    k_res = k.view({bsz, this->num_tp_key_value_heads, q_len, this->head_dim});
    v_res = v.view({bsz, this->num_tp_key_value_heads, q_len, this->head_dim});
  } else {
    q_res = q.view({bsz, q_len, this->num_tp_heads, this->head_dim}).transpose(1, 2);

    k_res = k.view({bsz, q_len, this->num_tp_key_value_heads, this->head_dim}).transpose(1, 2);
    v_res = v.view({bsz, q_len, this->num_tp_key_value_heads, this->head_dim}).transpose(1, 2);
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

  {
    auto [q_rot, k_rot] = muillm_rope_forward_no_cache(
      undef_tensor, // no position ids
      cos,
      sin,
      q_res,
      k_res
    );

    q_res = q_rot;
    k_res = k_rot;
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
  return this->forward(q_res, k_res, v_res, m);
}

//
// Python trampolines
//

// init
muillm_parallel_gemma3_attention_module_ptr muillm_parallel_gemma3_attention_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_linear_module_ptr_t o_proj,
  int num_tp_heads,
  int num_tp_key_value_heads,
  int head_dim,
  torch::Tensor& q_norm_weight,
  torch::Tensor& k_norm_weight,
  float norm_epsilon,
  float norm_weights_offset,
  int layer_index
) {

  MuiLLMParallelGemma3Attention* m = new MuiLLMParallelGemma3Attention(
    engine.engine_ptr,
    comm.comm_ptr,
    o_proj.ptr,
    num_tp_heads,
    num_tp_key_value_heads,
    head_dim,
    q_norm_weight,
    k_norm_weight,
    norm_epsilon,
    norm_weights_offset,
    layer_index
  );

  muillm_parallel_gemma3_attention_module_ptr_t ret;
  ret.ptr = m;
  return ret;
}

// deinit
void muillm_parallel_gemma3_attention_module_deinit_trampoline(
  muillm_parallel_gemma3_attention_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

// forward
at::Tensor muillm_parallel_gemma3_attention_module_forward_trampoline(
  muillm_parallel_gemma3_attention_module_ptr_t module_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& mask_) {

  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;

  MuiLLMParallelGemma3Attention* m = module_ptr.ptr;

  return m->forward(q, k, v, mask);
}

at::Tensor muillm_parallel_gemma3_attention_module_rope_forward_trampoline(
  muillm_parallel_gemma3_attention_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& mask_,
  torch::Tensor& cos,
  torch::Tensor& sin,
  torch::Tensor& cache_positions
) {
  MuiLLMParallelGemma3Attention* attention_module = module_ptr.ptr;
  MuillmKVCache* cache = cache_ptr.ptr;

  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;

  return attention_module->rope_forward(cache, q, k, v, mask, cos, sin, cache_positions);
}