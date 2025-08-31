#include "parallel_gemma3_decoder_module.h"

#include "../norm/rmsnorm.cuh"

MuiLLMParallelGemma3Decoder::MuiLLMParallelGemma3Decoder(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  MuiLLMParallelMultiLinear* multilinear,
  MuiLLMParallelGemma3Attention* attention,
  MuiLLMParallelGateUpDownMLPInterface* mlp,
  bool sliding_layer,
  torch::Tensor& post_attention_layer_norm_weight,
  float post_attention_layer_norm_epsilon,
  float post_attention_layer_norm_weights_offset,
  torch::Tensor& post_feedforward_layer_norm_weight,
  float post_feedforward_layer_norm_epsilon,
  float post_feedforward_layer_norm_weights_offset
) {
  this->engine = engine;

  this->multilinear = multilinear;
  this->attention = attention;
  this->mlp = mlp;

  this->sliding_layer = sliding_layer;

  this->post_attention_layer_norm_weight = post_attention_layer_norm_weight;
  this->post_attention_layer_norm_epsilon = post_attention_layer_norm_epsilon;
  this->post_attention_layer_norm_weights_offset = post_attention_layer_norm_weights_offset;

  this->post_feedforward_layer_norm_weight = post_feedforward_layer_norm_weight;
  this->post_feedforward_layer_norm_epsilon = post_feedforward_layer_norm_epsilon;
  this->post_feedforward_layer_norm_weights_offset = post_feedforward_layer_norm_weights_offset;
}

MuiLLMParallelGemma3Decoder::~MuiLLMParallelGemma3Decoder() {
  // nothing to do
}

torch::Tensor MuiLLMParallelGemma3Decoder::forward(
  MuillmKVCache* cache,
  torch::Tensor& h,
  torch::Tensor& mask,
  torch::Tensor& sliding_mask,
  std::tuple<torch::Tensor, torch::Tensor>& position_embeds_global,
  std::tuple<torch::Tensor, torch::Tensor>& position_embeds_local,
  torch::Tensor& cache_positions
) {
  auto undef_tensor = torch::Tensor();

  auto residual = h;

  auto qkv = this->multilinear->forward(
    h,
    /* collect */ false
  );

  auto q = qkv[0];
  auto k = qkv[1];
  auto v = qkv[2];

  auto& attention_mask = this->sliding_layer ? sliding_mask : mask;
  auto& position_embeds = this->sliding_layer ? position_embeds_local : position_embeds_global;
  auto [cos, sin] = position_embeds;

  auto attention_out = this->attention->rope_forward(
    cache,
    q,
    k,
    v,
    attention_mask,
    cos,
    sin,
    cache_positions
  );

  // post attention layer norm (with residual)
  h = muillm_rmsnorm_forward(
    this->post_attention_layer_norm_weight,
    attention_out,
    residual,
    this->post_attention_layer_norm_epsilon,
    this->post_attention_layer_norm_weights_offset
  );

  auto post_feedforward_residual = h;

  auto mlp_out = this->mlp->forward(
    h,
    /* residual */ undef_tensor,
    /*reduce*/ true
  );

  // post MLP layer norm (with residual)
  h = muillm_rmsnorm_forward(
    this->post_feedforward_layer_norm_weight,
    mlp_out,
    post_feedforward_residual,
    this->post_feedforward_layer_norm_epsilon,
    this->post_feedforward_layer_norm_weights_offset
  );

  return h;
}

muillm_parallel_gemma3_decoder_module_ptr_t muillm_parallel_gemma3_decoder_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_multilinear_module_ptr_t multilinear,
  muillm_parallel_gemma3_attention_module_ptr_t attention,
  muillm_parallel_igateupdownmlp_module_ptr_t mlp,
  bool sliding_layer,
  torch::Tensor& post_attention_layer_norm_weight,
  float post_attention_layer_norm_epsilon,
  float post_attention_layer_norm_weights_offset,
  torch::Tensor& post_feedforward_layer_norm_weight,
  float post_feedforward_layer_norm_epsilon,
  float post_feedforward_layer_norm_weights_offset
) {
  MuiLLMParallelGemma3Decoder* decoder_module = new MuiLLMParallelGemma3Decoder(
    engine.engine_ptr,
    comm.comm_ptr,
    multilinear.ptr,
    attention.ptr,
    mlp.ptr,
    sliding_layer,
    post_attention_layer_norm_weight,
    post_attention_layer_norm_epsilon,
    post_attention_layer_norm_weights_offset,
    post_feedforward_layer_norm_weight,
    post_feedforward_layer_norm_epsilon,
    post_feedforward_layer_norm_weights_offset
  );

  muillm_parallel_gemma3_decoder_module_ptr_t module_ptr;
  module_ptr.ptr = decoder_module;

  return module_ptr;
}

void muillm_parallel_gemma3_decoder_module_deinit_trampoline(
  muillm_parallel_gemma3_decoder_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

at::Tensor muillm_parallel_gemma3_decoder_module_forward(
  muillm_parallel_gemma3_decoder_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& mask_,
  std::optional<torch::Tensor>& sliding_mask_,
  std::tuple<torch::Tensor, torch::Tensor> position_embeds_global,
  std::tuple<torch::Tensor, torch::Tensor> position_embeds_local,
  torch::Tensor& cache_positions
) {
  auto cache = cache_ptr.ptr;

  MuiLLMParallelGemma3Decoder* decoder_module = module_ptr.ptr;

  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;
  torch::Tensor& sliding_mask = sliding_mask_.has_value() ? sliding_mask_.value() : undef_tensor;

  return decoder_module->forward(
    cache,
    h,
    mask,
    sliding_mask,
    position_embeds_global,
    position_embeds_local,
    cache_positions
  );
}