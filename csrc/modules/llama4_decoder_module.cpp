#include "llama4_decoder_module.h"

MuiLLMLlama4Decoder::MuiLLMLlama4Decoder(
  muillm_engine_t* engine,
  MuiLLMMultiLinear* multilinear,
  MuiLLMLlama4Attention* attention,
  MuiLLMGateUpDownMLPInterface* mlp,
  bool use_chunked_attention
) {
  this->engine = engine;

  this->multilinear = multilinear;
  this->attention = attention;
  this->mlp = mlp;

  this->use_chunked_attention = use_chunked_attention;
}

MuiLLMLlama4Decoder::~MuiLLMLlama4Decoder() {
  // nothing to do
}

torch::Tensor MuiLLMLlama4Decoder::forward(
  MuillmKVCache* cache,
  torch::Tensor& h,
  torch::Tensor& mask,
  torch::Tensor& chunked_mask,
  torch::Tensor& position_embeds,
  torch::Tensor& cache_positions
) {
  auto residual = h;

  auto qkv = this->multilinear->forward(
    h
  );

  auto q = qkv[0];
  auto k = qkv[1];
  auto v = qkv[2];

  auto& attention_mask = use_chunked_attention ? chunked_mask : mask;

  auto attention_out = this->attention->rope_forward(
    cache,
    q,
    k,
    v,
    attention_mask,
    residual,
    position_embeds,
    cache_positions
  );

  auto mlp_residual = attention_out;

  auto mlp_out = this->mlp->forward(
    attention_out,
    mlp_residual
  );

  return mlp_out;
}

muillm_llama4_decoder_module_ptr_t muillm_llama4_decoder_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_multilinear_module_ptr_t multilinear,
  muillm_llama4_attention_module_ptr_t attention,
  muillm_igateupdownmlp_module_ptr_t mlp,
  bool use_chunked_attention
) {
  MuiLLMLlama4Decoder* decoder_module = new MuiLLMLlama4Decoder(
    engine.engine_ptr,
    multilinear.ptr,
    attention.ptr,
    mlp.ptr,
    use_chunked_attention
  );

  muillm_llama4_decoder_module_ptr_t module_ptr;
  module_ptr.ptr = decoder_module;

  return module_ptr;
}

void muillm_llama4_decoder_module_deinit_trampoline(
  muillm_llama4_decoder_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

at::Tensor muillm_llama4_decoder_module_forward(
  muillm_llama4_decoder_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& mask_,
  std::optional<torch::Tensor>& chunked_mask_,
  torch::Tensor& position_embeds,
  torch::Tensor& cache_positions
) {
  auto cache = cache_ptr.ptr;

  MuiLLMLlama4Decoder* decoder_module = module_ptr.ptr;


  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;
  torch::Tensor& chunked_mask = chunked_mask_.has_value() ? chunked_mask_.value() : undef_tensor;

  return decoder_module->forward(
    cache,
    h,
    mask,
    chunked_mask,
    position_embeds,
    cache_positions
  );
}