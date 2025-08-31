#include "decoder_module.h"

MuiLLMDecoder::MuiLLMDecoder(
  muillm_engine_t* engine,
  MuiLLMMultiLinear* multilinear,
  MuiLLMAttention* attention,
  MuiLLMGateUpDownMLP* mlp
) {
  this->engine = engine;

  this->multilinear = multilinear;
  this->attention = attention;
  this->mlp = mlp;
}

MuiLLMDecoder::~MuiLLMDecoder() {
  // nothing to do
}

torch::Tensor MuiLLMDecoder::forward(
  MuillmKVCache* cache,
  torch::Tensor& h,
  torch::Tensor& m,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
) {
  auto residual = h;

  auto qkv = this->multilinear->forward(
    h
  );

  auto q = qkv[0];
  auto k = qkv[1];
  auto v = qkv[2];

  auto attention_out = this->attention->rope_forward(
    cache,
    q,
    k,
    v,
    m,
    residual,
    position_ids,
    cos_sin,
    cache_positions
  );

  auto mlp_residual = attention_out;

  auto mlp_out = this->mlp->forward(
    attention_out,
    mlp_residual
  );

  return mlp_out;
}

muillm_decoder_module_ptr_t muillm_decoder_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_multilinear_module_ptr_t multilinear,
  muillm_attention_module_ptr_t attention,
  muillm_igateupdownmlp_module_ptr_t mlp
) {
  MuiLLMDecoder* decoder_module = new MuiLLMDecoder(
    engine.engine_ptr,
    multilinear.ptr,
    attention.ptr,
    (MuiLLMGateUpDownMLP*)mlp.ptr
  );

  muillm_decoder_module_ptr_t module_ptr;
  module_ptr.ptr = decoder_module;

  return module_ptr;
}

void muillm_decoder_module_deinit_trampoline(
  muillm_decoder_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

at::Tensor muillm_decoder_module_forward(
  muillm_decoder_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& mask_,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
) {
  auto cache = cache_ptr.ptr;

  MuiLLMDecoder* decoder_module = module_ptr.ptr;


  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;

  return decoder_module->forward(
    cache,
    h,
    mask,
    position_ids,
    cos_sin,
    cache_positions
  );
}