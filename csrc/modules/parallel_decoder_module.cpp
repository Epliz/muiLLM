#include "parallel_decoder_module.h"

MuiLLMParallelDecoder::MuiLLMParallelDecoder(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  MuiLLMParallelMultiLinear* multilinear,
  MuiLLMParallelAttention* attention,
  MuiLLMParallelGateUpDownMLP* mlp
) {
  this->engine = engine;
  this->comm = comm;

  this->multilinear = multilinear;
  this->attention = attention;
  this->mlp = mlp;
}

MuiLLMParallelDecoder::~MuiLLMParallelDecoder() {
  // nothing to do
}

torch::Tensor MuiLLMParallelDecoder::forward(
  MuillmKVCache* cache,
  torch::Tensor& h,
  torch::Tensor& m,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
) {
  auto residual = h;

  auto qkv = this->multilinear->forward(
    h,
    /* collect */ false
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
    mlp_residual,
    /*reduce*/ true
  );

  return mlp_out;
}

muillm_parallel_decoder_module_ptr_t muillm_parallel_decoder_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_multilinear_module_ptr_t multilinear,
  muillm_parallel_attention_module_ptr_t attention,
  muillm_parallel_gateupdownmlp_module_ptr_t mlp
) {
  MuiLLMParallelDecoder* decoder_module = new MuiLLMParallelDecoder(
    engine.engine_ptr,
    comm.comm_ptr,
    multilinear.ptr,
    attention.ptr,
    mlp.ptr
  );

  muillm_parallel_decoder_module_ptr_t module_ptr;
  module_ptr.ptr = decoder_module;

  return module_ptr;
}

void muillm_parallel_decoder_module_deinit_trampoline(
  muillm_parallel_decoder_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

at::Tensor muillm_parallel_decoder_module_forward(
  muillm_parallel_decoder_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& mask_,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
) {
  auto cache = cache_ptr.ptr;

  MuiLLMParallelDecoder* decoder_module = module_ptr.ptr;


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