#include "parallel_decoder_module.h"

MuiLLMParallelDecoder::MuiLLMParallelDecoder(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  MuiLLMParallelAttention* attention,
  MuiLLMParallelGateUpDownMLP* mlp
) {
  this->engine = engine;
  this->comm = comm;

  this->attention = attention;
  this->mlp = mlp;
}

MuiLLMParallelDecoder::~MuiLLMParallelDecoder() {
  // nothing to do
}

torch::Tensor MuiLLMParallelDecoder::forward(
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

muillm_parallel_decoder_module_ptr_t muillm_parallel_decoder_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_attention_module_ptr_t attention,
  muillm_parallel_gateupdownmlp_module_ptr_t mlp
) {
  MuiLLMParallelDecoder* decoder_module = new MuiLLMParallelDecoder(
    engine.engine_ptr,
    comm.comm_ptr,
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
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& mask_,
  torch::Tensor& residual,
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
    q,
    k,
    v,
    mask,
    residual,
    position_ids,
    cos_sin,
    cache_positions
  );
}