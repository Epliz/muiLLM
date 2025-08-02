#include "parallel_decoder_stack.h"

MuiLLMParallelDecoderStack::MuiLLMParallelDecoderStack(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  MuillmRotaryEmbedding* rotary_embedding,
  std::vector<MuiLLMParallelDecoder*>& decoders
) {
  this->engine = engine;
  this->comm = comm;

  this->rotary_embedding = rotary_embedding;
  this->decoders = decoders;
}

MuiLLMParallelDecoderStack::~MuiLLMParallelDecoderStack() {
  // nothing to do
}

torch::Tensor MuiLLMParallelDecoderStack::forward(
  MuillmKVCache* cache,
  torch::Tensor& h,
  torch::Tensor& m,
  torch::Tensor& position_ids,
  torch::Tensor& cache_positions
) {
  torch::Tensor h_ = h;

  std::optional<std::tuple<torch::Tensor, torch::Tensor>> cos_sin = this->rotary_embedding != nullptr ?
      std::make_optional<std::tuple<torch::Tensor, torch::Tensor>>(rotary_embedding->compute_rotary_pos_emb(h, position_ids))
      : std::optional<std::tuple<torch::Tensor, torch::Tensor>>();

  for (auto decoder: this->decoders) {
    h_ = decoder->forward(cache, h_, m, position_ids, cos_sin, cache_positions);
  }

  return h_;
}

muillm_parallel_decoder_stack_ptr_t muillm_parallel_decoder_stack_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_rotary_embedding_module_ptr_t& rotary_embedding_module,
  std::vector<muillm_parallel_decoder_module_ptr_t>& decoders
) {
  std::vector<MuiLLMParallelDecoder*> decoders_;
  for (auto decoder: decoders) {
    decoders_.push_back(decoder.ptr);
  }

  MuiLLMParallelDecoderStack* decoder_stack = new MuiLLMParallelDecoderStack(
    engine.engine_ptr,
    comm.comm_ptr,
    rotary_embedding_module.ptr,
    decoders_
  );

  muillm_parallel_decoder_stack_ptr_t ret;
  ret.ptr = decoder_stack;
  return ret;
}

void muillm_parallel_decoder_stack_deinit_trampoline(
  muillm_parallel_decoder_stack_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

torch::Tensor muillm_parallel_decoder_stack_forward_trampoline(
  muillm_parallel_decoder_stack_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& m_,
  torch::Tensor& position_ids,
  torch::Tensor& cache_positions
) {
  auto undef_tensor = torch::Tensor();
  auto m = m_.has_value() ? m_.value() : undef_tensor;

  return module_ptr.ptr->forward(cache_ptr.ptr, h, m, position_ids, cache_positions);
}