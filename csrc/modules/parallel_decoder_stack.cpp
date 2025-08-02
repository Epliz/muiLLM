#include "parallel_decoder_stack.h"

MuiLLMParallelDecoderStack::MuiLLMParallelDecoderStack(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  MuiLLMEmbedding* embed_tokens,
  MuillmRotaryEmbedding* rotary_embedding,
  std::vector<MuiLLMParallelDecoder*>& decoders
) {
  this->engine = engine;
  this->comm = comm;

  this->embed_tokens = embed_tokens;
  this->rotary_embedding = rotary_embedding;
  this->decoders = decoders;
}

MuiLLMParallelDecoderStack::~MuiLLMParallelDecoderStack() {
  // nothing to do
}

torch::Tensor MuiLLMParallelDecoderStack::forward(
  MuillmKVCache* cache,
  torch::Tensor& input_ids,
  torch::Tensor& input_embeds,
  torch::Tensor& m,
  torch::Tensor& position_ids,
  torch::Tensor& cache_positions
) {
  // compute the input embeddings if not provided
  bool inputs_defined = input_embeds.defined();
  torch::Tensor h = inputs_defined ? input_embeds : this->embed_tokens->forward(input_ids);

  std::optional<std::tuple<torch::Tensor, torch::Tensor>> cos_sin = this->rotary_embedding != nullptr ?
      std::make_optional<std::tuple<torch::Tensor, torch::Tensor>>(rotary_embedding->compute_rotary_pos_emb(h, position_ids))
      : std::optional<std::tuple<torch::Tensor, torch::Tensor>>();

  torch::Tensor h_ = h;

  for (auto decoder: this->decoders) {
    h_ = decoder->forward(cache, h_, m, position_ids, cos_sin, cache_positions);
  }

  return h_;
}

muillm_parallel_decoder_stack_ptr_t muillm_parallel_decoder_stack_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_embedding_module_ptr_t& embed_tokens,
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
    embed_tokens.ptr,
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
  torch::Tensor& input_ids,
  std::optional<torch::Tensor>& input_embeds_,
  std::optional<torch::Tensor>& m_,
  torch::Tensor& position_ids,
  torch::Tensor& cache_positions
) {
  auto undef_tensor = torch::Tensor();
  auto input_embeds = input_embeds_.has_value() ? input_embeds_.value() : undef_tensor;
  auto m = m_.has_value() ? m_.value() : undef_tensor;

  return module_ptr.ptr->forward(cache_ptr.ptr, input_ids, input_embeds, m, position_ids, cache_positions);
}