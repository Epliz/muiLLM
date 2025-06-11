#include "parallel_llama4_decoder_stack.h"

MuiLLMParallelLlama4DecoderStack::MuiLLMParallelLlama4DecoderStack(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  std::vector<MuiLLMParallelLlama4Decoder*>& decoders
) {
  this->engine = engine;
  this->comm = comm;

  this->decoders = decoders;
}

MuiLLMParallelLlama4DecoderStack::~MuiLLMParallelLlama4DecoderStack() {
  // nothing to do
}

torch::Tensor MuiLLMParallelLlama4DecoderStack::forward(
  MuillmKVCache* cache,
  torch::Tensor& h,
  torch::Tensor& attention_mask,
  torch::Tensor& chunked_mask,
  torch::Tensor& position_embeds,
  torch::Tensor& cache_positions
) {
  torch::Tensor h_ = h;

  for (auto decoder: this->decoders) {
    h_ = decoder->forward(cache, h_, attention_mask, chunked_mask, position_embeds, cache_positions);
  }

  return h_;
}

muillm_parallel_llama4_decoder_stack_ptr_t muillm_parallel_llama4_decoder_stack_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  std::vector<muillm_parallel_llama4_decoder_module_ptr_t>& decoders
) {
  std::vector<MuiLLMParallelLlama4Decoder*> decoders_;
  for (auto decoder: decoders) {
    decoders_.push_back(decoder.ptr);
  }

  MuiLLMParallelLlama4DecoderStack* decoder_stack = new MuiLLMParallelLlama4DecoderStack(
    engine.engine_ptr,
    comm.comm_ptr,
    decoders_
  );

  muillm_parallel_llama4_decoder_stack_ptr_t ret;
  ret.ptr = decoder_stack;
  return ret;
}

void muillm_parallel_llama4_decoder_stack_deinit_trampoline(
  muillm_parallel_llama4_decoder_stack_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

torch::Tensor muillm_parallel_llama4_decoder_stack_forward_trampoline(
  muillm_parallel_llama4_decoder_stack_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& mask_,
  std::optional<torch::Tensor>& chunked_mask_,
  torch::Tensor& position_embeds,
  torch::Tensor& cache_positions
) {
  auto undef_tensor = torch::Tensor();
  auto mask = mask_.has_value() ? mask_.value() : undef_tensor;
  auto chunked_mask = chunked_mask_.has_value() ? chunked_mask_.value() : undef_tensor;

  return module_ptr.ptr->forward(cache_ptr.ptr, h, mask, chunked_mask, position_embeds, cache_positions);
}