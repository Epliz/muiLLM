#ifndef __MUILLM_PARALLEL_LLAMA4_DECODER_STACK_H__
#define __MUILLM_PARALLEL_LLAMA4_DECODER_STACK_H__

#include "../engine.h"
#include "../comm_torch.h"

#include "parallel_llama4_decoder_module.h"
#include "kvcache.h"

#include <vector>
#include <optional>

struct MuiLLMParallelLlama4DecoderStack: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;

  std::vector<MuiLLMParallelLlama4Decoder*> decoders;

  // methods
  MuiLLMParallelLlama4DecoderStack(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    std::vector<MuiLLMParallelLlama4Decoder*>& decoders
  );

  virtual ~MuiLLMParallelLlama4DecoderStack();

  torch::Tensor forward(
    MuillmKVCache* cache,
    torch::Tensor& h,
    torch::Tensor& attention_mask,
    torch::Tensor& chunked_mask,
    torch::Tensor& position_emebds,
    torch::Tensor& cache_positions
  );
};

// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_parallel_llama4_decoder_stack_ptr {
    MuiLLMParallelLlama4DecoderStack* ptr;
} muillm_parallel_llama4_decoder_stack_ptr_t;

muillm_parallel_llama4_decoder_stack_ptr_t muillm_parallel_llama4_decoder_stack_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  std::vector<muillm_parallel_llama4_decoder_module_ptr_t>& decoders
);

void muillm_parallel_llama4_decoder_stack_deinit_trampoline(
  muillm_parallel_llama4_decoder_stack_ptr_t module_ptr
);

torch::Tensor muillm_parallel_llama4_decoder_stack_forward_trampoline(
  muillm_parallel_llama4_decoder_stack_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& attention_mask,
  std::optional<torch::Tensor>& chunked_mask,
  torch::Tensor& position_embeds,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_PARALLEL_LLAMA4_DECODER_STACK_H__ */