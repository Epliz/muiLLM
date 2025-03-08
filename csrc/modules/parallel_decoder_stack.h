#ifndef __MUILLM_PARALLEL_DECODER_STACK_H__
#define __MUILLM_PARALLEL_DECODER_STACK_H__

#include "../engine.h"
#include "../comm_torch.h"

#include "parallel_decoder_module.h"
#include "kvcache.h"

#include <vector>
#include <optional>

struct MuiLLMParallelDecoderStack: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;

  std::vector<MuiLLMParallelDecoder*> decoders;

  // methods
  MuiLLMParallelDecoderStack(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    std::vector<MuiLLMParallelDecoder*>& decoders
  );

  virtual ~MuiLLMParallelDecoderStack();

  torch::Tensor forward(
    MuillmKVCache* cache,
    torch::Tensor& h,
    torch::Tensor& m,
    torch::Tensor& position_ids,
    std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
    torch::Tensor& cache_positions
  );
};

// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_parallel_decoder_stack_ptr {
    MuiLLMParallelDecoderStack* ptr;
} muillm_parallel_decoder_stack_ptr_t;

muillm_parallel_decoder_stack_ptr_t muillm_parallel_decoder_stack_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  std::vector<muillm_parallel_decoder_module_ptr_t>& decoders
);

void muillm_parallel_decoder_stack_deinit_trampoline(
  muillm_parallel_decoder_stack_ptr_t module_ptr
);

torch::Tensor muillm_parallel_decoder_stack_forward_trampoline(
  muillm_parallel_decoder_stack_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& m,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_PARALLEL_DECODER_STACK_H__ */