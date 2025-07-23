#ifndef __MUILLM_PARALLEL_LLAMA4_DECODER_MODULE_H__
#define __MUILLM_PARALLEL_LLAMA4_DECODER_MODULE_H__


#include "../engine.h"
#include "../comms/comm_torch.h"

#include "parallel_multilinear_module.h"
#include "parallel_llama4_attention_module.h"
#include "parallel_gateup_module_interface.h"
#include "kvcache.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>

struct MuiLLMParallelLlama4Decoder: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;

  MuiLLMParallelMultiLinear* multilinear;
  MuiLLMParallelLlama4Attention* attention;
  MuiLLMParallelGateUpDownMLPInterface* mlp;

  bool use_chunked_attention;

  // methods
  MuiLLMParallelLlama4Decoder(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    MuiLLMParallelMultiLinear* multilinear,
    MuiLLMParallelLlama4Attention* attention,
    MuiLLMParallelGateUpDownMLPInterface* mlp,
    bool use_chunked_attention
  );

  virtual ~MuiLLMParallelLlama4Decoder();

  torch::Tensor forward(
    MuillmKVCache* cache,
    torch::Tensor& h,
    torch::Tensor& mask,
    torch::Tensor& chunked_mask,
    torch::Tensor& position_embeds,
    torch::Tensor& cache_positions
  );
};

// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_parallel_llama4_decoder_module_ptr {
  MuiLLMParallelLlama4Decoder* ptr;
} muillm_parallel_llama4_decoder_module_ptr_t;

muillm_parallel_llama4_decoder_module_ptr_t muillm_parallel_llama4_decoder_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_multilinear_module_ptr_t multilinear,
  muillm_parallel_llama4_attention_module_ptr_t attention,
  muillm_parallel_igateupdownmlp_module_ptr_t mlp,
  bool use_chunked_attention
);

void muillm_parallel_llama4_decoder_module_deinit_trampoline(
  muillm_parallel_llama4_decoder_module_ptr_t module_ptr
);

at::Tensor muillm_parallel_llama4_decoder_module_forward(
  muillm_parallel_llama4_decoder_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& mask,
  std::optional<torch::Tensor>& chunked_mask,
  torch::Tensor& position_embeds,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_PARALLEL_LLAMA4_DECODER_MODULE_H__ */