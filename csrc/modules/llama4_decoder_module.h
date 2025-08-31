#ifndef __MUILLM_LLAMA4_DECODER_MODULE_H__
#define __MUILLM_LLAMA4_DECODER_MODULE_H__


#include "../engine.h"

#include "multilinear_module.h"
#include "llama4_attention_module.h"
#include "gateup_module_interface.h"
#include "kvcache.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>

struct MuiLLMLlama4Decoder: torch::nn::Module {
  // fields
  muillm_engine_t* engine;

  MuiLLMMultiLinear* multilinear;
  MuiLLMLlama4Attention* attention;
  MuiLLMGateUpDownMLPInterface* mlp;

  bool use_chunked_attention;

  // methods
  MuiLLMLlama4Decoder(
    muillm_engine_t* engine,
    MuiLLMMultiLinear* multilinear,
    MuiLLMLlama4Attention* attention,
    MuiLLMGateUpDownMLPInterface* mlp,
    bool use_chunked_attention
  );

  virtual ~MuiLLMLlama4Decoder();

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
typedef struct muillm_llama4_decoder_module_ptr {
  MuiLLMLlama4Decoder* ptr;
} muillm_llama4_decoder_module_ptr_t;

muillm_llama4_decoder_module_ptr_t muillm_llama4_decoder_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_multilinear_module_ptr_t multilinear,
  muillm_llama4_attention_module_ptr_t attention,
  muillm_igateupdownmlp_module_ptr_t mlp,
  bool use_chunked_attention
);

void muillm_llama4_decoder_module_deinit_trampoline(
  muillm_llama4_decoder_module_ptr_t module_ptr
);

at::Tensor muillm_llama4_decoder_module_forward(
  muillm_llama4_decoder_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& mask,
  std::optional<torch::Tensor>& chunked_mask,
  torch::Tensor& position_embeds,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_LLAMA4_DECODER_MODULE_H__ */