#ifndef __MUILLM_DECODER_MODULE_H__
#define __MUILLM_DECODER_MODULE_H__


#include "../engine.h"
#include "../comms/comm_torch.h"

#include "multilinear_module.h"
#include "attention_module.h"
#include "gateup_module.h"
#include "kvcache.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>

struct MuiLLMDecoder: torch::nn::Module {
  // fields
  muillm_engine_t* engine;

  MuiLLMMultiLinear* multilinear;
  MuiLLMAttention* attention;
  MuiLLMGateUpDownMLP* mlp;

  // methods
  MuiLLMDecoder(
    muillm_engine_t* engine,
    MuiLLMMultiLinear* multilinear,
    MuiLLMAttention* attention,
    MuiLLMGateUpDownMLP* mlp
  );

  virtual ~MuiLLMDecoder();

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
typedef struct muillm_decoder_module_ptr {
  MuiLLMDecoder* ptr;
} muillm_decoder_module_ptr_t;

muillm_decoder_module_ptr_t muillm_decoder_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_multilinear_module_ptr_t multilinear,
  muillm_attention_module_ptr_t attention,
  muillm_igateupdownmlp_module_ptr_t mlp
);

void muillm_decoder_module_deinit_trampoline(
  muillm_decoder_module_ptr_t module_ptr
);

at::Tensor muillm_decoder_module_forward(
  muillm_decoder_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& m,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_DECODER_MODULE_H__ */