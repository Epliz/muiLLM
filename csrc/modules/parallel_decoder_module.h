#ifndef __MUILLM_PARALLEL_DECODER_MODULE_H__
#define __MUILLM_PARALLEL_DECODER_MODULE_H__


#include "../engine.h"
#include "../comm_torch.h"

#include "parallel_multilinear_module.h"
#include "parallel_attention_module.h"
#include "parallel_gateup_module.h"
#include "kvcache.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>

struct MuiLLMParallelDecoder: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;

  MuiLLMParallelMultiLinear* multilinear;
  MuiLLMParallelAttention* attention;
  MuiLLMParallelGateUpDownMLP* mlp;

  // methods
  MuiLLMParallelDecoder(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    MuiLLMParallelMultiLinear* multilinear,
    MuiLLMParallelAttention* attention,
    MuiLLMParallelGateUpDownMLP* mlp
  );

  virtual ~MuiLLMParallelDecoder();

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
typedef struct muillm_parallel_decoder_module_ptr {
  MuiLLMParallelDecoder* ptr;
} muillm_parallel_decoder_module_ptr_t;

muillm_parallel_decoder_module_ptr_t muillm_parallel_decoder_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_multilinear_module_ptr_t multilinear,
  muillm_parallel_attention_module_ptr_t attention,
  muillm_parallel_gateupdownmlp_module_ptr_t mlp
);

void muillm_parallel_decoder_module_deinit_trampoline(
  muillm_parallel_decoder_module_ptr_t module_ptr
);

at::Tensor muillm_parallel_decoder_module_forward(
  muillm_parallel_decoder_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& m,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_PARALLEL_DECODER_MODULE_H__ */