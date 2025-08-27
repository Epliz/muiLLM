#ifndef __MUILLM_ATTENTION_MODULE_H__
#define __MUILLM_ATTENTION_MODULE_H__

#include "../engine.h"

#include "linear_module.h"
#include "kvcache.h"
#include "rotary_module.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>

struct MuiLLMAttention: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  
  MuillmRotaryEmbedding* rotary;
  MuiLLMLinear* o_proj;

  int num_heads;
  int num_key_value_heads;
  int head_dim;

  // methods
  MuiLLMAttention(
    muillm_engine_t* engine,
    MuillmRotaryEmbedding* rotary,
    MuiLLMLinear* o_proj,
    int num_heads,
    int num_key_value_heads,
    int head_dim
  );

  torch::Tensor rope_forward(
    MuillmKVCache* cache,
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& m,
    torch::Tensor& residual,
    torch::Tensor& position_ids,
    std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
    torch::Tensor& cache_positions
  );

  torch::Tensor forward(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& m,
    torch::Tensor& residual
  );
};


// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_attention_module_ptr {
  MuiLLMAttention* ptr;
} muillm_attention_module_ptr_t;

// init
muillm_attention_module_ptr muillm_attention_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_rotary_embedding_module_ptr_t rotary,
  muillm_linear_module_ptr_t o_proj,
  int num_heads,
  int num_key_value_heads,
  int head_dim
);

// deinit
void muillm_attention_module_deinit_trampoline(
  muillm_attention_module_ptr_t module_ptr
);

// forward
at::Tensor muillm_attention_module_forward_trampoline(
  muillm_attention_module_ptr_t module_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& m,
  std::optional<torch::Tensor>& residual
);

at::Tensor muillm_attention_module_rope_forward_trampoline(
  muillm_attention_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& m,
  torch::Tensor& residual,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_ATTENTION_MODULE_H__ */