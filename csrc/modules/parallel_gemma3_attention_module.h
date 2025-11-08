#ifndef __MUILLM_PARALLEL_GEMMA3_ATTENTION_MODULE_H__
#define __MUILLM_PARALLEL_GEMMA3_ATTENTION_MODULE_H__

#include "../engine.h"
#include "../comms/comm_torch.h"

#include "parallel_linear_module.h"
#include "kvcache.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>

struct MuiLLMParallelGemma3Attention: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;

  MuiLLMParallelLinear* o_proj;

  int layer_index;

  int num_tp_heads;
  int num_tp_key_value_heads;
  int head_dim;

  torch::Tensor q_norm_weight;
  torch::Tensor k_norm_weight;

  float norm_epsilon;
  float norm_weights_offset;

  // methods
  MuiLLMParallelGemma3Attention(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    MuiLLMParallelLinear* o_proj,
    int num_tp_heads,
    int num_tp_key_value_heads,
    int head_dim,
    torch::Tensor& q_norm_weight,
    torch::Tensor& k_norm_weight,
    float norm_epsilon,
    float norm_weights_offset,
    int layer_index
  );

  torch::Tensor rope_forward(
    MuillmKVCache* cache,
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& m,
    torch::Tensor& cos,
    torch::Tensor& sin,
    torch::Tensor& cache_positions
  );

  torch::Tensor forward(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& m
  );
};


// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_parallel_gemma3_attention_module_ptr {
  MuiLLMParallelGemma3Attention* ptr;
} muillm_parallel_gemma3_attention_module_ptr_t;

// init
muillm_parallel_gemma3_attention_module_ptr_t muillm_parallel_gemma3_attention_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_linear_module_ptr_t o_proj,
  int num_tp_heads,
  int num_tp_key_value_heads,
  int head_dim,
  torch::Tensor& q_norm_weight,
  torch::Tensor& k_norm_weight,
  float norm_epsilon,
  float norm_weights_offset,
  int layer_index
);

// deinit
void muillm_parallel_gemma3_attention_module_deinit_trampoline(
  muillm_parallel_gemma3_attention_module_ptr_t module_ptr
);

// forward
at::Tensor muillm_parallel_gemma3_attention_module_forward_trampoline(
  muillm_parallel_gemma3_attention_module_ptr_t module_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& m
);

at::Tensor muillm_parallel_gemma3_attention_module_rope_forward_trampoline(
  muillm_parallel_gemma3_attention_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& m,
  torch::Tensor& cos,
  torch::Tensor& sin,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_PARALLEL_GEMMA3_ATTENTION_MODULE_H__ */