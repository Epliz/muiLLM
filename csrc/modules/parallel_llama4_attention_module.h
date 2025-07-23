#ifndef __MUILLM_PARALLEL_LLAMA4_ATTENTION_MODULE_H__
#define __MUILLM_PARALLEL_LLAMA4_ATTENTION_MODULE_H__

#include "../engine.h"
#include "../comms/comm_torch.h"

#include "parallel_linear_module.h"
#include "kvcache.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>

struct MuiLLMParallelLlama4Attention: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;

  MuiLLMParallelLinear* o_proj;

  int layer_index;
  
  int num_tp_heads;
  int num_tp_key_value_heads;
  int head_dim;

  bool use_rope;

  bool use_qk_norm;
  float norm_epsilon;

  bool use_temperature_tuning;
  float attention_scale;
  float floor_scale;

  // methods
  MuiLLMParallelLlama4Attention(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    MuiLLMParallelLinear* o_proj,
    int num_tp_heads,
    int num_tp_key_value_heads,
    int head_dim,
    bool use_rope,
    bool use_qk_norm,
    float norm_epsilon,
    bool use_temperature_tuning,
    float attention_scale,
    float floor_scale,
    int layer_index
  );

  torch::Tensor rope_forward(
    MuillmKVCache* cache,
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& m,
    torch::Tensor& residual,
    torch::Tensor& position_embeds,
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
typedef struct muillm_parallel_llama4_attention_module_ptr {
  MuiLLMParallelLlama4Attention* ptr;
} muillm_parallel_llama4_attention_module_ptr_t;

// init
muillm_parallel_llama4_attention_module_ptr_t muillm_parallel_llama4_attention_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_linear_module_ptr_t o_proj,
  int num_tp_heads,
  int num_tp_key_value_heads,
  int head_dim,
  bool use_rope,
  bool use_qk_norm,
  float norm_epsilon,
  bool use_temperature_tuning,
  float attention_scale,
  float floor_scale,
  int layer_index
);

// deinit
void muillm_parallel_llama4_attention_module_deinit_trampoline(
  muillm_parallel_llama4_attention_module_ptr_t module_ptr
);

// forward
at::Tensor muillm_parallel_llama4_attention_module_forward_trampoline(
  muillm_parallel_llama4_attention_module_ptr_t module_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& m,
  std::optional<torch::Tensor>& residual
);

at::Tensor muillm_parallel_llama4_attention_module_rope_forward_trampoline(
  muillm_parallel_llama4_attention_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& m,
  torch::Tensor& residual,
  torch::Tensor& position_embeds,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_PARALLEL_LLAMA4_ATTENTION_MODULE_H__ */