#ifndef __MUILLM_PARALLEL_ATTENTION_MODULE_H__
#define __MUILLM_PARALLEL_ATTENTION_MODULE_H__

#include "../engine.h"
#include "../comm_torch.h"

#include "parallel_linear_module.h"

#include <torch/torch.h>

struct MuiLLMParallelAttention: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;
  
  MuiLLMParallelLinear* o_proj;

  // methods
  MuiLLMParallelAttention(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    MuiLLMParallelLinear* o_proj
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
typedef struct muillm_parallel_attention_module_ptr {
  MuiLLMParallelAttention* ptr;
} muillm_parallel_attention_module_ptr_t;

// init
muillm_parallel_attention_module_ptr muillm_parallel_attention_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_linear_module_ptr_t o_proj
);

// deinit
void muillm_parallel_attention_module_deinit_trampoline(
  muillm_parallel_attention_module_ptr_t module_ptr
);

// forward
at::Tensor muillm_parallel_attention_module_forward_trampoline(
  muillm_parallel_attention_module_ptr_t module_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& m,
  std::optional<torch::Tensor>& residual
);

#endif /* __MUILLM_PARALLEL_LINEAR_MODULE_H__ */