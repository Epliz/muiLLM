#ifndef __MUILLM_PARALLEL_LINEAR_MODULE_H__
#define __MUILLM_PARALLEL_LINEAR_MODULE_H__

#include "../engine.h"
#include "../comm_torch.h"


#include <torch/torch.h>

struct MuiLLMParallelLinear: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;
  
  torch::Tensor norm_weights{nullptr};
  torch::Tensor weights{nullptr};

  torch::Tensor mul_bias{nullptr};
  torch::Tensor add_bias{nullptr};

  float variance_epsilon;
  int sharding_dim;

  bool dispatchable;

  // methods
  MuiLLMParallelLinear(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    torch::Tensor& norm_weights,
    torch::Tensor& weights,
    torch::Tensor& mul_bias,
    torch::Tensor& add_bias,
    float variance_epsilon,
    int sharding_dim
  );

  torch::Tensor forward(
    torch::Tensor& inputs,
    torch::Tensor& residual,
    bool collect_outputs
  );
};

// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_parallel_linear_module_ptr {
  MuiLLMParallelLinear* ptr;
} muillm_parallel_linear_module_ptr_t;

// init
muillm_parallel_linear_module_ptr_t muillm_parallel_linear_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  torch::Tensor weights,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  std::optional<torch::Tensor> mul_bias_,
  std::optional<torch::Tensor> add_bias_,
  int sharding_dim
);

// deinit
void muillm_parallel_linear_module_deinit_trampoline(
  muillm_parallel_linear_module_ptr_t module_ptr
);

// forward
at::Tensor muillm_parallel_linear_module_forward_trampoline(
  muillm_parallel_linear_module_ptr_t module_ptr,
  torch::Tensor& inputs,
  std::optional<torch::Tensor> residual_,
  bool reduce
);

#endif /* __MUILLM_PARALLEL_LINEAR_MODULE_H__ */