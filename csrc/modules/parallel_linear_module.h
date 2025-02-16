#ifndef __MUILLM_PARALLEL_LINEAR_MODULE_H__
#define __MUILLM_PARALLEL_LINEAR_MODULE_H__

#include "../engine.h"
#include "../comm.h"

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

#endif /* __MUILLM_PARALLEL_LINEAR_MODULE_H__ */