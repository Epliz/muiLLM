#ifndef __MUILLM_LINEAR_MODULE_H__
#define __MUILLM_LINEAR_MODULE_H__

#include "../engine.h"

#include <optional>

#include <torch/torch.h>

struct MuiLLMLinear: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  
  torch::Tensor norm_weights{nullptr};
  torch::Tensor weights{nullptr};

  torch::Tensor mul_bias{nullptr};
  torch::Tensor add_bias{nullptr};

  float variance_epsilon;
  float norm_weights_offset;

  bool dispatchable;

  // methods
  MuiLLMLinear(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    torch::Tensor& weights,
    torch::Tensor& mul_bias,
    torch::Tensor& add_bias,
    float variance_epsilon,
    float norm_weights_offset
  );

  virtual ~MuiLLMLinear();

  torch::Tensor forward(
    torch::Tensor& inputs,
    torch::Tensor& residual
  );
};

// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_linear_module_ptr {
  MuiLLMLinear* ptr;
} muillm_linear_module_ptr_t;

// init
muillm_linear_module_ptr_t muillm_linear_module_init_trampoline(
  muillm_engine_ptr engine,
  torch::Tensor weights,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  float norm_weights_offset,
  std::optional<torch::Tensor> mul_bias_,
  std::optional<torch::Tensor> add_bias_
);

// deinit
void muillm_linear_module_deinit_trampoline(
  muillm_linear_module_ptr_t module_ptr
);

// forward
at::Tensor muillm_linear_module_forward_trampoline(
  muillm_linear_module_ptr_t module_ptr,
  torch::Tensor& inputs,
  std::optional<torch::Tensor> residual_
);

#endif /* __MUILLM_LINEAR_MODULE_H__ */