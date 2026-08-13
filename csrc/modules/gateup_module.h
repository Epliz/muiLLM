#ifndef __MUILLM_GATEUPDOWN_MODULE_H__
#define __MUILLM_GATEUPDOWN_MODULE_H__

#include "gateup_module_interface.h"

#include "../engine.h"

#include "../ffn/gateupmlpactivation.h"
#include "gateup_method.h"
#include "linear_module.h"

struct MuiLLMGateUpDownMLP: MuiLLMGateUpDownMLPInterface {
  // fields
  muillm_engine_t* engine;

  MuiGateUpMLPActivation activation;
  MuiLLMgateupmlpMethod method;

  MuiLLMLinear* gate_linear{nullptr};
  MuiLLMLinear* up_linear{nullptr};
  MuiLLMLinear* down_linear{nullptr};


  float variance_epsilon;
  float norm_weights_offset;

  bool dispatchable;

  // methods
  MuiLLMGateUpDownMLP(
    muillm_engine_t* engine,
    MuiGateUpMLPActivation activation,
    int method,
    torch::Tensor& norm_weights,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    float variance_epsilon,
    float norm_weights_offset
  );

  virtual ~MuiLLMGateUpDownMLP();

  // @override
  torch::Tensor forward(
    torch::Tensor& inputs,
    torch::Tensor& residual
  );
};

// init
muillm_igateupdownmlp_module_ptr_t muillm_gateupdownmlp_module_init_trampoline(
  muillm_engine_ptr engine,
  int activation,
  int method,
  std::optional<torch::Tensor>& norm_weights,
  torch::Tensor& gate_weights,
  torch::Tensor& up_weights,
  torch::Tensor& down_weights,
  float variance_epsilon,
  float norm_weights_offset
);

// deinit
void muillm_gateupdownmlp_module_deinit_trampoline(
  muillm_igateupdownmlp_module_ptr_t module_ptr
);

// forward
at::Tensor muillm_gateupdownmlp_module_forward_trampoline(
  muillm_igateupdownmlp_module_ptr_t module_ptr,
  torch::Tensor& inputs,
  std::optional<torch::Tensor> residual_
);

#endif /* __MUILLM_GATEUPDOWN_MODULE_H__ */