#ifndef __MUILLM_PARALLEL_GATEUPDOWN_MODULE_H__
#define __MUILLM_PARALLEL_GATEUPDOWN_MODULE_H__

#include "parallel_gateup_module_interface.h"

#include "../engine.h"
#include "../comms/comm_torch.h"

#include "../ffn/gateupmlpactivation.h"

enum MuiLLMgateupmlpMethod {
    // Basic method where Gate/Up projections + mul are done distinctly
    gateupmlp_UNFUSED = 0,
    // Method where the Gate/Up projections + mul are all fused
    gateupmlp_FUSED = 1,
    // Method where the Gate/Up projections are done in the same kernel
    // but split between blocks to have more blocks.
    // A final reduction is done in an epilogue kernel
    gateupmlp_SPLIT = 2
};

struct MuiLLMParallelGateUpDownMLP: MuiLLMParallelGateUpDownMLPInterface {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;

  MuiGateUpMLPActivation activation;
  MuiLLMgateupmlpMethod method;
  
  torch::Tensor norm_weights{nullptr};
  torch::Tensor gate_weights{nullptr};
  torch::Tensor up_weights{nullptr};
  torch::Tensor down_weights{nullptr};

  float variance_epsilon;
  float norm_weights_offset;

  bool dispatchable;

  // methods
  MuiLLMParallelGateUpDownMLP(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    MuiGateUpMLPActivation activation,
    int method,
    torch::Tensor& norm_weights,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    float variance_epsilon,
    float norm_weights_offset
  );

  virtual ~MuiLLMParallelGateUpDownMLP();

  // @override
  torch::Tensor forward(
    torch::Tensor& inputs,
    torch::Tensor& residual,
    bool reduce
  );
};

// init
muillm_parallel_igateupdownmlp_module_ptr_t muillm_parallel_gateupdownmlp_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
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
void muillm_parallel_gateupdownmlp_module_deinit_trampoline(
  muillm_parallel_igateupdownmlp_module_ptr_t module_ptr
);

// forward
at::Tensor muillm_parallel_gateupdownmlp_module_forward_trampoline(
  muillm_parallel_igateupdownmlp_module_ptr_t module_ptr,
  torch::Tensor& inputs,
  std::optional<torch::Tensor> residual_,
  bool reduce
);

#endif /* __MUILLM_PARALLEL_GATEUPDOWN_MODULE_H__ */