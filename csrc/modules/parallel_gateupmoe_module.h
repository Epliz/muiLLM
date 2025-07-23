#ifndef __MUILLM_PARALLEL_GATEUPDOWNMOE_MODULE_H__
#define __MUILLM_PARALLEL_GATEUPDOWNMOE_MODULE_H__

#include "parallel_gateup_module_interface.h"

#include "../engine.h"
#include "../comms/comm_torch.h"

#include "linear_module.h"


struct MuiLLMParallelGateUpDownMLPMoE: MuiLLMParallelGateUpDownMLPInterface {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;

  MuiLLMLinear* router;
  
  torch::Tensor norm_weights{nullptr};
  torch::Tensor gate_weights{nullptr};
  torch::Tensor up_weights{nullptr};
  torch::Tensor down_weights{nullptr};

  float variance_epsilon;
  
  int num_shared_experts;
  int num_dynamic_experts;
  int num_routed_experts;

  bool dispatchable;

  // methods
  MuiLLMParallelGateUpDownMLPMoE(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    MuiLLMLinear* router,
    int num_shared_experts,
    int num_dynamic_experts,
    int num_routed_experts,
    torch::Tensor& norm_weights,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    float variance_epsilon
  );

  virtual ~MuiLLMParallelGateUpDownMLPMoE();

  // @override
  torch::Tensor forward(
    torch::Tensor& inputs,
    torch::Tensor& residual,
    bool reduce
  );
};

// init
muillm_parallel_igateupdownmlp_module_ptr_t muillm_parallel_gateupdownmlpmoe_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_linear_module_ptr_t router_module_ptr,
  int num_shared_experts,
  int num_dynamic_experts,
  int num_routed_experts,
  std::optional<torch::Tensor>& norm_weights,
  torch::Tensor& gate_weights,
  torch::Tensor& up_weights,
  torch::Tensor& down_weights,
  float variance_epsilon
);

// deinit
void muillm_parallel_gateupdownmlpmoe_module_deinit_trampoline(
  muillm_parallel_igateupdownmlp_module_ptr_t module_ptr
);

// forward
at::Tensor muillm_parallel_gateupdownmlpmoe_module_forward_trampoline(
  muillm_parallel_igateupdownmlp_module_ptr_t module_ptr,
  torch::Tensor& inputs,
  std::optional<torch::Tensor> residual_,
  bool reduce
);

#endif /* __MUILLM_PARALLEL_GATEUPDOWNMOE_MODULE_H__ */