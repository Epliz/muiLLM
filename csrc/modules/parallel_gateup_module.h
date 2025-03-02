#ifndef __MUILLM_PARALLEL_GATEUPDOWN_MODULE_H__
#define __MUILLM_PARALLEL_GATEUPDOWN_MODULE_H__


#include "../engine.h"
#include "../comm_torch.h"


#include <torch/torch.h>

enum MuiLLMGateUpSiluMethod {
    // Basic method where Gate/Up projections + mul are done distinctly
    GATEUPSILU_UNFUSED = 0,
    // Method where the Gate/Up projections + mul are all fused
    GATEUPSILU_FUSED = 1,
    // Method where the Gate/Up projections are done in the same kernel
    // but split between blocks to have more blocks.
    // A final reduction is done in an epilogue kernel
    GATEUPSILU_SPLIT = 2
};

struct MuiLLMParallelGateUpDownMLP: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;

  MuiLLMGateUpSiluMethod method;
  
  torch::Tensor norm_weights{nullptr};
  torch::Tensor gate_weights{nullptr};
  torch::Tensor up_weights{nullptr};
  torch::Tensor down_weights{nullptr};

  float variance_epsilon;

  bool dispatchable;

  // methods
  MuiLLMParallelGateUpDownMLP(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    int method,
    torch::Tensor& norm_weights,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    float variance_epsilon
  );

  virtual ~MuiLLMParallelGateUpDownMLP();

  torch::Tensor forward(
    torch::Tensor& inputs,
    torch::Tensor& residual
  );
};


// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_parallel_gateupdownmlp_module_ptr {
  MuiLLMParallelGateUpDownMLP* ptr;
} muillm_parallel_gateupdownmlp_module_ptr_t;

// init
muillm_parallel_gateupdownmlp_module_ptr_t muillm_parallel_gateupdownmlp_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  int method,
  std::optional<torch::Tensor>& norm_weights,
  torch::Tensor& gate_weights,
  torch::Tensor& up_weights,
  torch::Tensor& down_weights,
  float variance_epsilon
);

// deinit
void muillm_parallel_gateupdownmlp_module_deinit_trampoline(
  muillm_parallel_gateupdownmlp_module_ptr_t module_ptr
);

// forward
at::Tensor muillm_parallel_gateupdownmlp_module_forward_trampoline(
  muillm_parallel_gateupdownmlp_module_ptr_t module_ptr,
  torch::Tensor& inputs,
  std::optional<torch::Tensor> residual_
);

#endif /* __MUILLM_PARALLEL_GATEUPDOWN_MODULE_H__ */