#ifndef __MUILLM_PARALLEL_GATEUPDOWN_MODULE_INTERFACE_H__
#define __MUILLM_PARALLEL_GATEUPDOWN_MODULE_INTERFACE_H__


#include <torch/torch.h>

struct MuiLLMParallelGateUpDownMLPInterface: torch::nn::Module {

  // methods

  virtual ~MuiLLMParallelGateUpDownMLPInterface();

  virtual torch::Tensor forward(
    torch::Tensor& inputs,
    torch::Tensor& residual,
    bool reduce
  ) = 0;
};

// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_parallel_igateupdownmlp_module_ptr {
  MuiLLMParallelGateUpDownMLPInterface* ptr;
} muillm_parallel_igateupdownmlp_module_ptr_t;

#endif /* __MUILLM_PARALLEL_GATEUPDOWN_MODULE_INTERFACE_H__ */