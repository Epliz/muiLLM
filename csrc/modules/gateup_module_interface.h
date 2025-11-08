#ifndef __MUILLM_GATEUPDOWN_MODULE_INTERFACE_H__
#define __MUILLM_GATEUPDOWN_MODULE_INTERFACE_H__


#include <torch/torch.h>

struct MuiLLMGateUpDownMLPInterface: torch::nn::Module {

  // methods

  virtual ~MuiLLMGateUpDownMLPInterface();

  virtual torch::Tensor forward(
    torch::Tensor& inputs,
    torch::Tensor& residual
  ) = 0;
};

// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_igateupdownmlp_module_ptr {
  MuiLLMGateUpDownMLPInterface* ptr;
} muillm_igateupdownmlp_module_ptr_t;

#endif /* __MUILLM_GATEUPDOWN_MODULE_INTERFACE_H__ */