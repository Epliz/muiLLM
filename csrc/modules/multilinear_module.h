#ifndef __MUILLM_MULTILINEAR_H__
#define __MUILLM_MULTILINEAR_H__

#include "../engine.h"

#include "linear_module.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>


struct MuiLLMMultiLinear: torch::nn::Module {
  // fields
  muillm_engine_t* engine;

  MuiLLMLinear* linear;
  std::vector<std::tuple<int, int>> slices;

  // methods
  MuiLLMMultiLinear(
    muillm_engine_t* engine,
    MuiLLMLinear* linear,
    std::vector<std::tuple<int, int>>& slices
  );

  virtual ~MuiLLMMultiLinear();

  std::vector<torch::Tensor> forward(
    torch::Tensor& input
  );

  std::vector<torch::Tensor> slice_outputs(
    torch::Tensor& all_outputs
  );
};


// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_multilinear_module_ptr {
  MuiLLMMultiLinear* ptr;
} muillm_multilinear_module_ptr_t;

muillm_multilinear_module_ptr_t muillm_multilinear_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_linear_module_ptr_t linear,
  std::vector<std::tuple<int, int>> slices
);

void muillm_multilinear_module_deinit_trampoline(
  muillm_multilinear_module_ptr_t ptr
);

std::vector<torch::Tensor> muillm_multilinear_module_forward_trampoline(
  muillm_multilinear_module_ptr_t ptr,
  torch::Tensor input
);

#endif /* __MUILLM_MULTILINEAR_H__ */