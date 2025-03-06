#ifndef __MUILLM_PARALLEL_MULTILINEAR_H__
#define __MUILLM_PARALLEL_MULTILINEAR_H__

#include "../engine.h"
#include "../comm_torch.h"

#include "parallel_linear_module.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>


struct MuiLLMParallelMultiLinear: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;

  MuiLLMParallelLinear* linear;
  std::vector<std::tuple<int, int>> slices;
  int sharding_dim;

  // methods
  MuiLLMParallelMultiLinear(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    MuiLLMParallelLinear* linear,
    std::vector<std::tuple<int, int>>& slices,
    int sharding_dim
  );

  virtual ~MuiLLMParallelMultiLinear();

  std::vector<torch::Tensor> forward(
    torch::Tensor& input,
    bool collect_outputs
  );

  std::vector<torch::Tensor> slice_outputs(
    torch::Tensor& all_outputs
  );

  torch::Tensor collect_output(
    torch::Tensor& output
  );
};


// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_parallel_multilinear_module_ptr {
  MuiLLMParallelMultiLinear* ptr;
} muillm_parallel_multilinear_module_ptr_t;

muillm_parallel_multilinear_module_ptr_t muillm_parallel_multilinear_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_linear_module_ptr_t linear,
  std::vector<std::tuple<int, int>> slices,
  int sharding_dim
);

void muillm_parallel_multilinear_module_deinit_trampoline(
  muillm_parallel_multilinear_module_ptr_t ptr
);

std::vector<torch::Tensor> muillm_parallel_multilinear_module_forward_trampoline(
  muillm_parallel_multilinear_module_ptr_t ptr,
  torch::Tensor input,
  bool collect_outputs
);

#endif /* __MUILLM_PARALLEL_MULTILINEAR_H__ */