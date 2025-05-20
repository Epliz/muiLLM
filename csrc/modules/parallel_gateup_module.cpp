#include "parallel_gateup_module.h"

#include "../parallel_gateup_kernels.cuh"

MuiLLMParallelGateUpDownMLP::MuiLLMParallelGateUpDownMLP(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  int method,
  torch::Tensor& norm_weights,
  torch::Tensor& gate_weights,
  torch::Tensor& up_weights,
  torch::Tensor& down_weights,
  float variance_epsilon
) {
  this->engine = engine;
  this->comm = comm;
  this->method = static_cast<MuiLLMGateUpSiluMethod>(method);

  this->norm_weights = norm_weights;
  this->gate_weights = gate_weights;
  this->up_weights = up_weights;
  this->down_weights = down_weights;

  this->variance_epsilon = variance_epsilon;

  auto wdtype = gate_weights.dtype();
  bool dispatchable_type = (wdtype == at::kHalf);
  bool dispatchable_device = gate_weights.device().is_cuda();
  this->dispatchable = dispatchable_device && dispatchable_type;
}

MuiLLMParallelGateUpDownMLP::~MuiLLMParallelGateUpDownMLP() {
  // nothing to do
}

torch::Tensor MuiLLMParallelGateUpDownMLP::forward(
  torch::Tensor& inputs,
  torch::Tensor& residual,
  bool reduce
) {
  if (!this->dispatchable) {
    TORCH_CHECK(false, "MuiLLMParallelGateUpDownMLP not dispatchable");
  }

  if (this->method == GATEUPSILU_FUSED) {
    return muillm_parallel_gateupsilu_forward(
      this->engine,
      this->comm,
      this->norm_weights,
      this->variance_epsilon,
      this->gate_weights,
      this->up_weights,
      this->down_weights,
      residual,
      inputs,
      reduce
    );
  } else if (this->method == GATEUPSILU_SPLIT) {
    return muillm_parallel_gateupsilu_split_forward(
      this->engine,
      this->comm,
      this->norm_weights,
      this->variance_epsilon,
      this->gate_weights,
      this->up_weights,
      this->down_weights,
      residual,
      inputs,
      reduce
    );
  } else {
    TORCH_CHECK(false, "Unsupported method");
  }
}

muillm_parallel_gateupdownmlp_module_ptr_t muillm_parallel_gateupdownmlp_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  int method,
  std::optional<torch::Tensor>& norm_weights_,
  torch::Tensor& gate_weights,
  torch::Tensor& up_weights,
  torch::Tensor& down_weights,
  float variance_epsilon
) {
  
  auto undef_tensor = torch::Tensor();

  torch::Tensor& norm_weights = norm_weights_.has_value() ? norm_weights_.value() : undef_tensor;

  MuiLLMParallelGateUpDownMLP* mlp_module = new MuiLLMParallelGateUpDownMLP(
    engine.engine_ptr,
    comm.comm_ptr,
    method,
    norm_weights,
    gate_weights,
    up_weights,
    down_weights,
    variance_epsilon
  );

  muillm_parallel_gateupdownmlp_module_ptr_t module_ptr;
  module_ptr.ptr = mlp_module;
  return module_ptr;
}

void muillm_parallel_gateupdownmlp_module_deinit_trampoline(
  muillm_parallel_gateupdownmlp_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

at::Tensor muillm_parallel_gateupdownmlp_module_forward_trampoline(
    muillm_parallel_gateupdownmlp_module_ptr_t module_ptr,
    torch::Tensor& inputs,
    std::optional<torch::Tensor> residual_,
    bool reduce
) {
  auto undef_tensor = torch::Tensor();
  torch::Tensor& residual = residual_.has_value() ? residual_.value() : undef_tensor;

  return module_ptr.ptr->forward(inputs, residual, reduce);
}