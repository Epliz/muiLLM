#include "gateup_module.h"

#include "../ffn/gateup.cuh"

MuiLLMGateUpDownMLP::MuiLLMGateUpDownMLP(
  muillm_engine_t* engine,
  MuiGateUpMLPActivation activation,
  int method,
  torch::Tensor& norm_weights,
  torch::Tensor& gate_weights,
  torch::Tensor& up_weights,
  torch::Tensor& down_weights,
  float variance_epsilon,
  float norm_weights_offset
) {
  this->engine = engine;
  this->method = static_cast<MuiLLMgateupmlpMethod>(method);
  this->activation = activation;

  this->norm_weights = norm_weights;
  this->gate_weights = gate_weights;
  this->up_weights = up_weights;
  this->down_weights = down_weights;

  this->variance_epsilon = variance_epsilon;
  this->norm_weights_offset = norm_weights_offset;

  auto wdtype = gate_weights.dtype();
  bool dispatchable_type = (wdtype == torch::kFloat16) || (wdtype == torch::kBFloat16);
  bool dispatchable_device = gate_weights.device().is_cuda();
  this->dispatchable = dispatchable_device && dispatchable_type;
}

MuiLLMGateUpDownMLP::~MuiLLMGateUpDownMLP() {
  // nothing to do
}

torch::Tensor MuiLLMGateUpDownMLP::forward(
  torch::Tensor& inputs,
  torch::Tensor& residual
) {
  if (!this->dispatchable) {
    TORCH_CHECK(false, "MuiLLMGateUpDownMLP not dispatchable");
  }

  if (this->method == gateupmlp_FUSED) {
    return muillm_gateupmlp_forward(
      this->engine,
      this->activation,
      this->norm_weights,
      this->variance_epsilon,
      this->norm_weights_offset,
      this->gate_weights,
      this->up_weights,
      this->down_weights,
      residual,
      inputs
    );
  } else {
    TORCH_CHECK(false, "Unsupported method");
  }
}

muillm_igateupdownmlp_module_ptr_t muillm_gateupdownmlp_module_init_trampoline(
  muillm_engine_ptr engine,
  int activation,
  int method,
  std::optional<torch::Tensor>& norm_weights_,
  torch::Tensor& gate_weights,
  torch::Tensor& up_weights,
  torch::Tensor& down_weights,
  float variance_epsilon,
  float norm_weights_offset
) {
  
  auto undef_tensor = torch::Tensor();

  torch::Tensor& norm_weights = norm_weights_.has_value() ? norm_weights_.value() : undef_tensor;

  MuiLLMGateUpDownMLP* mlp_module = new MuiLLMGateUpDownMLP(
    engine.engine_ptr,
    static_cast<MuiGateUpMLPActivation>(activation),
    method,
    norm_weights,
    gate_weights,
    up_weights,
    down_weights,
    variance_epsilon,
    norm_weights_offset
  );

  muillm_igateupdownmlp_module_ptr_t module_ptr;
  module_ptr.ptr = mlp_module;
  return module_ptr;
}

void muillm_gateupdownmlp_module_deinit_trampoline(
  muillm_igateupdownmlp_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

at::Tensor muillm_gateupdownmlp_module_forward_trampoline(
    muillm_igateupdownmlp_module_ptr_t module_ptr,
    torch::Tensor& inputs,
    std::optional<torch::Tensor> residual_
) {
  auto undef_tensor = torch::Tensor();
  torch::Tensor& residual = residual_.has_value() ? residual_.value() : undef_tensor;

  return ((MuiLLMGateUpDownMLP*)module_ptr.ptr)->forward(inputs, residual);
}