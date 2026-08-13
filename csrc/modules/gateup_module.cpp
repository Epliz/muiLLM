#include "gateup_module.h"

#include "../ffn/gateup.cuh"
#include "../linear/activation.h"

static inline mui_activation to_linear_activation(MuiGateUpMLPActivation activation) {
  switch (activation) {
    case MuiGateUpMLPActivation::SILU:
      return mui_activation::Silu;
    case MuiGateUpMLPActivation::GELU_TANH:
      return mui_activation::Gelu_Tanh;
    default:
      TORCH_CHECK(false, "Unsupported activation");
  }
}

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

  mui_activation linear_activation = to_linear_activation(activation);

  auto undef_tensor = torch::Tensor();
  this->gate_linear = new MuiLLMLinear(
    engine,
    norm_weights,
    gate_weights,
    /* add_bias */ undef_tensor,
    variance_epsilon,
    norm_weights_offset,
    /* activation */ linear_activation
  );
  this->up_linear = new MuiLLMLinear(
    engine,
    norm_weights,
    up_weights,
    /* add_bias */ undef_tensor,
    variance_epsilon,
    norm_weights_offset,
    /* activation */ mui_activation::Identity
  );
  this->down_linear = new MuiLLMLinear(
    engine,
    undef_tensor,
    down_weights,
    /* add_bias */ undef_tensor,
    0.f,  // variance_epsilon
    0.f,  // norm_weights_offset
    /* activation */ mui_activation::Identity
  );

  auto wdtype = gate_weights.dtype();
  bool dispatchable_type = (wdtype == torch::kFloat16) || (wdtype == torch::kBFloat16);
  bool dispatchable_device = gate_weights.device().is_cuda();
  this->dispatchable = dispatchable_device && dispatchable_type;
}

MuiLLMGateUpDownMLP::~MuiLLMGateUpDownMLP() {
  delete this->gate_linear;
  delete this->up_linear;
  delete this->down_linear;
}

torch::Tensor MuiLLMGateUpDownMLP::forward(
  torch::Tensor& inputs,
  torch::Tensor& residual
) {
  if (!this->dispatchable) {
    TORCH_CHECK(false, "MuiLLMGateUpDownMLP not dispatchable");
  }

  if (this->method == gateupmlp_FUSED) {
    auto numelements = inputs.numel();
    auto K = inputs.size(inputs.dim() - 1);

    if (numelements > (MUILLM_GATEUP_KERNELS_MAX_BATCH_SIZE * K)) {
      // cannot use the fused kernels, so fallback to the linear modules
      auto undef_tensor = torch::Tensor();
      auto gate_proj = this->gate_linear->forward(
        inputs,
        /* mul_residual */ undef_tensor,
        /* residual */ undef_tensor
      );

      auto up_proj = this->up_linear->forward(
        inputs,
        /* mul_residual */ gate_proj,
        /* residual */ undef_tensor
      );

      auto down_proj = this->down_linear->forward(
        up_proj,
        /* mul_residual */ undef_tensor,
        /* residual */ residual
      );
      return down_proj;
    } else {
      // can use the fused kernels
      return muillm_gateupmlp_forward(
        this->engine,
        this->activation,
        this->gate_linear->norm_weights,
        this->gate_linear->variance_epsilon,
        this->gate_linear->norm_weights_offset,
        this->gate_linear->weights,
        this->up_linear->weights,
        this->down_linear->weights,
        residual,
        inputs
      );
    }

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