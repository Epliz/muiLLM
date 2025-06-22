#include "linear_module.h"

#include "../linear/linear.cuh"
#include "../norm/rmsnorm.cuh"
#include "../comm_torch.h"

//
// actual module
//

MuiLLMLinear::MuiLLMLinear(
  muillm_engine_t* engine,
  torch::Tensor& norm_weights,
  torch::Tensor& weights,
  torch::Tensor& mul_bias,
  torch::Tensor& add_bias,
  float variance_epsilon
) {
  this->engine = engine;

  // we don't register as parameter in case it duplicates the memory
  this->norm_weights = norm_weights;
  this->weights = weights;
  this->mul_bias = mul_bias;
  this->add_bias = add_bias;

  this->variance_epsilon = variance_epsilon;

  auto wdtype = weights.dtype();
  bool dispatchable_type = (wdtype == torch::kFloat16) || (wdtype == torch::kBFloat16);
  bool dispatchable_device = weights.device().is_cuda();
  this->dispatchable = dispatchable_type && dispatchable_device;
}

MuiLLMLinear::~MuiLLMLinear() {
  // nothing to do
}

torch::Tensor MuiLLMLinear::forward(
    torch::Tensor& inputs,
    torch::Tensor& residual
) {
  // TODO: is numel slow?
  auto num_elements = inputs.numel();
  if (this->dispatchable && num_elements == inputs.size(inputs.dim() - 1)) {
    return muillm_linear_activ_forward(
      this->engine,
      this->norm_weights,
      this->variance_epsilon,
      this->weights,
      /* activ */ mui_activation::Identity,
      this->mul_bias,
      this->add_bias,
      residual,
      inputs
    );
  } else {
    // normalize if needed
    torch::Tensor normalized_inputs;
    if (this->norm_weights.defined()) {
      normalized_inputs = muillm_rmsnorm_forward(
        this->norm_weights,
        inputs,
        this->variance_epsilon
      );
    } else {
      normalized_inputs = inputs;
    }

    // linear
    auto output = torch::nn::functional::linear(normalized_inputs, this->weights, this->add_bias);

    // residual
    if (residual.defined()) {
      output = output + residual;
    }

    return output;
  }
}

//
// Python trampolines
//

// init
muillm_linear_module_ptr_t muillm_linear_module_init_trampoline(
  muillm_engine_ptr engine,
  torch::Tensor weights,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  std::optional<torch::Tensor> mul_bias_,
  std::optional<torch::Tensor> add_bias_) {

  auto undef_tensor = torch::Tensor();

  torch::Tensor& norm_weights = norm_weights_.has_value() ? norm_weights_.value() : undef_tensor;
  torch::Tensor& mul_bias = mul_bias_.has_value() ? mul_bias_.value() : undef_tensor;
  torch::Tensor& add_bias = add_bias_.has_value() ? add_bias_.value() : undef_tensor;

  MuiLLMLinear* m = new MuiLLMLinear(
    engine.engine_ptr,
    norm_weights,
    weights,
    mul_bias,
    add_bias,
    epsilon
  );

  muillm_linear_module_ptr_t ret;
  ret.ptr = m;
  return ret;
}

// deinit
void muillm_linear_module_deinit_trampoline(
  muillm_linear_module_ptr_t module_ptr) {
  delete module_ptr.ptr;
}

// forward
at::Tensor muillm_linear_module_forward_trampoline(
  muillm_linear_module_ptr_t module_ptr,
  torch::Tensor& inputs,
  std::optional<torch::Tensor> residual_) {

  auto undef_tensor = torch::Tensor();
  torch::Tensor& residual = residual_.has_value() ? residual_.value() : undef_tensor;
  
  MuiLLMLinear* m = module_ptr.ptr;

  return m->forward(inputs, residual);
}
