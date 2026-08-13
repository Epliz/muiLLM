#include "linear_module.h"

#include "../linear/linear.cuh"
#include "../norm/rmsnorm.cuh"
#include "../comms/comm_torch.h"

//
// actual module
//

MuiLLMLinear::MuiLLMLinear(
  muillm_engine_t* engine,
  torch::Tensor& norm_weights,
  torch::Tensor& weights,
  torch::Tensor& add_bias,
  float variance_epsilon,
  float norm_weights_offset,
  mui_activation activation
) {
  this->engine = engine;

  // we don't register as parameter in case it duplicates the memory
  this->norm_weights = norm_weights;
  this->weights = weights;
  this->add_bias = add_bias;

  this->variance_epsilon = variance_epsilon;
  this->norm_weights_offset = norm_weights_offset;

  this->activation = activation;

  auto wdtype = weights.dtype();
  bool dispatchable_type = (wdtype == torch::kFloat16) || (wdtype == torch::kBFloat16);
  bool dispatchable_device = weights.device().is_cuda();
  this->dispatchable = dispatchable_type && dispatchable_device;
}

MuiLLMLinear::~MuiLLMLinear() {
  // nothing to do
}

torch::Tensor MuiLLMLinear::forward(
    torch::Tensor& inputs, // B x K
    torch::Tensor& mul_residual, // B x N
    torch::Tensor& residual
) {
  auto undef_tensor = torch::Tensor();

  // our custom kernels can only hand specific batch sizes, so check if suitable
  auto num_elements = inputs.numel();
  auto K = inputs.size(inputs.dim() - 1);

  if (this->dispatchable && num_elements <= (MUILLM_LINEAR_KERNELS_MAX_BATCH_SIZE * K)) {
    return muillm_linear_activ_forward(
      this->engine,
      this->norm_weights,
      this->variance_epsilon,
      this->norm_weights_offset,
      this->weights,
      this->activation,
      this->add_bias,
      mul_residual,
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
        /* residual */ undef_tensor,
        this->variance_epsilon,
        this->norm_weights_offset
      );
    } else {
      normalized_inputs = inputs;
    }

    // linear
    auto output = torch::nn::functional::linear(normalized_inputs, this->weights, this->add_bias);

    if (this->activation == mui_activation::Silu) {
      output = torch::silu(output);
    } else if (this->activation == mui_activation::Gelu_Tanh) {
      output = torch::gelu(output, /* approximate */ "tanh");
    } else if (this->activation != mui_activation::Identity) {
      TORCH_CHECK(false, "Unsupported activation");
    }

    // mul residual
    if (mul_residual.defined()) {
      output = output * mul_residual;
    }
  
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
  float norm_weights_offset,
  std::optional<torch::Tensor> add_bias_) {

  auto undef_tensor = torch::Tensor();

  torch::Tensor& norm_weights = norm_weights_.has_value() ? norm_weights_.value() : undef_tensor;
  torch::Tensor& add_bias = add_bias_.has_value() ? add_bias_.value() : undef_tensor;

  MuiLLMLinear* m = new MuiLLMLinear(
    engine.engine_ptr,
    norm_weights,
    weights,
    add_bias,
    epsilon,
    norm_weights_offset
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
  std::optional<torch::Tensor> mul_residual_,
  std::optional<torch::Tensor> residual_) {

  auto undef_tensor = torch::Tensor();
  torch::Tensor& mul_residual = mul_residual_.has_value() ? mul_residual_.value() : undef_tensor;
  torch::Tensor& residual = residual_.has_value() ? residual_.value() : undef_tensor;
  
  MuiLLMLinear* m = module_ptr.ptr;

  return m->forward(inputs, mul_residual, residual);
}
