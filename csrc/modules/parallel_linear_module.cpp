#include "parallel_linear_module.h"

#include "../parallel_linear_kernels.cuh"
#include "../norm/rmsnorm.cuh"
#include "../comm_torch.h"

//
// actual module
//

MuiLLMParallelLinear::MuiLLMParallelLinear(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  torch::Tensor& norm_weights,
  torch::Tensor& weights,
  torch::Tensor& mul_bias,
  torch::Tensor& add_bias,
  float variance_epsilon,
  int sharding_dim
) {
  this->engine = engine;
  this->comm = comm;

  // we don't register as parameter in case it duplicates the memory
  this->norm_weights = norm_weights;
  this->weights = weights;
  this->mul_bias = mul_bias;
  this->add_bias = add_bias;

  this->variance_epsilon = variance_epsilon;
  this->sharding_dim = sharding_dim;

  auto wdtype = weights.dtype();
  bool dispatchable_type = (wdtype == torch::kFloat16) || (wdtype == torch::kBFloat16);
  bool dispatchable_device = weights.device().is_cuda();
  this->dispatchable = dispatchable_type && dispatchable_device;
}

MuiLLMParallelLinear::~MuiLLMParallelLinear() {
  // nothing to do
}

torch::Tensor MuiLLMParallelLinear::forward(
    torch::Tensor& inputs,
    torch::Tensor& residual,
    bool collect_outputs
) {
  // TODO: is numel slow?
  auto num_elements = inputs.numel();
  if (this->dispatchable && num_elements == inputs.size(inputs.dim() - 1)) {
    return muillm_parallel_linear_activ_forward(
      this->engine,
      this->comm,
      this->norm_weights,
      this->variance_epsilon,
      this->weights,
      /* activ */ mui_activation::Identity,
      this->mul_bias,
      this->add_bias,
      residual,
      this->sharding_dim, // 0 for row-wise, 1 for column-wise
      /* reduce */ collect_outputs,
      inputs
    );
  } else {
    // normalize if needed
    torch::Tensor normalized_inputs;
    if (this->norm_weights.defined()) {
      if (this->sharding_dim == 1) {
        TORCH_CHECK(false, "normalizing sharded inputs is unsupported for sharding_dim 1");
      }

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
    if (this->comm->rank == 0 && residual.defined()) {
      output = output + residual;
    }

    // collect outputs
    if (collect_outputs) {
      return MuiLLMParallelLinear::collect_output(this->comm, output, this->sharding_dim);
    } else {
      return output;
    }
  }
}

torch::Tensor MuiLLMParallelLinear::collect_output(
  muillm_comm_t* comm,
  torch::Tensor& output,
  int sharding_dim
) {
  if (sharding_dim == 0) {
    // need an all gather
    TORCH_CHECK(false, "all_gather is not implemented");
  } else if (sharding_dim == 1) {
    // need an all-reduce
    muillm_comm_error_t muillm_error;
    if ((muillm_error = muillm_all_reduce_sum(
        comm,
        output
      )) != MUILLM_COMM_SUCCESS) {
      TORCH_CHECK(false, "reduction failed");
    }
  } else {
    TORCH_CHECK(false, "unsupported sharding dim");
  }

  return output;
}

//
// Python trampolines
//

// init
muillm_parallel_linear_module_ptr_t muillm_parallel_linear_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  torch::Tensor weights,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  std::optional<torch::Tensor> mul_bias_,
  std::optional<torch::Tensor> add_bias_,
  int sharding_dim) {

  auto undef_tensor = torch::Tensor();

  torch::Tensor& norm_weights = norm_weights_.has_value() ? norm_weights_.value() : undef_tensor;
  torch::Tensor& mul_bias = mul_bias_.has_value() ? mul_bias_.value() : undef_tensor;
  torch::Tensor& add_bias = add_bias_.has_value() ? add_bias_.value() : undef_tensor;

  MuiLLMParallelLinear* m = new MuiLLMParallelLinear(
    engine.engine_ptr,
    comm.comm_ptr,
    norm_weights,
    weights,
    mul_bias,
    add_bias,
    epsilon,
    sharding_dim
  );

  muillm_parallel_linear_module_ptr_t ret;
  ret.ptr = m;
  return ret;
}

// deinit
void muillm_parallel_linear_module_deinit_trampoline(
  muillm_parallel_linear_module_ptr_t module_ptr) {
  delete module_ptr.ptr;
}

// forward
at::Tensor muillm_parallel_linear_module_forward_trampoline(
  muillm_parallel_linear_module_ptr_t module_ptr,
  torch::Tensor& inputs,
  std::optional<torch::Tensor> residual_,
  bool reduce) {

  auto undef_tensor = torch::Tensor();
  torch::Tensor& residual = residual_.has_value() ? residual_.value() : undef_tensor;
  
  MuiLLMParallelLinear* m = module_ptr.ptr;

  return m->forward(inputs, residual, reduce);
}
