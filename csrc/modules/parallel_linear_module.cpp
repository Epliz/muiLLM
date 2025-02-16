#include "parallel_linear_module.h"

#include "../parallel_linear_kernels.cuh"
#include "../rmsnorm_kernels.cuh"
#include "../comm_torch.h"

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
  bool dispatchable_type = (wdtype == at::kHalf);
  bool dispatchable_device = weights.device().is_cuda();
  this->dispatchable = dispatchable_type && dispatchable_device;
}

torch::Tensor MuiLLMParallelLinear::forward(
    torch::Tensor& inputs,
    torch::Tensor& residual,
    bool collect_outputs
) {
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
      if (sharding_dim == 0) {
        // need an all gather
        TORCH_CHECK(false, "all_gather is not implemented");
      } else if (sharding_dim == 1) {
        // need an all-reduce
        muillm_comm_error_t muillm_error;
        if ((muillm_error = muillm_all_reduce_sum(
            this->comm,
            output
          )) != MUILLM_COMM_SUCCESS) {
          TORCH_CHECK(false, "reduction failed");
        }
      } else {
        TORCH_CHECK(false, "unsupported sharding dim");
      }
    }
  
    return output;
  }
}