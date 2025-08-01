#include "parallel_linear_kernels.cuh"

#include <ATen/cuda/CUDAContext.h>

//
// Python trampolines
//

at::Tensor muillm_parallel_linear_forward_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  torch::Tensor x,
  torch::Tensor weights,
  std::optional<torch::Tensor> norm_weights_,
  float epsilon,
  std::optional<torch::Tensor> mul_bias_,
  std::optional<torch::Tensor> add_bias_,
  std::optional<torch::Tensor> residual_,
  int sharding_dim,
  bool reduce) {

  auto undef_tensor = torch::Tensor();
  torch::Tensor empty_tensor_list;

  torch::Tensor& norm_weights = norm_weights_.has_value() ? norm_weights_.value() : empty_tensor_list;
  torch::Tensor& mul_biases = mul_bias_.has_value() ? mul_bias_.value() : empty_tensor_list;
  torch::Tensor& add_biases = add_bias_.has_value() ? add_bias_.value() : empty_tensor_list;
  torch::Tensor residual = residual_.has_value() ? residual_.value() : undef_tensor;

  return muillm_parallel_linear_activ_forward(
      engine.engine_ptr,
      comm.comm_ptr,
      norm_weights,
      epsilon,
      weights,
      mui_activation::Identity,
      mul_biases,
      add_biases,
      residual,
      sharding_dim,
      reduce,
      x
  );
}

//
// Actual stuff
//

at::Tensor muillm_parallel_linear_activ_forward(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& weights,
    mui_activation activ,
    torch::Tensor& mul_bias,
    torch::Tensor& add_bias,
    torch::Tensor& residual,
    int sharding_dim, // 0 for row-wise, 1 for column-wise
    bool reduce,
    torch::Tensor& x
) {
  int rank = comm->rank;
  // TODO: change this once we support interleaving
  size_t tp_level = comm->local_size;

  auto undef_tensor = torch::Tensor();

  if (reduce) {
    if (tp_level > MUILLM_COMM_MAX_GPUS) {
      TORCH_CHECK(false, "tensor level parallelism higher than what is supported by comms");
    }

    auto device = x.device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
  
    // we fuse part of the reduction with the GEMV operation
    // by making the GEMV operation write into the reduction buffers
    muillm_comm_error_t muillm_error;
    void** buffers;


    const auto N = weights.size(0);

    // for both row-wise sharding and column wise sharding,
    // we need the reduction/collection buffer to be big enough to hold the output of the linear layers
    // not the final output size (which for row-wise sharding is bigger)
    size_t in_count = N;
    size_t output_count = sharding_dim == 1 ? N : (N * tp_level);
  
    auto dtype = x.dtype();

    muillm_comm_datatype_t datatype;
  
    if (dtype == torch::kFloat16) {
      datatype = MUILLM_COMM_FP16;
    } else if (dtype == torch::kBFloat16) {
      datatype = MUILLM_COMM_BF16;
    } else {
      // error
      TORCH_CHECK(false, "Unsupported dtype for all_reduce_sum");
    }

    if ((muillm_error = muillm_comm_get_buffers(comm, in_count, datatype, &buffers, stream)) != MUILLM_COMM_SUCCESS) {
      TORCH_CHECK(false, "failed to get reduction buffers");
    }

    auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

    // y has the same dimensions as x, except the last dim that is given by
    // the out_features of weights
    auto output_sizes = x.sizes().vec();
    output_sizes[output_sizes.size() - 1] = output_count;

    auto output = torch::empty(output_sizes, output_options);
    void* output_ptr = output.data_ptr();

    muillm_linear_activ_forward_placed_output(
      engine,
      norm_weights,
      epsilon,
      weights,
      activ,
      mul_bias,
      add_bias,
      // we apply the residual only on device 0
      rank == 0 ? residual : undef_tensor,
      x,
      buffers[rank],
      stream
    );

    // finish the reduction
    if (sharding_dim == 0) {
      // need an all gather
      TORCH_CHECK(false, "all_gather is not implemented");
    } else if (sharding_dim == 1) {
      // need an all-reduce
      if ((muillm_error = muillm_comm_placed_all_reduce_sum(
        comm,
        (const void**) buffers,
        output_ptr,
        output_count,
        datatype,
        stream
        )) != MUILLM_COMM_SUCCESS) {
        TORCH_CHECK(false, "reduction failed");
      }
    } else {
      TORCH_CHECK(false, "unsupported sharding dim");
    }

    return output;
  } else {
    return muillm_linear_activ_forward(
        engine,
        norm_weights,
        epsilon,
        weights,
        activ,
        mul_bias,
        add_bias,
        // we apply the residual only on device 0
        rank == 0 ? residual : undef_tensor,
        x
    );
  }
}