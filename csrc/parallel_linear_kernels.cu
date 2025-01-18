#include "parallel_linear_kernels.cuh"
#include "comm_torch.cuh"

#include <ATen/cuda/CUDAContext.h>

#include <iostream>

std::vector<at::Tensor> muillm_parallel_linear_activ_forward(
    muillm_comm_t* comm,
    std::vector<torch::Tensor>& norm_weights,
    float epsilon,
    std::vector<torch::Tensor>& weights,
    mui_activation activ,
    std::vector<torch::Tensor>& mul_bias,
    std::vector<torch::Tensor>& add_bias,
    torch::Tensor& residual,
    int sharding_dim, // 0 for row-wise, 1 for column-wise
    bool reduce,
    std::vector<torch::Tensor>& x
) {
  size_t tp_level = weights.size();

  std::vector<at::Tensor> outputs;

  size_t norm_weights_size = norm_weights.size();
  size_t mul_bias_size = mul_bias.size();
  size_t add_bias_size = add_bias.size();

  auto undef_tensor = torch::Tensor();

  if (reduce) {
    if (tp_level > MUILLM_COMM_MAX_GPUS) {
      TORCH_CHECK(false, "tensor level parallelism higher than what is supported by comms");
    }

    // we fuse part of the reduction with the GEMV operation
    // by making the GEMV operation write into the reduction buffers
    muillm_comm_error_t muillm_error;
    muillm_comm_buffer_set_t* buffer_set = nullptr;


    const auto N = weights[0].size(0);

    // for both row-wise sharding and column wise sharding,
    // we need the reduction/collection buffer to be big enough to hold the output of the linear layers
    // not the final output size (which for row-wise sharding is bigger)
    size_t in_count = N;
    size_t output_count = sharding_dim == 1 ? N : (N * tp_level);
  
    muillm_comm_datatype_t datatype = MUILLM_COMM_FP16;
  
    if ((muillm_error = muillm_comm_get_buffer_set(comm, in_count, datatype, &buffer_set)) != MUILLM_COMM_SUCCESS) {
      TORCH_CHECK(false, "failed to get reduction buffers");
    }

    // get pointers on the output tensors to pass to the reduction
    void* output_ptrs[MUILLM_COMM_MAX_GPUS];

    for (size_t t = 0; t < tp_level; t++) {
      auto device = x[t].device();
      cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

      auto dtype = torch::kFloat16;
      auto output_options = at::TensorOptions()
                                .dtype(dtype)
                                .layout(at::kStrided)
                                .device(device) // same output device as inputs
                                .requires_grad(false);

      // y has the same dimensions as x, except the last dim that is given by
      // the out_features of weights
      auto output_sizes = x[t].sizes().vec();
      output_sizes[output_sizes.size() - 1] = output_count;

      auto output = torch::empty(output_sizes, output_options);
      void* output_ptr = output.data_ptr();

      muillm_linear_activ_forward_placed_output(
        t < norm_weights_size ? norm_weights[t] : undef_tensor,
        epsilon,
        weights[t],
        activ,
        t < mul_bias_size ?  mul_bias[t] : undef_tensor,
        t < add_bias_size ? add_bias[t] : undef_tensor,
        // we apply the residual only on device 0
        t == 0 ? residual : undef_tensor,
        x[t],
        buffer_set->buffers[t]
      );

      outputs.push_back(output);
      output_ptrs[t] = output_ptr;
    }

    // finish the reduction
    if (sharding_dim == 0) {
      // need an all gather
      if ((muillm_error = muillm_comm_all_gather(
        comm,
        (const void**) buffer_set->buffers,
        in_count,
        (void**) output_ptrs,
        output_count,
        datatype
        )) != MUILLM_COMM_SUCCESS) {
        TORCH_CHECK(false, "all-gather failed");
      }
    } else if (sharding_dim == 1) {
      // need an all-reduce
      if ((muillm_error = muillm_comm_all_reduce_sum(
        comm,
        (const void**) buffer_set->buffers,
        (void**) output_ptrs,
        output_count,
        datatype
        )) != MUILLM_COMM_SUCCESS) {
        TORCH_CHECK(false, "reduction failed");
      }
    } else {
      TORCH_CHECK(false, "unsupported sharding dim");
    }

  } else {
    for (size_t t = 0; t < tp_level; t++) {
      outputs.push_back(
          muillm_linear_activ_forward(
              t < norm_weights_size ? norm_weights[t] : undef_tensor,
              epsilon,
              weights[t],
              activ,
              t < mul_bias_size ?  mul_bias[t] : undef_tensor,
              t < add_bias_size ? add_bias[t] : undef_tensor,
              // we apply the residual only on device 0
              t == 0 ? residual : undef_tensor,
              x[t]
          )
      );
    }
  }

  return outputs;
}