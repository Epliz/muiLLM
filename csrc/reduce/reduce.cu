#include "reduce.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>


#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

void muillm_reduce_sum_fp16(
  hipStream_t stream,
  unsigned B,
  unsigned M,
  unsigned N,
  bool reduce_last_dim,
  const half* x,
  half* y
);

void muillm_reduce_sum_bf16(
  hipStream_t stream,
  unsigned B,
  unsigned M,
  unsigned N,
  bool reduce_last_dim,
  const __hip_bfloat16* x,
  __hip_bfloat16* y
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void muillm_reduce_sum_forward_placed_output(
    torch::Tensor x, // shape ()
    int dim,
    bool keep_dim,
    void* output_ptr,
    hipStream_t stream
) {
  CHECK_INPUT(x);

  int num_dims = x.dim();
  TORCH_CHECK(num_dims >= 1, "Input tensor must have at least one dimension");

  auto dtype = x.dtype();

  if (dim < 0) {
    dim += num_dims; // convert negative dimension to positive
  }

  auto input_sizes = x.sizes();

  bool reduce_last_dim = (dim == num_dims - 1);

  int B, M, N;
  // calculate the number of blocks and threads
  if (reduce_last_dim) {
    // B can be 1
    B = 1;
    // M is the product of all dimensions before the last dimension
    M = 1;
    for (int i = 0; i < num_dims - 1; ++i) {
      M *= input_sizes[i];
    }
    // N is the last dimension
    N = input_sizes[num_dims - 1];
  } else {
    // B is the product of all dimensions before the reduced dimension
    B = 1;
    for (int i = 0; i < dim; ++i) {
      B *= input_sizes[i];
    }
    // M is the size of the reduced dimension
    M = input_sizes[dim];
    // N is the product of all dimensions after the reduced dimension
    N = 1;
    for (int i = dim + 1; i < num_dims; ++i) {
      N *= input_sizes[i];
    }
  }

  if (dtype == torch::kBFloat16) {
    // call the bfloat16 kernel
    muillm_reduce_sum_bf16(
        stream,
        B,
        M,
        N,
        reduce_last_dim,
        (__hip_bfloat16*)x.data_ptr(),
        (__hip_bfloat16*)output_ptr
    );
    return;
  } else if (dtype == torch::kFloat16) {
    muillm_reduce_sum_fp16(
        stream,
        B,
        M,
        N,
        reduce_last_dim,
        (const half*)x.data_ptr(),
        (half*)output_ptr
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type for reduction");
  }
}

at::Tensor muillm_reduce_sum_forward(
    torch::Tensor x,
    int dim,
    bool keep_dim
) {

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  if (dim < 0) {
    dim += x.dim();
  }

  TORCH_CHECK(dim >= 0 && dim < x.dim(), "Invalid dimension for reduction");

  // the output size is the same as input size except for the reduced dimension
  auto output_sizes = x.sizes().vec();
  if (!keep_dim) {
    TORCH_CHECK(x.dim() > 0, "Cannot remove dimension for a scalar tensor");
    // get the size of the output
    output_sizes.erase(output_sizes.begin() + dim); // remove the dimension
  } else {
    // keep dim
    output_sizes[dim] = 1; // reduce the dimension
  }

  auto dtype = x.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto y = torch::empty(output_sizes, output_options);

  muillm_reduce_sum_forward_placed_output(
      x,
      dim,
      keep_dim,
      y.data_ptr(),
      stream
  );

  return y;
}