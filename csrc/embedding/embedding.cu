#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "embedding.cuh"

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

void muillm_embedding_forward_xx16_indices_i32(
  hipStream_t stream,
  unsigned N,
  unsigned E,
  const uint16_t* weights,
  const uint32_t* x,
  uint16_t* out_embeddings,
  int simd_lanes
);

void muillm_embedding_forward_xx16_indices_i64(
  hipStream_t stream,
  unsigned N,
  unsigned E,
  const uint16_t* weights,
  const uint64_t* x,
  uint16_t* out_embeddings,
  int simd_lanes
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void muillm_embedding_forward_placed_output(
    muillm_engine_t* engine,
    torch::Tensor& weights,
    torch::Tensor& x,
    void* output_ptr,
    hipStream_t stream) {
  CHECK_INPUT(weights);
  CHECK_INPUT(x);

  auto embed_dtype = weights.dtype();
  auto index_dtype = x.dtype();

  // N is the number of embeddings to copy, which
  // is the number of elements in c
  const auto N = x.numel();
  const auto E = weights.size(1);

  int simd_lanes = engine->gpu_infos[0]->simd_lanes;

  // try to occupy enough to saturate memory bandwidth
  /*
  while ((num_blocks * threads_per_blocks < 8 * simd_lanes) && threads_per_blocks < 256) {
    threads_per_blocks *= 2;
  }
  */

  if (embed_dtype == torch::kFloat16 || embed_dtype == torch::kBFloat16) {
    if (index_dtype == torch::kInt32) {
      muillm_embedding_forward_xx16_indices_i32(
        stream,
        N,
        E,
        (const uint16_t*)weights.data_ptr(),
        (const uint32_t*)x.data_ptr(),
        (uint16_t*) output_ptr,
        simd_lanes
      );
    } else if (index_dtype == torch::kInt64) {
      muillm_embedding_forward_xx16_indices_i64(
        stream,
        N,
        E,
        (const uint16_t*)weights.data_ptr(),
        (const uint64_t*)x.data_ptr(),
        (uint16_t*) output_ptr,
        simd_lanes
      );
    } else {
      TORCH_CHECK(false, "Unsupported index dtype for embedding_forward");
    }
  } else {
    TORCH_CHECK(false, "Unsupported embedding dtype for embedding_forward");
  }
}

at::Tensor muillm_embedding_forward(
    muillm_engine_t* engine,
    torch::Tensor& weights,
    torch::Tensor& x) {
  CHECK_INPUT(x);

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  const auto E = weights.size(1);

  auto embed_dtype = weights.dtype();
  auto output_options = at::TensorOptions()
                            .dtype(embed_dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // y has the same dimensions as x, except the last dim that is given by
  // the embedding size
  auto output_sizes = x.sizes().vec();
  output_sizes.push_back(E);

  auto y = torch::empty(output_sizes, output_options);

  void* output_ptr = y.data_ptr();

  // call with the placed output
  muillm_embedding_forward_placed_output(
    engine,
    weights,
    x,
    output_ptr,
    stream
  );

  return y;
}