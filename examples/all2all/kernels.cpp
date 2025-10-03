#include <torch/extension.h>
#include <tuple>

#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_fp16.h>

#include <stdint.h>

#include <iostream>

void all2all_dispatch_pack_fp16(
    hipStream_t stream,
    const half* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    half* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    int hidden_dim,
    int max_recv,
    int local_expert_offset,
    int total_recv
);

void all2all_dispatch_pack_fp32(
    hipStream_t stream,
    const float* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    float* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    int hidden_dim,
    int max_recv,
    int local_expert_offset,
    int total_recv
);

#define META_DIM 4

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void all2all_dispatch_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ indices,
    int64_t* __restrict__ send_counts,
    int64_t* __restrict__ send_offsets,
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int world_size
);

void all2all_dispatch_pack_send_buffers_fp32(
    hipStream_t stream,
    const float* __restrict__ x,
    const int32_t* __restrict__ indices,
    int64_t* __restrict__ send_offsets,
    int32_t* __restrict__ send_meta,
    float* __restrict__ send_buf,
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int hidden_dim,
    int world_size,
    int rank
);

void all2all_dispatch_pack_send_buffers_fp16(
    hipStream_t stream,
    const half* __restrict__ x,
    const int32_t* __restrict__ indices,
    int64_t* __restrict__ send_offsets,
    int32_t* __restrict__ send_meta,
    half* __restrict__ send_buf,
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int hidden_dim,
    int world_size,
    int rank
);

// outputs: send_counts, send_meta, send_buf
// send_counts: shape [world_size]
// send_meta: shape [num_tokens * num_experts_per_token, meta_dim]
// send_buf: shape [num_tokens * num_experts_per_token, hidden_dim]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> all2all_dispatch_build_meta(
  // inputs
  torch::Tensor& indices, // shape [num_tokens, experts_per_token]
  torch::Tensor& x, // shape [num_tokens, hidden_dim]
  int num_local_experts,
  int world_size,
  int rank
) {
  CHECK_INPUT(indices);
  CHECK_INPUT(x);

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  int num_tokens = indices.size(0);
  int num_experts_per_token = indices.size(1);
  int hidden_dim = x.size(1);

  auto dtype = x.dtype();
  auto index_dtype = indices.dtype();

  if (index_dtype != torch::kInt32) {
    TORCH_CHECK(false, "indices must be int32");
  }

  //
  // allocate the final output tensors
  //
  auto send_counts_options = at::TensorOptions()
                            .dtype(torch::kInt64)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto send_counts = torch::zeros({world_size}, send_counts_options);

  int total_send = num_tokens * num_experts_per_token;

  auto send_meta_options = at::TensorOptions()
                            .dtype(torch::kInt32)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto send_meta = torch::empty({total_send, META_DIM}, send_meta_options);

  auto send_buf_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto send_buf = torch::empty({total_send, hidden_dim}, send_buf_options);


  //
  // allocate the additional temporary tensors
  //

  // allocate the additional temporary tensors
  auto send_offsets = torch::empty({world_size}, send_counts_options);

  //
  // First we need to compute the send counts for each rank and the write offsets in the send buffer
  //
  all2all_dispatch_compute_send_counts(
    stream,
    (const int32_t*) indices.data_ptr(),
    (int64_t*) send_counts.data_ptr(),
    (int64_t*) send_offsets.data_ptr(),
    num_local_experts,
    num_tokens,
    num_experts_per_token,
    world_size
  );

  //
  // Then we need to pack the send buffer and the send meta data
  //
  if (dtype == at::kFloat) {
    all2all_dispatch_pack_send_buffers_fp32(
      stream,
      (const float*) x.data_ptr(),
      (const int32_t*) indices.data_ptr(),
      (int64_t*) send_offsets.data_ptr(),
      (int32_t*) send_meta.data_ptr(),
      (float*) send_buf.data_ptr(),
      num_local_experts,
      num_tokens,
      num_experts_per_token,
      hidden_dim,
      world_size,
      rank
    );
  } else if (dtype == at::kHalf) {
    all2all_dispatch_pack_send_buffers_fp16(
      stream,
      (const half*) x.data_ptr(),
      (const int32_t*) indices.data_ptr(),
      (int64_t*) send_offsets.data_ptr(),
      (int32_t*) send_meta.data_ptr(),
      (half*) send_buf.data_ptr(),
      num_local_experts,
      num_tokens,
      num_experts_per_token,
      hidden_dim,
      world_size,
      rank
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type");
  }

  return std::make_tuple(send_counts, send_meta, send_buf);
}

// outputs: expert_num_tokens, expert_x, expert_meta
// expert_num_tokens: shape [num_local_experts]
// expert_x: shape [num_local_experts, max_recv, hidden_dim]
// expert_meta: shape [num_local_experts, max_recv, meta_dim]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> all2all_dispatch_pack(
  // inputs
  torch::Tensor& recv_buf, // shape [total_recv, hidden_dim]
  torch::Tensor& recv_meta, // shape [total_recv, meta_dim] (expert_id, src_rank, src_token_id, topk_offset)
  int num_local_experts,
  int max_recv,
  int rank) {

  CHECK_INPUT(recv_buf);
  CHECK_INPUT(recv_meta);

  auto device = recv_buf.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  int total_recv = recv_buf.size(0);
  int hidden_dim = recv_buf.size(1);
  int meta_dim = recv_meta.size(1);

  if (meta_dim != META_DIM) {
    TORCH_CHECK(false, "meta_dim must be ", META_DIM);
  }

  auto dtype = recv_buf.dtype();
  auto meta_dtype = recv_meta.dtype();

  if (meta_dtype != torch::kInt32) {
    TORCH_CHECK(false, "meta_dtype must be int32");
  }

  auto num_experts_output_options = at::TensorOptions()
                            .dtype(torch::kInt32)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto expert_output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);


  auto meta_output_options = at::TensorOptions()
                            .dtype(torch::kInt32)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto expert_num_tokens = torch::zeros({num_local_experts}, num_experts_output_options);
  auto expert_x = torch::empty({num_local_experts, max_recv, hidden_dim}, expert_output_options);
  auto expert_meta = torch::empty({num_local_experts, max_recv, meta_dim}, meta_output_options);

  int local_expert_offset = rank * num_local_experts;

  if (dtype == at::kFloat) {

    all2all_dispatch_pack_fp32(
      stream,
      (const float*) recv_buf.data_ptr(),
      (const int32_t*) recv_meta.data_ptr(),
      (int32_t*) expert_num_tokens.data_ptr(),
      (float*) expert_x.data_ptr(),
      (int32_t*) expert_meta.data_ptr(),
      hidden_dim,
      max_recv,
      local_expert_offset,
      total_recv
    );
  } else if (dtype == at::kHalf) {
    all2all_dispatch_pack_fp16(
      stream,
      (const half*) recv_buf.data_ptr(),
      (const int32_t*) recv_meta.data_ptr(),
      (int32_t*) expert_num_tokens.data_ptr(),
      (half*) expert_x.data_ptr(),
      (int32_t*) expert_meta.data_ptr(),
      hidden_dim,
      max_recv,
      local_expert_offset,
      total_recv
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type");
  }

  return std::make_tuple(expert_num_tokens, expert_x, expert_meta);
}

void all2all_compute_fp32(
  hipStream_t stream,
  const int32_t* __restrict__ expert_num_tokens,
  const float* __restrict__ expert_x,
  float* __restrict__ expert_y,
  int num_local_experts,
  int max_recv,
  int hidden_dim,
  int rank
);

void all2all_compute_fp16(
  hipStream_t stream,
  const int32_t* __restrict__ expert_num_tokens,
  const half* __restrict__ expert_x,
  half* __restrict__ expert_y,
  int num_local_experts,
  int max_recv,
  int hidden_dim,
  int rank
);

// output: expert_y shape [num_local_experts, max_recv, hidden_dim]
at::Tensor all2all_compute(
  torch::Tensor& expert_num_tokens, // shape [num_local_experts]
  torch::Tensor& expert_x, // shape [num_local_experts, max_recv, hidden_dim]
  int rank
) {
  CHECK_INPUT(expert_x);

  auto device = expert_x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  int num_local_experts = expert_x.size(0);
  int max_recv = expert_x.size(1);
  int hidden_dim = expert_x.size(2);

  auto dtype = expert_x.dtype();

  auto expert_y_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto expert_y = torch::empty({num_local_experts, max_recv, hidden_dim}, expert_y_options);

  if (dtype == at::kFloat) {
    all2all_compute_fp32(
      stream,
      (const int32_t*) expert_num_tokens.data_ptr(),
      (const float*) expert_x.data_ptr(),
      (float*) expert_y.data_ptr(),
      num_local_experts,
      max_recv,
      hidden_dim,
      rank
    );
  } else if (dtype == at::kHalf) {
    all2all_compute_fp16(
      stream,
      (const int32_t*) expert_num_tokens.data_ptr(),
      (const half*) expert_x.data_ptr(),
      (half*) expert_y.data_ptr(),
      num_local_experts,
      max_recv,
      hidden_dim,
      rank
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type");
  }

  return expert_y;
}

void all2all_combine_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens,
    const int32_t* __restrict__ expert_meta,
    int64_t* __restrict__ send_counts,
    int64_t* __restrict__ send_offsets,
    int num_local_experts,
    int max_recv,
    int world_size
);

void all2all_combine_pack_send_buffers_fp32(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens,
    const int32_t* __restrict__ expert_meta,
    const float* __restrict__ expert_y,
    int64_t* __restrict__ send_offsets,
    int32_t* __restrict__ send_meta,
    float* __restrict__ send_buf,
    int total_send,
    int num_local_experts,
    int max_recv,
    int hidden_dim
);

void all2all_combine_pack_send_buffers_fp16(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens,
    const int32_t* __restrict__ expert_meta,
    const half* __restrict__ expert_y,
    int64_t* __restrict__ send_offsets,
    int32_t* __restrict__ send_meta,
    half* __restrict__ send_buf,
    int total_send,
    int num_local_experts,
    int max_recv,
    int hidden_dim
);

// outputs: send_counts, send_meta, send_buf
// send_counts: shape [world_size]
// send_meta: shape [total_send, meta_dim]
// send_buf: shape [total_send, hidden_dim]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> all2all_combine_build_meta(
  // inputs
  torch::Tensor& expert_num_tokens, // shape [num_local_experts]
  torch::Tensor& expert_meta, // shape [num_local_experts, max_recv, meta_dim] (expert_id, src_rank, src_token_id, topk_offset)
  torch::Tensor& expert_y, // shape [num_local_experts, max_recv, hidden_dim]
  int total_send,
  int world_size,
  int rank
) {
  CHECK_INPUT(expert_num_tokens);
  CHECK_INPUT(expert_meta);
  CHECK_INPUT(expert_y);

  auto device = expert_y.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  int num_local_experts = expert_num_tokens.size(0);
  int max_recv = expert_meta.size(1);
  int meta_dim = expert_meta.size(2);
  int hidden_dim = expert_y.size(2);

  if (meta_dim != META_DIM) {
    TORCH_CHECK(false, "meta_dim must be ", META_DIM);
  }

  auto dtype = expert_y.dtype();
  auto meta_dtype = expert_meta.dtype();

  if (meta_dtype != torch::kInt32) {
    TORCH_CHECK(false, "meta_dtype must be int32");
  }

  if (expert_num_tokens.dtype() != torch::kInt32) {
    TORCH_CHECK(false, "expert_num_tokens dtype must be int32");
  }

  //
  // allocate the final output tensors
  //
  auto send_counts_options = at::TensorOptions()
                            .dtype(torch::kInt64)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto send_counts = torch::zeros({world_size}, send_counts_options);

  auto send_meta_options = at::TensorOptions()
                            .dtype(torch::kInt32)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto send_meta = torch::empty({total_send, meta_dim}, send_meta_options);

  auto send_buf_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto send_buf = torch::empty({total_send, hidden_dim}, send_buf_options);


  //
  // allocate the additional temporary tensors
  //

  // allocate the additional temporary tensors
  auto send_offsets = torch::empty({world_size}, send_counts_options);

  //
  // First we need to compute the send counts for each rank and the write offsets in the send buffer
  //
  all2all_combine_compute_send_counts(
    stream,
    (const int32_t*) expert_num_tokens.data_ptr(),
    (const int32_t*) expert_meta.data_ptr(),
    (int64_t*) send_counts.data_ptr(),
    (int64_t*) send_offsets.data_ptr(),
    num_local_experts,
    max_recv,
    world_size
  );

  //
  // Then we need to pack the send buffer and the send meta data
  //
  if (dtype == at::kFloat) {
    all2all_combine_pack_send_buffers_fp32(
      stream,
      (const int32_t*) expert_num_tokens.data_ptr(),
      (const int32_t*) expert_meta.data_ptr(),
      (const float*) expert_y.data_ptr(),
      (int64_t*) send_offsets.data_ptr(),
      (int32_t*) send_meta.data_ptr(),
      (float*) send_buf.data_ptr(),
      total_send,
      num_local_experts,
      max_recv,
      hidden_dim
    );
  } else if (dtype == at::kHalf) {
    all2all_combine_pack_send_buffers_fp16(
      stream,
      (const int32_t*) expert_num_tokens.data_ptr(),
      (const int32_t*) expert_meta.data_ptr(),
      (const half*) expert_y.data_ptr(),
      (int64_t*) send_offsets.data_ptr(),
      (int32_t*) send_meta.data_ptr(),
      (half*) send_buf.data_ptr(),
      total_send,
      num_local_experts,
      max_recv,
      hidden_dim
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type");
  }

  return std::make_tuple(send_counts, send_meta, send_buf);
}

void all2all_combine_pack_fp32(
    hipStream_t stream,
    const float* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    const float* __restrict__ weights,
    float* __restrict__ scaled_expert_outputs,
    float* __restrict__ output,
    int hidden_dim,
    int total_recv,
    int num_tokens,
    int experts_per_token
);

void all2all_combine_pack_fp16(
    hipStream_t stream,
    const half* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    const float* __restrict__ weights,
    float* __restrict__ scaled_expert_outputs,
    half* __restrict__ output,
    int hidden_dim,
    int total_recv,
    int num_tokens,
    int experts_per_token
);

torch::Tensor all2all_combine_pack(
  // inputs
  torch::Tensor& recv_buf, // shape [total_recv, hidden_dim]
  torch::Tensor& recv_meta, // shape [total_recv, meta_dim] (expert_id, src_rank, src_token_id, topk_offset)
  torch::Tensor& weights, // shape [num_tokens, experts_per_token]
  // output
  torch::Tensor& output // shape [max_num_tokens, hidden_dim]
) { 

  CHECK_INPUT(recv_buf);
  CHECK_INPUT(recv_meta);
  CHECK_INPUT(weights);
  CHECK_INPUT(output);

  auto device = recv_buf.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  int total_recv = recv_buf.size(0);
  int hidden_dim = recv_buf.size(1);
  int meta_dim = recv_meta.size(1);
  int num_tokens = weights.size(0);
  int experts_per_token = weights.size(1);

  if (meta_dim != META_DIM) {
    TORCH_CHECK(false, "meta_dim must be ", META_DIM);
  }

  auto dtype = recv_buf.dtype();
  auto meta_dtype = recv_meta.dtype();
  auto weight_dtype = weights.dtype();

  if (meta_dtype != torch::kInt32) {
    TORCH_CHECK(false, "meta_dtype must be int32");
  }

  if (weight_dtype != torch::kFloat) {
    TORCH_CHECK(false, "weights must be float");
  }

  if (output.dtype() != dtype) {
    TORCH_CHECK(false, "output dtype must match recv_buf dtype");
  }

  auto scaled_expert_output_options = at::TensorOptions()
                            .dtype(at::kFloat)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // allocate empty tensor to hold scaled expert outputs shape [num_tokens, experts_per_token, hidden_dim]
  // makes it possible to avoid accumulations with atomicAdd
  torch::Tensor scaled_expert_outputs = torch::empty({num_tokens, experts_per_token, hidden_dim}, scaled_expert_output_options);

  if (dtype == at::kFloat) {
    all2all_combine_pack_fp32(
      stream,
      (const float*) recv_buf.data_ptr(),
      (const int32_t*) recv_meta.data_ptr(),
      (const float*) weights.data_ptr(),
      (float*) scaled_expert_outputs.data_ptr(),
      (float*) output.data_ptr(),
      hidden_dim,
      total_recv,
      num_tokens,
      experts_per_token
    );
  } else if (dtype == at::kHalf) {
    all2all_combine_pack_fp16(
      stream,
      (const half*) recv_buf.data_ptr(),
      (const int32_t*) recv_meta.data_ptr(),
      (const float*) weights.data_ptr(),
      (float*) scaled_expert_outputs.data_ptr(),
      (half*) output.data_ptr(),
      hidden_dim,
      total_recv,
      num_tokens,
      experts_per_token
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type");
  }

  return output;
}