#include "generation.cuh"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define THREADS_PER_BLOCK 256

// kernel to copy the previous tokens in the new array
__global__ void muillm_copy_prev_tokens_kernel(
  // inputs
  const uint64_t* input_ids,
  // outputs
  uint64_t* next_input_ids, // size [B, T+1]
  // other arguments
  int T
) {
  unsigned batch_idx = blockIdx.y;
  unsigned tok_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  if (tok_idx < T) {
    next_input_ids[batch_idx * (T + 1) + tok_idx] = input_ids[batch_idx * T + tok_idx];
  }
}

// kernel to copy the next tokens and check if the sequences
// are finished or not
__global__ void muillm_add_tokens_check_finished_kernel(
    // inputs
    const uint32_t* next_tokens,
    const uint64_t* unfinished_sequences,
    const uint64_t* eos_token_ids,
    // outputs
    uint64_t* next_input_ids, // size [B, T+1]
    uint64_t* next_unfinished_sequences,
    // other arguments
    int64_t max_length,
    int pad_token_id,
    int B,
    int T,
    int NUM_EOS_TOKENS,
    bool has_max_length_stopping_criteria,
    bool has_eos_stopping_criteria
) {
  // each threads processes one batch entry
  unsigned batch_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  if (batch_idx >= B) {
    // out of bounds -> exit
    return;
  }

  uint64_t next_token = next_tokens[batch_idx];

  bool prev_unfinished = unfinished_sequences[batch_idx];

  // first update the next_input_ids (so that we insert EOS tokens)
  if (has_eos_stopping_criteria) {
    // if the sequence was previously unfinished will copy out the next token
    // (potentially it is eos_token_id and will finished the sequence though)
    next_token = prev_unfinished ? next_token : pad_token_id;
  }

  next_input_ids[batch_idx] = next_token;

  // copy out the next token
  next_input_ids[batch_idx * (T + 1) + T] = next_token;

  // then update the unfinished_sequences mask
  bool unfinished = prev_unfinished;
  if (has_max_length_stopping_criteria) {
    unfinished = unfinished & (T <= max_length);
  }

  if (has_eos_stopping_criteria) {
    if (NUM_EOS_TOKENS == 1) {
      unfinished = unfinished & (next_token != eos_token_ids[0]);
    } else if (NUM_EOS_TOKENS == 2) {
      // if there are two eos tokens, we check if the next token is one of them
      bool has_eos_tok0 = (next_token == eos_token_ids[0]);
      bool has_eos_tok1 = (next_token == eos_token_ids[1]);
      unfinished = unfinished & (!has_eos_tok0 && !has_eos_tok1);
    } else if (NUM_EOS_TOKENS == 3) {
      // if there are three eos tokens, we check if the next token is one of them
      bool has_eos_tok0 = (next_token == eos_token_ids[0]);
      bool has_eos_tok1 = (next_token == eos_token_ids[1]);
      bool has_eos_tok2 = (next_token == eos_token_ids[2]);
      unfinished = unfinished & (!has_eos_tok0 && !has_eos_tok1 && !has_eos_tok2);
    } else if (NUM_EOS_TOKENS == 4) {
      // if there are four eos tokens, we check if the next token is one of them
      bool has_eos_tok0 = (next_token == eos_token_ids[0]);
      bool has_eos_tok1 = (next_token == eos_token_ids[1]);
      bool has_eos_tok2 = (next_token == eos_token_ids[2]);
      bool has_eos_tok3 = (next_token == eos_token_ids[3]);
      unfinished = unfinished & (!has_eos_tok0 && !has_eos_tok1 && !has_eos_tok2 && !has_eos_tok3);
    } else {
      // if there are more than four eos tokens, we check if the next token is one of them
      // this is a bit inefficient, but it is not expected to be used often
      bool is_eos_token = false;
      for (int i = 0; i < NUM_EOS_TOKENS; ++i) {
        if (next_token == eos_token_ids[i]) {
          is_eos_token = true;
          break;
        }
      }
        unfinished = unfinished & !is_eos_token;
    }
  }

  next_unfinished_sequences[batch_idx] = unfinished;
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> muillm_add_tokens_check_finished(
    torch::Tensor input_ids,
    torch::Tensor next_tokens,
    torch::Tensor unfinished_sequences,
    bool has_eos_stopping_criteria,
    int pad_token_id,
    torch::Tensor eos_token_ids,
    bool has_max_length_stopping_criteria,
    int max_length
) {
  CHECK_INPUT(input_ids);
  CHECK_INPUT(next_tokens);
  CHECK_INPUT(unfinished_sequences);

  auto device = input_ids.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  const auto B = input_ids.size(0);
  const auto T = input_ids.size(1);
  const auto NUM_EOS_TOKENS = eos_token_ids.size(0);

  if (unfinished_sequences.size(0) != B) {
    TORCH_CHECK(false, "unfinished_sequences doesn't have the shape [B]");
  }

  auto input_ids_dtype = input_ids.dtype();

  if (input_ids_dtype != torch::kInt64) {
    TORCH_CHECK(false, "input_ids must be of type int64");
  }

  auto output_options = at::TensorOptions()
                            .dtype(input_ids_dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto next_input_ids = torch::empty({B, T + 1}, output_options);

  auto unfinished_sequences_dtype = unfinished_sequences.dtype();

  if (unfinished_sequences_dtype != torch::kInt64) {
    TORCH_CHECK(false, "unfinished_sequences must be of type int64");
  }

  auto unfinished_output_options = at::TensorOptions()
                            .dtype(unfinished_sequences_dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto next_unfinished_sequences = torch::empty_like(unfinished_sequences, unfinished_output_options);

  auto next_tokens_dtype = next_tokens.dtype();
  if (next_tokens_dtype != torch::kInt) {
    TORCH_CHECK(false, "next_tokens must be of type int");
  }

  { // call the cuda kernel to copy the previous tokens in the new array
    const int threads_per_block = THREADS_PER_BLOCK;
    const int num_blocks_x = DIV_ROUND_UP(T, threads_per_block);
    const int num_blocks_y = B;

    muillm_copy_prev_tokens_kernel<<<dim3(num_blocks_x, num_blocks_y), threads_per_block, 0, stream>>>(
        // input pointers
        (const uint64_t*) input_ids.data_ptr(),
        // output pointers
        (uint64_t*) next_input_ids.data_ptr(),
        // other args
        T
    );
  }
  { // call the cuda kernel to copy the next tokens and check if the sequences are finished
    const int threads_per_block = THREADS_PER_BLOCK;
    const int num_blocks = DIV_ROUND_UP(B, threads_per_block);

    muillm_add_tokens_check_finished_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        // input pointers
        (const uint32_t*) next_tokens.data_ptr(),
        (const uint64_t*) unfinished_sequences.data_ptr(),
        (const uint64_t*) eos_token_ids.data_ptr(),
        // output pointers
        (uint64_t*) next_input_ids.data_ptr(),
        (uint64_t*) next_unfinished_sequences.data_ptr(),
        // other args
        max_length,
        pad_token_id,
        B,
        T,
        NUM_EOS_TOKENS,
        has_eos_stopping_criteria,
        has_max_length_stopping_criteria
    );
  }

  return std::make_tuple(next_input_ids, next_unfinished_sequences);
}