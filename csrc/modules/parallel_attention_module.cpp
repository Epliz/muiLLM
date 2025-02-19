#include "parallel_attention_module.h"
#include "../causal_transformer_decoding.cuh"


//
// Python trampolines
//

// init
muillm_parallel_attention_module_ptr muillm_parallel_attention_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_linear_module_ptr_t o_proj
) {

  MuiLLMParallelAttention* m = new MuiLLMParallelAttention(
    engine.engine_ptr,
    comm.comm_ptr,
    o_proj.ptr
  );

  muillm_parallel_attention_module_ptr_t ret;
  ret.ptr = m;
  return ret;
}

// deinit
void muillm_parallel_attention_module_deinit_trampoline(
  muillm_parallel_attention_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

// forward
at::Tensor muillm_parallel_attention_module_forward_trampoline(
  muillm_parallel_attention_module_ptr_t module_ptr,
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  std::optional<torch::Tensor>& mask_,
  std::optional<torch::Tensor>& residual_) {

  auto undef_tensor = torch::Tensor();
  torch::Tensor& mask = mask_.has_value() ? mask_.value() : undef_tensor;
  torch::Tensor& residual = residual_.has_value() ? residual_.value() : undef_tensor;
  
  MuiLLMParallelAttention* m = module_ptr.ptr;

  return m->forward(q, k, v, mask, residual);
}

//
// Actual code
//

MuiLLMParallelAttention::MuiLLMParallelAttention(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  MuiLLMParallelLinear* o_proj
) {
  this->engine = engine;
  this->comm = comm;
  this->o_proj = o_proj;
}

torch::Tensor MuiLLMParallelAttention::forward(
  torch::Tensor& q,
  torch::Tensor& k,
  torch::Tensor& v,
  torch::Tensor& m,
  torch::Tensor& residual
) {
  bool masked = m.defined();
  auto attn_output = masked ?
      muillm_causal_transformer_decoding_masked(q, k, v, m)
    : muillm_causal_transformer_decoding_no_mask(q, k, v);


  // o proj
  auto proj_attn_output = this->o_proj->forward(attn_output, residual, /*collect_outputs*/ true);
  return proj_attn_output;
}