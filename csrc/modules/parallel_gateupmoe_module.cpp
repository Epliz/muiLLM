#include "parallel_gateupmoe_module.h"

#include "../parallel_gateupmoe_kernels.cuh"
#include "../topk/topk.cuh"

MuiLLMParallelGateUpDownMLPMoE::MuiLLMParallelGateUpDownMLPMoE(
  muillm_engine_t* engine,
  muillm_comm_t* comm,
  MuiLLMLinear* router,
  int num_shared_experts,
  int num_dynamic_experts,
  int num_routed_experts,
  torch::Tensor& norm_weights,
  torch::Tensor& gate_weights,
  torch::Tensor& up_weights,
  torch::Tensor& down_weights,
  float variance_epsilon
) {
  this->engine = engine;
  this->comm = comm;

  this->router = router;

  this->num_shared_experts = num_shared_experts;
  this->num_dynamic_experts = num_dynamic_experts;
  this->num_routed_experts = num_routed_experts;

  this->norm_weights = norm_weights;
  this->gate_weights = gate_weights;
  this->up_weights = up_weights;
  this->down_weights = down_weights;

  this->variance_epsilon = variance_epsilon;

  auto wdtype = gate_weights.dtype();
  bool dispatchable_type = (wdtype == torch::kFloat16 || wdtype == torch::kBFloat16);
  bool dispatchable_device = gate_weights.device().is_cuda();
  this->dispatchable = dispatchable_device && dispatchable_type;
}

MuiLLMParallelGateUpDownMLPMoE::~MuiLLMParallelGateUpDownMLPMoE() {
  // nothing to do
}

torch::Tensor MuiLLMParallelGateUpDownMLPMoE::forward(
  torch::Tensor& inputs,
  torch::Tensor& residual,
  bool reduce
) {
  if (!this->dispatchable) {
    TORCH_CHECK(false, "MuiLLMParallelGateUpDownMLPMoE not dispatchable");
  }

  auto router_logits = this->router->forward(inputs, residual);

  auto [router_scores, router_indices] = muillm_topk_sigmoid_forward(
    router_logits,
    this->num_routed_experts
  );

  return muillm_parallel_gateupsilumoe_forward(
    this->engine,
    this->comm,
    this->num_shared_experts,
    this->num_dynamic_experts,
    this->norm_weights,
    this->variance_epsilon,
    this->gate_weights,
    this->up_weights,
    this->down_weights,
    residual,
    inputs,
    router_scores,
    router_indices,
    reduce
  );
}

muillm_parallel_igateupdownmlp_module_ptr_t muillm_parallel_gateupdownmlpmoe_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_linear_module_ptr_t router_module_ptr,
  int num_shared_experts,
  int num_dynamic_experts,
  int num_routed_experts,
  std::optional<torch::Tensor>& norm_weights_,
  torch::Tensor& gate_weights,
  torch::Tensor& up_weights,
  torch::Tensor& down_weights,
  float variance_epsilon
) {
  
  auto undef_tensor = torch::Tensor();

  torch::Tensor& norm_weights = norm_weights_.has_value() ? norm_weights_.value() : undef_tensor;

  MuiLLMParallelGateUpDownMLPMoE* mlp_module = new MuiLLMParallelGateUpDownMLPMoE(
    engine.engine_ptr,
    comm.comm_ptr,
    router_module_ptr.ptr,
    num_shared_experts,
    num_dynamic_experts,
    num_routed_experts,
    norm_weights,
    gate_weights,
    up_weights,
    down_weights,
    variance_epsilon
  );

  muillm_parallel_igateupdownmlp_module_ptr_t module_ptr;
  module_ptr.ptr = mlp_module;
  return module_ptr;
}

void muillm_parallel_gateupdownmlpmoe_module_deinit_trampoline(
  muillm_parallel_igateupdownmlp_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

at::Tensor muillm_parallel_gateupdownmlpmoe_module_forward_trampoline(
    muillm_parallel_igateupdownmlp_module_ptr_t module_ptr,
    torch::Tensor& inputs,
    std::optional<torch::Tensor> residual_,
    bool reduce
) {
  auto undef_tensor = torch::Tensor();
  torch::Tensor& residual = residual_.has_value() ? residual_.value() : undef_tensor;

  return ((MuiLLMParallelGateUpDownMLPMoE*)module_ptr.ptr)->forward(inputs, residual, reduce);
}