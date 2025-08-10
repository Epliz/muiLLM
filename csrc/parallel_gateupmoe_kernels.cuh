#ifndef __MUILLM_PARALLEL_GATEUPMOE_KERNELS_CUH__
#define __MUILLM_PARALLEL_GATEUPMOE_KERNELS_CUH__


#include "engine.h"
#include "comms/comm_torch.h"

#include <torch/extension.h>

// parallel Gate/Up Silu (FFN)
at::Tensor muillm_parallel_gateupmlpmoe_forward(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    int num_shared_experts,
    int num_dynamic_experts,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x,
    torch::Tensor& router_scores,
    torch::Tensor& router_indices,
    bool reduce
);

at::Tensor muillm_parallel_gateupmlpmoe_split_forward(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    int num_shared_experts,
    int num_dynamic_experts,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x,
    torch::Tensor& router_scores,
    torch::Tensor& router_indices,
    bool reduce
);

at::Tensor muillm_parallel_gateupmlpmoe_forward_trampoline(
    muillm_engine_ptr engine,
    muillm_comm_ptr comm,
    int num_shared_experts,
    int num_dynamic_experts,
    std::optional<torch::Tensor> norm_weights_,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor down_weights,
    std::optional<torch::Tensor> residual_,
    torch::Tensor x,
    torch::Tensor router_scores,
    torch::Tensor router_indices,
    bool reduce
);

at::Tensor muillm_parallel_gateupmlpmoe_split_forward_trampoline(
    muillm_engine_ptr engine,
    muillm_comm_ptr comm,
    int num_shared_experts,
    int num_dynamic_experts,
    std::optional<torch::Tensor> norm_weights_,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor down_weights,
    std::optional<torch::Tensor> residual_,
    torch::Tensor x,
    torch::Tensor router_scores,
    torch::Tensor router_indices,
    bool reduce
);

#endif /* __MUILLM_PARALLEL_GATEUPMOE_KERNELS_CUH__ */