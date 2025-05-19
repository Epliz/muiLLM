#ifndef __MUILLM_GATEUPMOE_KERNELS_CUH__
#define __MUILLM_GATEUPMOE_KERNELS_CUH__

#include "engine.h"

#include <torch/extension.h>

void muillm_gateupsilumoe_forward_placed_output(
    muillm_engine_t* engine,
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
    void* output_ptr
);

void muillm_gateupsilumoe_split_forward_placed_output(
    muillm_engine_t* engine,
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
    void* output_ptr
);

at::Tensor muillm_gateupsilumoe_forward(
    muillm_engine_t* engine,
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
    torch::Tensor& router_indices
);

at::Tensor muillm_gateupsilumoe_split_forward(
    muillm_engine_t* engine,
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
    torch::Tensor& router_indices
);

// python trampoline
at::Tensor muillm_gateupsilumoe_forward_trampoline(
    muillm_engine_ptr engine,
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
    torch::Tensor router_indices
);

at::Tensor muillm_gateupsilumoe_split_forward_trampoline(
    muillm_engine_ptr engine,
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
    torch::Tensor router_indices
);

#endif // __MUILLM_GATEUPMOE_KERNELS_CUH__