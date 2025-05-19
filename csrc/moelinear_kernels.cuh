#ifndef __MUILLM_MOELINEAR_KERNELS_CUH__
#define __MUILLM_MOELINEAR_KERNELS_CUH__

#include "engine.h"

#include <torch/extension.h>

enum mui_activation {
    Identity = 0,
    Silu = 1
};

// variant where the output needs to be placed somewhere precise
// (used when fusing reductions by parallel linear)
void muillm_moelinear_activ_forward_placed_output(
    muillm_engine_t* engine,
    int num_shared_experts,
    int num_dynamic_experts,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& weights,
    mui_activation activ,
    torch::Tensor& mul_bias,
    torch::Tensor& add_bias,
    torch::Tensor& residual,
    torch::Tensor& x,
    torch::Tensor& router_indices,
    void* output_ptr,
    hipStream_t stream
);

at::Tensor muillm_moelinear_activ_forward(
    muillm_engine_t* engine,
    int num_shared_experts,
    int num_dynamic_experts,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& weights,
    mui_activation activ,
    torch::Tensor& mul_bias,
    torch::Tensor& add_bias,
    torch::Tensor& residual,
    torch::Tensor& x,
    torch::Tensor& router_indices
);

// python trampoline
at::Tensor muillm_moelinear_forward_trampoline(
    muillm_engine_ptr engine,
    int num_shared_experts,
    int num_dynamic_experts,
    torch::Tensor x,
    torch::Tensor router_indices,
    torch::Tensor weights,
    std::optional<torch::Tensor> norm_weights_,
    float epsilon,
    std::optional<torch::Tensor> mul_bias_,
    std::optional<torch::Tensor> add_bias_,
    std::optional<torch::Tensor> residual_
);

#endif // __MUILLM_MOELINEAR_KERNELS_CUH__