#ifndef __MUILLM_GATEUP_KERNELS_CUH__
#define __MUILLM_GATEUP_KERNELS_CUH__

#include "../engine.h"

#include <torch/extension.h>

#include "gateupmlpactivation.h"

#define MUILLM_GATEUP_KERNELS_MAX_BATCH_SIZE 16

void muillm_gateupmlp_forward_placed_output(
    muillm_engine_t* engine,
    MuiGateUpMLPActivation activation,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x,
    void* output_ptr);

at::Tensor muillm_gateupmlp_forward(
    muillm_engine_t* engine,
    MuiGateUpMLPActivation activation,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x);

// python trampoline
at::Tensor muillm_gateupmlp_forward_trampoline(
    muillm_engine_ptr engine,
    int activation,
    std::optional<torch::Tensor> norm_weights_,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor down_weights,
    std::optional<torch::Tensor> residual_,
    torch::Tensor x
);

#endif // __MUILLM_GATEUP_KERNELS_CUH__