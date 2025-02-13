#ifndef __MUILLM_GATEUP_KERNELS_CUH__
#define __MUILLM_GATEUP_KERNELS_CUH__

#include "engine.h"

#include <torch/extension.h>

void muillm_gateupsilu_forward_placed_output(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x,
    void* output_ptr);

void muillm_gateupsilu_split_forward_placed_output(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x,
    void* output_ptr);

at::Tensor muillm_gateupsilu_forward(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x);

at::Tensor muillm_gateupsilu_split_forward(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x);

#endif // __MUILLM_GATEUP_KERNELS_CUH__