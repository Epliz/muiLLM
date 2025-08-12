#ifndef __MUILLM_LINEAR_KERNELS_CUH__
#define __MUILLM_LINEAR_KERNELS_CUH__

#include "../engine.h"

#include <torch/extension.h>

#include "activation.h"

// variant where the output needs to be placed somewhere precise
// (used when fusing reductions by parallel linear)
void muillm_linear_activ_forward_placed_output(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor& weights,
    mui_activation activ,
    torch::Tensor& mul_bias,
    torch::Tensor& add_bias,
    torch::Tensor& residual,
    torch::Tensor& x,
    void* output_ptr,
    hipStream_t stream
);

at::Tensor muillm_linear_activ_forward(
    muillm_engine_t* engine,
    torch::Tensor& norm_weights,
    float epsilon,
    float norm_weights_offset,
    torch::Tensor& weights,
    mui_activation activ,
    torch::Tensor& mul_bias,
    torch::Tensor& add_bias,
    torch::Tensor& residual,
    torch::Tensor& x
);

// python trampoline
at::Tensor muillm_linear_forward_trampoline(
    muillm_engine_ptr engine,
    torch::Tensor x,
    torch::Tensor weights,
    std::optional<torch::Tensor> norm_weights_,
    float epsilon,
    float norm_weights_offset,
    std::optional<torch::Tensor> mul_bias_,
    std::optional<torch::Tensor> add_bias_,
    std::optional<torch::Tensor> residual_
);

#endif // __MUILLM_LINEAR_KERNELS_CUH__