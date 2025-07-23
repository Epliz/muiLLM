#ifndef __MUILLM_PARALLEL_LINEAR_KERNELS_CUH__
#define __MUILLM_PARALLEL_LINEAR_KERNELS_CUH__

#include "engine.h"
#include "comms/comm_torch.h"
#include "linear/linear.cuh" // for activation enum

#include <optional>

#include <torch/extension.h>

at::Tensor muillm_parallel_linear_activ_forward(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& weights,
    mui_activation activ,
    torch::Tensor& mul_bias,
    torch::Tensor& add_bias,
    torch::Tensor& residual,
    int sharding_dim, // 0 for row-wise, 1 for column-wise
    bool reduce,
    torch::Tensor& x
);

// python trampoline
at::Tensor muillm_parallel_linear_forward_trampoline(
    muillm_engine_ptr engine,
    muillm_comm_ptr comm,
    torch::Tensor x,
    torch::Tensor weights,
    std::optional<torch::Tensor> norm_weights_,
    float epsilon,
    std::optional<torch::Tensor> mul_bias_,
    std::optional<torch::Tensor> add_bias_,
    std::optional<torch::Tensor> residual_,
    int sharding_dim,
    bool reduce
);

#endif // __MUILLM_PARALLEL_LINEAR_KERNELS_CUH__