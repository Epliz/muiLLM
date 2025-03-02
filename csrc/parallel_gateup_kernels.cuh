#ifndef __MUILLM_PARALLEL_GATEUP_KERNELS_CUH__
#define __MUILLM_PARALLEL_GATEUP_KERNELS_CUH__


#include "engine.h"
#include "comm_torch.h"

#include <torch/extension.h>

// parallel Gate/Up Silu (FFN)
at::Tensor muillm_parallel_gateupsilu_forward(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x);

at::Tensor muillm_parallel_gateupsilu_split_forward(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x);

at::Tensor muillm_parallel_gateupsilu_forward_trampoline(
    muillm_engine_ptr engine,
    muillm_comm_ptr comm,
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor down_weights,
    torch::Tensor residual,
    torch::Tensor x
);

at::Tensor muillm_parallel_gateupsilu_split_forward_trampoline(
    muillm_engine_ptr engine,
    muillm_comm_ptr comm,
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor down_weights,
    torch::Tensor residual,
    torch::Tensor x
);

#endif /* __MUILLM_PARALLEL_GATEUP_KERNELS_CUH__ */