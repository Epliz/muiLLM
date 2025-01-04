#ifndef __MUILLM_PARALLEL_LINEAR_KERNELS_CUH__
#define __MUILLM_PARALLEL_LINEAR_KERNELS_CUH__

#include "linear_kernels.cuh"
#include "comm.h"

#include <vector>

std::vector<at::Tensor> muillm_parallel_linear_activ_forward(
    muillm_comm_t* comm,
    std::vector<torch::Tensor>& norm_weights,
    float epsilon,
    std::vector<torch::Tensor>& weights,
    mui_activation activ,
    std::vector<torch::Tensor>& mul_bias,
    std::vector<torch::Tensor>& add_bias,
    torch::Tensor& residual,
    bool reduce,
    std::vector<torch::Tensor>& x
);

#endif // __MUILLM_PARALLEL_LINEAR_KERNELS_CUH__