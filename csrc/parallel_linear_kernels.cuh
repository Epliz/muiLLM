#ifndef __MUILLM_PARALLEL_LINEAR_KERNELS_CUH__
#define __MUILLM_PARALLEL_LINEAR_KERNELS_CUH__

#include "linear_kernels.cuh"
#include "comm.h"

#include <vector>

at::Tensor muillm_parallel_linear_activ_forward(
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

#endif // __MUILLM_PARALLEL_LINEAR_KERNELS_CUH__