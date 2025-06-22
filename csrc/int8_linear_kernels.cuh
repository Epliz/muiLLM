#ifndef __MUILLM_INT8_LINEAR_KERNELS_CUH__
#define __MUILLM_INT8_LINEAR_KERNELS_CUH__

#include "linear/activation.h"

at::Tensor muillm_int8_linear_activ_forward(
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor weights,
    torch::Tensor scales_min_vals,
    int group_size_shift,
    mui_activation activ,
    torch::Tensor mul_bias,
    torch::Tensor add_bias,
    torch::Tensor x
);

#endif // __MUILLM_INT8_LINEAR_KERNELS_CUH__