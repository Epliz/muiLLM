#ifndef __MUILLM_LINEAR_KERNELS_CUH__
#define __MUILLM_LINEAR_KERNELS_CUH__

#include <torch/extension.h>

enum mui_activation {
    Identity = 0,
    Silu = 1
};

at::Tensor muillm_linear_activ_forward(
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor weights,
    mui_activation activ,
    torch::Tensor mul_bias,
    torch::Tensor add_bias,
    torch::Tensor residual,
    torch::Tensor x
);

#endif // __MUILLM_LINEAR_KERNELS_CUH__