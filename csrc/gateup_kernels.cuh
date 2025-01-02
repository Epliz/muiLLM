#ifndef __MUILLM_GATEUP_KERNELS_CUH__
#define __MUILLM_GATEUP_KERNELS_CUH__

#include <torch/extension.h>

at::Tensor muillm_gateupsilu_forward(
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor x);

at::Tensor muillm_gateupsilu_split_forward(
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor x);

#endif // __MUILLM_GATEUP_KERNELS_CUH__