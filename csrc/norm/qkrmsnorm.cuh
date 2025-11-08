#ifndef __MUILLM_QKRMSNORM_KERNELS_H__
#define __MUILLM_QKRMSNORM_KERNELS_H__

#include <torch/extension.h>

#include <tuple>

std::tuple<at::Tensor, at::Tensor> muillm_qkrmsnorm_forward(
    torch::Tensor q_weights,
    torch::Tensor k_weights,
    torch::Tensor q,
    torch::Tensor k,
    float epsilon,
    float weight_offset
);

#endif /* __MUILLM_QKRMSNORM_KERNELS_H__ */