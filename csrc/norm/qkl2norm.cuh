#ifndef __MUILLM_QKL2NORM_KERNELS_H__
#define __MUILLM_QKL2NORM_KERNELS_H__

#include <torch/extension.h>

#include <tuple>

std::tuple<at::Tensor, at::Tensor> muillm_qkl2norm_forward(
    torch::Tensor q,
    torch::Tensor k,
    float epsilon);

#endif /* __MUILLM_QKL2NORM_KERNELS_H__ */