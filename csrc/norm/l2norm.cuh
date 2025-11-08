#ifndef __MUILLM_L2NORM_KERNELS_H__
#define __MUILLM_L2NORM_KERNELS_H__

#include <torch/extension.h>

#include <optional>

at::Tensor muillm_l2norm_forward(
    torch::Tensor x,
    torch::Tensor residual, // optional
    float epsilon);

// python trampoline
at::Tensor muillm_l2norm_forward_trampoline(
    torch::Tensor x,
    std::optional<torch::Tensor> residual,
    float epsilon
);

#endif /* __MUILLM_L2NORM_KERNELS_H__ */