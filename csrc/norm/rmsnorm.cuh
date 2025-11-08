#ifndef __MUILLM_RMSNORM_KERNELS_H__
#define __MUILLM_RMSNORM_KERNELS_H__

#include <torch/extension.h>

#include <optional>

at::Tensor muillm_rmsnorm_forward(
    torch::Tensor weights,
    torch::Tensor x,
    torch::Tensor residual, // optional
    float epsilon,
    float weight_offset
);

// python trampoline
at::Tensor muillm_rmsnorm_forward_trampoline(
    torch::Tensor weights,
    torch::Tensor x,
    std::optional<torch::Tensor> residual,
    float epsilon,
    float weight_offset
);

#endif /* __MUILLM_RMSNORM_KERNELS_H__ */