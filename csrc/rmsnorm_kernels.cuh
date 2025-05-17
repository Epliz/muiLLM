#ifndef __MUILLM_RMSNORM_KERNELS_H__
#define __MUILLM_RMSNORM_KERNELS_H__

#include <torch/extension.h>

at::Tensor muillm_rmsnorm_forward(
    torch::Tensor weights,
    torch::Tensor inputs,
    float epsilon);

#endif /* __MUILLM_RMSNORM_KERNELS_H__ */