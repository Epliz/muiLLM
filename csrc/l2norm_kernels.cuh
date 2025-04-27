#ifndef __MUILLM_L2NORM_KERNELS_H__
#define __MUILLM_L2NORM_KERNELS_H__

#include <torch/extension.h>

at::Tensor muillm_l2norm_forward(
    torch::Tensor weights,
    float epsilon);

#endif /* __MUILLM_L2NORM_KERNELS_H__ */