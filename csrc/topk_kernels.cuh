#ifndef __MUILLM_TOPK_KERNELS_CUH__
#define __MUILLM_TOPK_KERNELS_CUH__

#include <torch/extension.h>
#include <tuple>

// return top-k values and indices
std::tuple<at::Tensor, at::Tensor> muillm_topk_sigmoid_forward(
    torch::Tensor x,
    int k
);

#endif /* __MUILLM_TOPK_KERNELS_CUH__ */