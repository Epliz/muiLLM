#ifndef __MUILLM_REDUCE_CUH__
#define __MUILLM_REDUCE_CUH__

#include <torch/extension.h>

void muillm_reduce_sum_forward_placed_output(
    torch::Tensor x,
    int dim,
    bool keep_dim,
    void* output_ptr,
    hipStream_t stream
);

at::Tensor muillm_reduce_sum_forward(
    torch::Tensor x,
    int dim,
    bool keep_dim
);

#endif /* __MUILLM_REDUCE_CUH__ */