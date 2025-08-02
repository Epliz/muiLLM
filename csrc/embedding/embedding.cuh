#ifndef __MUILLM_EMBEDDING_KERNELS_CUH__
#define __MUILLM_EMBEDDING_KERNELS_CUH__

#include "../engine.h"

#include <torch/extension.h>

// variant where the output needs to be placed somewhere precise
// (used when fusing reductions)
void muillm_embedding_forward_placed_output(
    muillm_engine_t* engine,
    torch::Tensor& weights,
    torch::Tensor& inputs,
    void* output_ptr,
    hipStream_t stream
);

at::Tensor muillm_embedding_forward(
    muillm_engine_t* engine,
    torch::Tensor& weights,
    torch::Tensor& inputs
);

#endif // __MUILLM_EMBEDDING_KERNELS_CUH__