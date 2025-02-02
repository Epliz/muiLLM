#ifndef __MUILLM_COMM_TORCH_CUH__
#define __MUILLM_COMM_TORCH_CUH__

#include <torch/extension.h>
#include <vector>

#include "comm.h"

void muillm_all_reduce_sum(
    muillm_comm_t* comm,
    std::vector<torch::Tensor>& tensors
);

std::vector<torch::Tensor> muillm_broadcast(
    muillm_comm_t* comm,
    torch::Tensor& tensor
);

#endif // __MUILLM_COMM_TORCH_CUH__