#ifndef __MUILLM_COMM_TORCH_H__
#define __MUILLM_COMM_TORCH_H__

#include <torch/extension.h>

#include "comm.h"

muillm_comm_error_t muillm_all_reduce_sum(
    muillm_comm_t* comm,
    torch::Tensor& tensor
);

muillm_comm_error_t muillm_broadcast(
    muillm_comm_t* comm,
    torch::Tensor& tensor,
    int src
);

#endif /* __MUILLM_COMM_TORCH_H__ */