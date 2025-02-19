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


// needed because Pybind11 can't seem to be able to deal with opaque pointers
struct muillm_comm_ptr {
    muillm_comm_t* comm_ptr;
  };
  
muillm_comm_ptr muillm_comm_init_trampoline(
    muillm_engine_ptr engine,
    int world_size,
    int local_size,
    int rank,
    int local_rank
);

void muillm_all_reduce_sum_trampoline(
    muillm_comm_ptr comm,
    torch::Tensor& tensor
);

void muillm_broadcast_trampoline(
    muillm_comm_ptr comm,
    torch::Tensor& tensor,
    int src
);



#endif /* __MUILLM_COMM_TORCH_H__ */