#ifndef __MUILLM_COMM_HPP__
#define __MUILLM_COMM_HPP__

#include "comm_base.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <stdint.h>
#include <stddef.h>

typedef struct muillm_comm muillm_comm_t;

muillm_comm_error_t muillm_comm_init(
    int world_size,
    int local_size,
    int rank,
    int local_rank,
    muillm_comm_t** comm,
    hipStream_t stream
);

muillm_comm_error_t muillm_comm_all_reduce_sum(
    muillm_comm_t* comm,
    void* src_ptr,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
);

muillm_comm_error_t muillm_comm_broadcast(
    muillm_comm_t* comm,
    int src,
    const void* src_ptr,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
);

#endif // __MUILLM_COMM_HPP__