#ifndef __MUILLM_COMM_STAGED_HPP__
#define __MUILLM_COMM_STAGED_HPP__

#include "comm_base.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef struct muillm_comm_staged_recv_buffer_set {
  void* staged_recv_buffer;
  void* staged_recv_buffer_cpu;
} muillm_comm_staged_recv_buffer_set_t;

// buffers into which we place signal values for barriers
// we use two sets of buffers and ping-pong between them
// so that even if some GPUs are very fast, we don't get the issue of one
// GPU being stuck between the other GPUs already have reached the next barrier
typedef struct muillm_comm_wait_buffer_set {
  int* wait_buffer;
  void* wait_buffer_cpu;
} muillm_comm_wait_buffer_set_t;

typedef struct muillm_comm_staged: muillm_comm {

  // buffers into which we place signal values for barriers
  // we use two sets of buffers and ping-pong between them
  // so that even if some GPUs are very fast, we don't get the issue of one
  // GPU being stuck between the other GPUs already have reached the next barrier
  muillm_comm_wait_buffer_set_t wait_buffer_set;
  muillm_comm_wait_buffer_set_t second_wait_buffer_set;

  // For when we do GPU->CPU->GPU data exchanges
  muillm_comm_staged_recv_buffer_set_t staged_recv_buffer_set;
  muillm_comm_staged_recv_buffer_set_t second_staged_recv_buffer_set;

  // sequence number to use to signal the barrier 
  int seq_no;
} muillm_comm_staged_t;

muillm_comm_error_t __init_staged_comm(
    int world_size,
    int local_size,
    int rank,
    int local_rank,
    muillm_comm_staged_t** comm_ptr
);

void __local_staged_gpu_barrier(
    muillm_comm_staged_t* comm,
    hipStream_t stream
);

void __all_reduce_sum_staged(
    muillm_comm_staged_t* comm,
    void* src_ptr,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
);

void __broadcast_staged(
    muillm_comm_staged_t* comm,
    int src_rank,
    void* ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
);

#endif // __MUILLM_COMM_STAGED_HPP__