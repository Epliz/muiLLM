#ifndef __MUILLM_COMM_P2P_HPP__
#define __MUILLM_COMM_P2P_HPP__

#include "comm_base.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef struct muillm_comm_p2p_recv_buffer_set {
  void* p2p_recv_buffer;
  // all receive buffers, including the local and remote ones
  void** all_p2p_recv_buffers;
  hipMemPool_t* memPools; // mem pools to import pointers
} muillm_comm_p2p_recv_buffer_set_t;

typedef struct muillm_comm_event_set {
  // local event from this rank
  hipEvent_t event;
  // all events, including the local and remote ones
  hipEvent_t* all_events;
} muillm_comm_event_set_t;

typedef struct muillm_comm_p2p: muillm_comm {

  // XXXXXXXXXXX
  // TODO: determine if we need two sets of buffers in case
  // some GPUs are too far ahead (probably needed for p2p)
  // XXXXXXXXXXX

  // For when we do GPU<->GPU data exchanges
  // receive buffers of the (as many as needed, typically one per local GPU
  // as the local reduce )
  muillm_comm_p2p_recv_buffer_set_t p2p_recv_buffer_set;
  muillm_comm_p2p_recv_buffer_set_t second_p2p_recv_buffer_set;

  muillm_comm_event_set_t wait_event_set;
  muillm_comm_event_set_t second_wait_event_set;

  // sequence number to use to signal the barrier 
  int seq_no;
} muillm_comm_p2p_t;

muillm_comm_error_t __init_p2p_comm(
    int world_size,
    int local_size,
    int rank,
    int local_rank,
    muillm_comm_p2p_t** comm_ptr
);

void __local_p2p_gpu_barrier(
    muillm_comm_p2p_t* comm,
    hipStream_t stream
);

void __all_reduce_sum_p2p(
    muillm_comm_p2p_t* comm,
    void* src_ptr,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
);

void __broadcast_p2p(
    muillm_comm_p2p_t* comm,
    int src_rank,
    void* ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
);

#endif // __MUILLM_COMM_P2P_HPP__