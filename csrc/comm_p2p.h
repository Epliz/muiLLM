#ifndef __MUILLM_COMM_P2P_HPP__
#define __MUILLM_COMM_P2P_HPP__

#include "comm_base.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef struct muillm_comm_p2p_buffer_set {
  void* buffers[MUILLM_COMM_MAX_GPUS];
  size_t capacity;
} muillm_comm_p2p_buffer_set_t;

typedef struct muillm_comm_p2p: muillm_comm {

  // reduction buffer sets
  muillm_comm_p2p_buffer_set_t* first_buffers;
  muillm_comm_p2p_buffer_set_t* second_buffers;

  // shared signal memory to synchronize GPUs
  uint32_t* signal_host;
  uint32_t* signal;

  uint32_t signal_seq_no;

  // event to flush the caches
  hipEvent_t acquire_event;

} muillm_comm_p2p_t;

muillm_comm_error_t muillm_comm_p2p_init_comm(
    int world_size,
    int local_size,
    int rank,
    int local_rank,
    const muillm_comm_local_socket_t* local_socket,
    muillm_comm_p2p_t** comm_ptr,
    hipStream_t stream
);

muillm_comm_error_t muillm_comm_p2p_placed_all_reduce_sum(
  muillm_comm_p2p_t* comm,
  const void** src_ptrs,
  void* dst_ptr,
  size_t count,
  muillm_comm_datatype_t datatype,
  hipStream_t stream
);

muillm_comm_error_t muillm_comm_p2p_all_reduce_sum(
    muillm_comm_p2p_t* comm,
    const void* src_ptr,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
);

muillm_comm_error_t muillm_comm_p2p_broadcast(
  muillm_comm_p2p_t* comm,
  int src,
  const void* src_ptr,
  void* dst_ptr,
  size_t count,
  muillm_comm_datatype_t datatype,
  hipStream_t stream
);

muillm_comm_error_t muillm_comm_p2p_get_buffers(
  muillm_comm_p2p_t* comm,
  size_t count,
  muillm_comm_datatype_t datatype,
  void*** buffers,
  hipStream_t stream
);

#endif // __MUILLM_COMM_P2P_HPP__