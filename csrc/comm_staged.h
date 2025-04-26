#ifndef __MUILLM_COMM_STAGED_HPP__
#define __MUILLM_COMM_STAGED_HPP__

#include "comm_base.h"
#include "engine.h"

typedef struct muillm_comm_staged_buffer_set {
  void* buffers[MUILLM_COMM_MAX_GPUS];
  void* host_buffers[MUILLM_COMM_MAX_GPUS];
  size_t capacity;
} muillm_comm_staged_buffer_set_t;

typedef struct muillm_comm_staged: muillm_comm {

  // reduction buffer sets
  muillm_comm_staged_buffer_set_t* first_buffers;
  muillm_comm_staged_buffer_set_t* second_buffers;

  // shared signal memory to synchronize GPUs
  uint64_t* signal_host;
  uint64_t* signal;

  uint64_t signal_seq_no;

  // event to flush the caches
  hipEvent_t acquire_event;

} muillm_comm_staged_t;

muillm_comm_error_t muillm_comm_staged_init_comm(
  muillm_engine_t* engine,
  int world_size,
  int local_size,
  int rank,
  int local_rank,
  const muillm_comm_local_socket_t* local_socket,
  muillm_comm_staged_t** comm_ptr,
  hipStream_t stream
);

muillm_comm_error_t muillm_comm_staged_placed_all_reduce_sum(
  muillm_comm_staged_t* comm,
  const void** src_ptrs,
  void* dst_ptr,
  size_t count,
  muillm_comm_datatype_t datatype,
  hipStream_t stream
);

muillm_comm_error_t muillm_comm_staged_all_reduce_sum(
    muillm_comm_staged_t* comm,
    const void* src_ptr,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
);

muillm_comm_error_t muillm_comm_staged_broadcast(
  muillm_comm_staged_t* comm,
  int src,
  const void* src_ptr,
  void* dst_ptr,
  size_t count,
  muillm_comm_datatype_t datatype,
  hipStream_t stream
);

muillm_comm_error_t muillm_comm_staged_get_buffers(
  muillm_comm_staged_t* comm,
  size_t count,
  muillm_comm_datatype_t datatype,
  void*** buffers,
  hipStream_t stream
);

#endif // __MUILLM_COMM_STAGED_HPP__