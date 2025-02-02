#ifndef __MUILLM_COMM_HPP__
#define __MUILLM_COMM_HPP__

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <stddef.h>

typedef enum muillm_comm_error {
  MUILLM_COMM_SUCCESS = 0,

  MUILLM_COMM_UNKNOWN_ERROR
} muillm_comm_error_t;

typedef enum muillm_comm_datatype {
  MUILLM_COMM_BOOL = 0,
  MUILLM_COMM_INT8,
  MUILLM_COMM_INT16,
  MUILLM_COMM_INT32,
  MUILLM_COMM_INT64,
  MUILLM_COMM_FP16,
  MUILLM_COMM_FP32,
  MUILLM_COMM_FP64
} muillm_comm_datatype_t;

#define MUILLM_COMM_MAX_GPUS 8

typedef struct muillm_comm_buffer_set {
  void* buffers[MUILLM_COMM_MAX_GPUS];
  size_t capacity;
} muillm_comm_buffer_set_t;

typedef struct muillm_comm {
  // reduction buffer sets
  muillm_comm_buffer_set_t* first_buffers;
  muillm_comm_buffer_set_t* second_buffers;

  hipStream_t* streams;
  hipEvent_t* acquire_events;
  hipEvent_t* release_events;
  uint64_t** signals;

  uint64_t signal_seq_no;
  int local_size;
} muillm_comm_t;

muillm_comm_error_t muillm_comm_init(
  int local_size,
  bool allocate_streams,
  muillm_comm_t** comm_ptr
);

muillm_comm_error_t muillm_comm_all_reduce_sum(
  muillm_comm_t* comm,
  const void** src_ptrs,
  void** dst_ptrs,
  size_t count,
  muillm_comm_datatype_t datatype
);

muillm_comm_error_t muillm_comm_broadcast(
  muillm_comm_t* comm,
  const void* src_ptr,
  void** dst_ptrs,
  size_t count,
  muillm_comm_datatype_t datatype
);

muillm_comm_error_t muillm_comm_all_gather(
  muillm_comm_t* comm,
  const void** src_ptrs,
  size_t in_count,
  void** dst_ptrs,
  size_t dst_count,
  muillm_comm_datatype_t datatype
);

muillm_comm_error_t muillm_comm_get_buffer_set(muillm_comm_t* comm, size_t count, muillm_comm_datatype_t datatype, muillm_comm_buffer_set_t** buffer_set);

#endif // __MUILLM_COMM_HPP__