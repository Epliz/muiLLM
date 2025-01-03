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
  MUILLM_COMM_FP16 = 0,
  MUILLM_COMM_FP32
} muillm_comm_datatype_t;

typedef struct muillm_comm {
  int local_size;

  hipStream_t* streams;
  hipEvent_t* acquire_events;
  hipEvent_t* release_events;
  uint64_t** signals;
  uint64_t signal_seq_no;
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

#endif // __MUILLM_COMM_HPP__