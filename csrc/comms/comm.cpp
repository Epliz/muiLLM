#include "comm.h"

#include "comm_p2p.h"
#include "comm_staged.h"

#include <signal.h>
#include <unistd.h>
#include <stdio.h>

void handler(int signo)
{
  int i = 1;
  printf("pid=%d, got signal=%d\n", getpid(), signo);
  fflush(stdout);
  while (i) { }
}

muillm_comm_error_t muillm_comm_init(
  muillm_engine_t* engine,
  int world_size,
  int local_size,
  int rank,
  int local_rank,
  muillm_comm_t** comm_ptr,
  hipStream_t stream
) {
  muillm_comm_error_t muillm_error;


  // Set up the sigaction
  struct sigaction sa;
  sa.sa_handler = handler;
  sa.sa_flags = 0; // or SA_RESTART
  sigemptyset(&sa.sa_mask);

  if (sigaction(SIGSEGV, &sa, NULL) == -1) {
    perror("Error setting signal handler for SIGSEGV");
    exit(EXIT_FAILURE); 
  }
  if (sigaction(SIGABRT, &sa, NULL) == -1) {
    perror("Error setting signal handler for SIGABRT");
    exit(EXIT_FAILURE); 
  }

  // establish the local socket connection
  printf("(rank %d) Opening local socket...\n", local_rank);
  muillm_comm_local_socket_t local_socket;
  if ((muillm_error = __open_local_socket(local_size, local_rank, &local_socket)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }
  printf("(rank %d) Opened local socket\n", local_rank);

  // try p2p comms first, but it might not always work
  muillm_error = muillm_comm_p2p_init_comm(
    engine,
    world_size,
    local_size,
    rank,
    local_rank,
    &local_socket,
    (muillm_comm_p2p_t**) comm_ptr,
    stream
  );

  if (muillm_error == MUILLM_COMM_SUCCESS) {
    return MUILLM_COMM_SUCCESS;
  }

  // try staged comms as a fallback
  return muillm_comm_staged_init_comm(
    engine,
    world_size,
    local_size,
    rank,
    local_rank,
    &local_socket,
    (muillm_comm_staged_t**) comm_ptr,
    stream
  );

}


muillm_comm_error_t muillm_comm_placed_all_reduce_sum(
  muillm_comm_t* comm,
  const void** src_ptrs,
  void* dst_ptr,
  size_t count,
  muillm_comm_datatype_t datatype,
  hipStream_t stream
) {
if (comm->transfer_method == MUILLM_COMM_METHOD_P2P_TRANSFER) {
  return muillm_comm_p2p_placed_all_reduce_sum(
    (muillm_comm_p2p_t*) comm,
    src_ptrs,
    dst_ptr,
    count,
    datatype,
    stream
  );
} else if (comm->transfer_method == MUILLM_COMM_METHOD_STAGED_TRANSFER) {
  return muillm_comm_staged_placed_all_reduce_sum(
    (muillm_comm_staged_t*) comm,
    src_ptrs,
    dst_ptr,
    count,
    datatype,
    stream
  );
} else {
  return MUILLM_COMM_UNKNOWN_ERROR;
}
}

muillm_comm_error_t muillm_comm_all_reduce_sum(
    muillm_comm_t* comm,
    const void* src_ptr,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  if (comm->transfer_method == MUILLM_COMM_METHOD_P2P_TRANSFER) {
    return muillm_comm_p2p_all_reduce_sum(
      (muillm_comm_p2p_t*) comm,
      src_ptr,
      dst_ptr,
      count,
      datatype,
      stream
    );
  } else if (comm->transfer_method == MUILLM_COMM_METHOD_STAGED_TRANSFER) {
    return muillm_comm_staged_all_reduce_sum(
      (muillm_comm_staged_t*) comm,
      src_ptr,
      dst_ptr,
      count,
      datatype,
      stream
    );
  } else {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
}

muillm_comm_error_t muillm_comm_broadcast(
  muillm_comm_t* comm,
  int src,
  const void* src_ptr,
  void* dst_ptr,
  size_t count,
  muillm_comm_datatype_t datatype,
  hipStream_t stream
) {
  if (comm->transfer_method == MUILLM_COMM_METHOD_P2P_TRANSFER) {
    return muillm_comm_p2p_broadcast(
      (muillm_comm_p2p_t*) comm,
      src,
      src_ptr,
      dst_ptr,
      count,
      datatype,
      stream
    );
  } else if (comm->transfer_method == MUILLM_COMM_METHOD_STAGED_TRANSFER) {
    return muillm_comm_staged_broadcast(
      (muillm_comm_staged_t*) comm,
      src,
      src_ptr,
      dst_ptr,
      count,
      datatype,
      stream
    );
  } else {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
}

muillm_comm_error_t muillm_comm_get_buffers(
  muillm_comm_t* comm,
  size_t count,
  muillm_comm_datatype_t datatype,
  void*** buffers,
  hipStream_t stream
) {
  if (comm->transfer_method == MUILLM_COMM_METHOD_P2P_TRANSFER) {
    return muillm_comm_p2p_get_buffers(
      (muillm_comm_p2p_t*) comm,
      count,
      datatype,
      buffers,
      stream
    );
  } else if (comm->transfer_method == MUILLM_COMM_METHOD_STAGED_TRANSFER) {
    return muillm_comm_staged_get_buffers(
      (muillm_comm_staged_t*) comm,
      count,
      datatype,
      buffers,
      stream
    );
  } else {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
}