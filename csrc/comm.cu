#include "comm.h"

#include "comm_staged.h"
#include "comm_p2p.h"

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
    int world_size,
    int local_size,
    int rank,
    int local_rank,
    muillm_comm_t** comm_ptr
) {
  muillm_comm_method_t transfer_method = MUILLM_COMM_METHOD_P2P_TRANSFER;
  if (transfer_method == MUILLM_COMM_METHOD_STAGED_TRANSFER) {
    return __init_staged_comm(
      world_size,
      local_size,
      rank,
      local_rank,
      (muillm_comm_staged_t**) comm_ptr
    );
  } else if (transfer_method == MUILLM_COMM_METHOD_P2P_TRANSFER) {
    return __init_p2p_comm(
      world_size,
      local_size,
      rank,
      local_rank,
      (muillm_comm_p2p_t**) comm_ptr
    );
  }

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
}

void muillm_comm_barrier(
    muillm_comm_t* comm,
    hipStream_t stream
) {
  // sync the GPUs
  if (comm->transfer_method == MUILLM_COMM_METHOD_STAGED_TRANSFER) {
    __local_staged_gpu_barrier((muillm_comm_staged_t*) comm, stream);
  } else if (comm->transfer_method == MUILLM_COMM_METHOD_P2P_TRANSFER) {
    __local_p2p_gpu_barrier((muillm_comm_p2p_t*) comm, stream);
  } else {
    printf("Unsupported sync method\n");
  }
}

void muillm_comm_all_reduce_sum(
    muillm_comm_t* comm,
    void* src_ptr,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {

  if (comm->transfer_method == MUILLM_COMM_METHOD_P2P_TRANSFER) {
    __all_reduce_sum_p2p(
      (muillm_comm_p2p_t*) comm,
      src_ptr,
      dst_ptr,
      count,
      datatype,
      stream
    );
  } else if (comm->transfer_method == MUILLM_COMM_METHOD_STAGED_TRANSFER) {
    __all_reduce_sum_staged(
      (muillm_comm_staged_t*) comm,
      src_ptr,
      dst_ptr,
      count,
      datatype,
      stream
    );
  } else {
    printf("Unsupported reduce method\n");
  }

}

void muillm_comm_broadcast(
    muillm_comm_t* comm,
    int src_rank,
    void* ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {

  if (comm->transfer_method == MUILLM_COMM_METHOD_P2P_TRANSFER) {
    __broadcast_p2p(
      (muillm_comm_p2p_t*) comm,
      src_rank,
      ptr,
      count,
      datatype,
      stream
    );
  } else if (comm->transfer_method == MUILLM_COMM_METHOD_STAGED_TRANSFER) {
    __broadcast_staged(
      (muillm_comm_staged_t*) comm,
      src_rank,
      ptr,
      count,
      datatype,
      stream
    );
  } else {
    printf("Unsupported broadcast method\n");
  }
}