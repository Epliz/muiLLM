#ifndef __MUILLM_COMM_BASE_HPP__
#define __MUILLM_COMM_BASE_HPP__

#include <stdint.h>
#include <stddef.h>

typedef enum muillm_comm_error {
  MUILLM_COMM_SUCCESS = 0,

  MUILLM_COMM_UNSUPPORTED_SIZE,

  MUILLM_SOCKET_CREATION_FAILED,
  MUILLM_SOCKET_BIND_FAILED,
  MUILLM_SOCKET_LISTEN_FAILED,
  MUILLM_SOCKET_ACCEPT_FAILED,
  MUILLM_SOCKET_CONNECT_FAILED,

  MUILLM_SOCKET_READ_ERROR,
  MUILLM_SOCKET_WRITE_ERROR,

  MUILLM_UNKNOWN_ERROR = -1
} muillm_comm_error_t;

typedef enum muillm_comm_datatype {
  MUILLM_COMM_BOOL,
  MUILLM_COMM_INT32,
  MUILLM_COMM_FP16,
  MUILLM_COMM_FP32
} muillm_comm_datatype_t;

#define CPU_CACHELINE_SIZE 64
#define INT_CACHELINE_SIZE (CPU_CACHELINE_SIZE / sizeof(int))

#define GPU_CACHELINE_SIZE 128
// 2MiB is the shareable page size
#define GPU_SHAREABLE_PAGE_SIZE (2 * 1024 * 1024)

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define ALIGN_UP(a, b) (DIV_ROUND_UP((a), (b)) * (b))

// returns the size in bytes for the given datatype and number of elements
static inline size_t __comm_size(
    muillm_comm_datatype_t datatype,
    size_t count
) {
  switch (datatype) {
    case MUILLM_COMM_BOOL: {
      return 1 * count;
    }
    case MUILLM_COMM_INT32: {
      return 4 * count;
    }
    case MUILLM_COMM_FP16: {
      return 2 * count;
    }
    case MUILLM_COMM_FP32: {
      return 4 * count;
    }
    default: {
      return 0;
    }
  }
}

typedef enum muillm_comm_method {
  MUILLM_COMM_METHOD_P2P_TRANSFER,
  MUILLM_COMM_METHOD_STAGED_TRANSFER,
} muillm_comm_method_t;

typedef struct muillm_comm {
  muillm_comm_method_t transfer_method;

  int world_size;
  int local_size;
  int rank;
  int local_rank;

  // local sockets for server side exchanges
  int server_fd; // socket to accept new connections, only one rank will have it
  int* server_to_client_fds; // socket to communicate from the main server to all other ranks
  int client_to_server_fd; // socket for all other ranks to communicate to the server

  // usable size of the buffers
  size_t recv_buffer_size;

} muillm_comm_t;

muillm_comm_error_t __open_local_socket(
    muillm_comm_t* comm,
    int local_size,
    int local_rank
);

void __local_socket_barrier(
    muillm_comm_t* comm
);

void __local_socket_broadcast(
    muillm_comm_t* comm,
    int src_local_rank,
    void* ptr,
    size_t byte_count
);

void __local_socket_all_gather(
    muillm_comm_t* comm,
    void* in_ptr,
    size_t byte_count,
    void* out_ptr
);

#endif // __MUILLM_COMM_BASE_HPP__