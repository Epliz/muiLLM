import os
from typing import Tuple
import torch
import torch.distributed as dist

import torch.autograd.profiler as profiler

try:
    # relative import for when used as a package
    from .task import input_t, output_t
except ImportError:
    # absolute import for when used in submission
    from task import input_t, output_t

COMM_KERNELS_CPP_CODE = """
#include <torch/extension.h>
#include <iostream>

#ifndef __MUILLM_BASE_HPP__
#define __MUILLM_BASE_HPP__

typedef enum muillm_error {
  MUILLM_SUCCESS = 0,
  MUILLM_UNKNOWN_ERROR
} muillm_error_t;

#define MUILLM_MAX_GPUS 8

#endif // __MUILLM_BASE_HPP__

#ifndef __MUILLM_GPU_INFO_H__
#define __MUILLM_GPU_INFO_H__

typedef enum muillm_gpu_family {
  MUILLM_GPU_FAMILY_UNKNOWN = 0,
  MUILLM_GPU_FAMILY_RDNA,
  MUILLM_GPU_FAMILY_CDNA,
  MUILLM_GPU_FAMILY_UDNA
} muillm_gpu_family_t;

typedef enum muillm_gpu_arch {
  MUILLM_GPU_ARCH_UNKNOWN = 0,
  MUILLM_GPU_ARCH_RDNA1,
  MUILLM_GPU_ARCH_RDNA2,
  MUILLM_GPU_ARCH_RDNA3,
  MUILLM_GPU_ARCH_RDNA4,
  MUILLM_GPU_ARCH_MI100,
  MUILLM_GPU_ARCH_MI200,
  MUILLM_GPU_ARCH_MI300,
  MUILLM_GPU_ARCH_MI350,
  MUILLM_GPU_ARCH_MI400
} muillm_gpu_arch_t;

typedef struct muillm_gpu_info {
  muillm_gpu_arch_t arch;
  muillm_gpu_family_t family;
  // number of threads in warp: 64 for gfx9, 32 for gfx10+
  int warp_size;
  // number of simd lanes on a device ("cuda cores")
  int simd_lanes;
} muillm_gpu_info_t;

muillm_error_t muillm_detect_gpu_properties(
  int device,
  muillm_gpu_info_t* gpu_info
);

#endif /* __MUILLM_GPU_INFO_H__ */

// GPU INFO

#include <hip/hip_runtime.h>

#include <cstring>

muillm_error_t muillm_detect_gpu_properties(
    int device,
    muillm_gpu_info_t* gpu_info
) {
  
  hipDeviceProp_t properties;
  if (hipGetDeviceProperties(&properties, device) != hipSuccess) {
    TORCH_CHECK(false, "an error happened when detecting GPU properties");
    return MUILLM_UNKNOWN_ERROR;
  }

  // detect the GPU family
  const char* gfx101x = "gfx101"; // RDNA1
  const char* gfx105x = "gfx103"; // RDNA2
  const char* gfx11x = "gfx11"; // RDNA3, RDNA3,5
  const char* gfx12x = "gfx12"; // RDNA4
  const char* gfx908 = "gfx908"; // MI100
  const char* gfx90a = "gfx90a"; // MI200
  const char* gfx94x = "gfx94"; // MI300, MI300a
  const char* gfx95x = "gfx95"; // MI350
  const char* gfx125x = "gfx125"; // MI400

  if (strncmp(properties.gcnArchName, gfx908, strlen(gfx908)) == 0) {
    gpu_info->arch = MUILLM_GPU_ARCH_MI100;
    gpu_info->family = MUILLM_GPU_FAMILY_CDNA;
  } else if (strncmp(properties.gcnArchName, gfx90a, strlen(gfx90a)) == 0) {
    gpu_info->arch = MUILLM_GPU_ARCH_MI200;
    gpu_info->family = MUILLM_GPU_FAMILY_CDNA;
  } else if (strncmp(properties.gcnArchName, gfx94x, strlen(gfx94x)) == 0) {
    gpu_info->arch = MUILLM_GPU_ARCH_MI300;
    gpu_info->family = MUILLM_GPU_FAMILY_CDNA;
  } else if (strncmp(properties.gcnArchName, gfx95x, strlen(gfx95x)) == 0) {
    gpu_info->arch = MUILLM_GPU_ARCH_MI350;
    gpu_info->family = MUILLM_GPU_FAMILY_CDNA;
  } else if (strncmp(properties.gcnArchName, gfx101x, strlen(gfx101x)) == 0) {
    gpu_info->arch = MUILLM_GPU_ARCH_RDNA1;
    gpu_info->family = MUILLM_GPU_FAMILY_RDNA;
  } else if (strncmp(properties.gcnArchName, gfx105x, strlen(gfx105x)) == 0) {
    gpu_info->arch = MUILLM_GPU_ARCH_RDNA2;
    gpu_info->family = MUILLM_GPU_FAMILY_RDNA;
  } else if (strncmp(properties.gcnArchName, gfx11x, strlen(gfx11x)) == 0) {
    gpu_info->arch = MUILLM_GPU_ARCH_RDNA3;
    gpu_info->family = MUILLM_GPU_FAMILY_RDNA;
  } else if (strncmp(properties.gcnArchName, gfx125x, strlen(gfx125x)) == 0) {
    gpu_info->arch = MUILLM_GPU_ARCH_MI400;
    gpu_info->family = MUILLM_GPU_FAMILY_UDNA;
  } else if (strncmp(properties.gcnArchName, gfx12x, strlen(gfx12x)) == 0) {
    gpu_info->arch = MUILLM_GPU_ARCH_RDNA4;
    gpu_info->family = MUILLM_GPU_FAMILY_RDNA;
  }  else {
    gpu_info->arch = MUILLM_GPU_ARCH_UNKNOWN;
    gpu_info->family = MUILLM_GPU_FAMILY_UNKNOWN;
  }

  int cu_count = properties.multiProcessorCount;

  // AMD reports the number of WGPs in sm processor count instead of the CU count
  // CUs still have 64 simd lanes per CU, but WGPs have 128
  int simd_lanes_per_cu = gpu_info->family == MUILLM_GPU_FAMILY_CDNA ? 64 : 128;

  gpu_info->warp_size = properties.warpSize;
  gpu_info->simd_lanes = cu_count * simd_lanes_per_cu;

  return MUILLM_SUCCESS;
}

#ifndef __MUILLM_COMM_BASE_HPP__
#define __MUILLM_COMM_BASE_HPP__

#include <stdint.h>
#include <stddef.h>

typedef enum muillm_comm_error {
  MUILLM_COMM_SUCCESS = 0,

  MUILLM_COMM_UNSUPPORTED_SIZE,

  MUILLM_COMM_SOCKET_CREATION_FAILED,
  MUILLM_COMM_SOCKET_BIND_FAILED,
  MUILLM_COMM_SOCKET_LISTEN_FAILED,
  MUILLM_COMM_SOCKET_ACCEPT_FAILED,
  MUILLM_COMM_SOCKET_CONNECT_FAILED,

  MUILLM_COMM_SOCKET_READ_ERROR,
  MUILLM_COMM_SOCKET_WRITE_ERROR,

  MUILLM_COMM_UNKNOWN_ERROR = -1
} muillm_comm_error_t;

typedef enum muillm_comm_datatype {
  MUILLM_COMM_BOOL = 0,
  MUILLM_COMM_INT8,
  MUILLM_COMM_INT16,
  MUILLM_COMM_INT32,
  MUILLM_COMM_INT64,
  MUILLM_COMM_FP16,
  MUILLM_COMM_BF16,
  MUILLM_COMM_FP32,
  MUILLM_COMM_FP64
} muillm_comm_datatype_t;

#define MUILLM_COMM_MAX_GPUS (MUILLM_MAX_GPUS)

#define CPU_CACHELINE_SIZE 64
#define INT_CACHELINE_SIZE (CPU_CACHELINE_SIZE / sizeof(int))

#define GPU_CACHELINE_SIZE 128
// 2MiB is the shareable page size
#define GPU_SHAREABLE_PAGE_SIZE (2 * 1024 * 1024)

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define ALIGN_UP(a, b) (DIV_ROUND_UP((a), (b)) * (b))

static size_t __next_power_of_2(size_t n) {
  size_t r = 1;
  while (r < n) {
    r *= 2;
  }
  return r;
}

// returns the size in bytes for the given datatype and number of elements
static inline size_t __comm_size(
    muillm_comm_datatype_t datatype,
    size_t count
) {
  switch (datatype) {
    case MUILLM_COMM_BOOL: {
      return 1 * count;
    }
    case MUILLM_COMM_INT8: {
      return 1 * count;
    }
    case MUILLM_COMM_INT16: {
      return 2 * count;
    }
    case MUILLM_COMM_INT32: {
      return 4 * count;
    }
    case MUILLM_COMM_INT64: {
      return 8 * count;
    }
    case MUILLM_COMM_FP16: {
      return 2 * count;
    }
    case MUILLM_COMM_BF16: {
      return 2 * count;
    }
    case MUILLM_COMM_FP32: {
      return 4 * count;
    }
    case MUILLM_COMM_FP64: {
      return 8 * count;
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

typedef struct muillm_comm_local_socket {
  // local sockets for server side exchanges
  int server_fd; // socket to accept new connections, only one rank will have it
  int* server_to_client_fds; // socket to communicate from the main server to all other ranks
  int client_to_server_fd; // socket for all other ranks to communicate to the server
} muillm_comm_local_socket_t;

// base structure
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
} muillm_comm_t;

muillm_comm_error_t __open_local_socket(
    int local_size,
    int local_rank,
    muillm_comm_local_socket_t* local_socket
);

muillm_comm_error_t __local_socket_barrier(
    muillm_comm_t* comm
);

muillm_comm_error_t __local_socket_broadcast(
    muillm_comm_t* comm,
    int src_local_rank,
    void* ptr,
    size_t byte_count
);

muillm_comm_error_t __local_socket_all_gather(
    muillm_comm_t* comm,
    void* in_ptr,
    size_t byte_count,
    void* out_ptr
);

void __allocate_locked_shared_cpu_mem(
  muillm_comm_t* comm,
  size_t size,
  void** shm_addr_ptr,
  void** device_ptr_ptr
);

void __deallocate_locked_shared_cpu_mem(
  muillm_comm_t* comm,
  void* host_addr
);

#endif // __MUILLM_COMM_BASE_HPP__

#include <hip/hip_runtime.h>

#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <errno.h>
#include <poll.h>

#define MUILLM_COMM_SOCKET_PATH "/tmp/muillm_comm_socket"

static int full_read(int fd, void* ptr, size_t byte_count) {
  size_t to_read_count = byte_count;

  uint8_t* byte_ptr = (uint8_t*) ptr;
  while (to_read_count > 0) {
    int read_status = read(fd, byte_ptr, to_read_count);

    if (read_status < 0) {
      TORCH_CHECK(false, "an error happened when reading from socket");
      return -1;
    }

    byte_ptr += read_status;
    to_read_count -= read_status;
  }

  return byte_count;
}

static int full_write(int fd, const void* ptr, size_t byte_count) {
  size_t to_write_count = byte_count;

  const uint8_t* byte_ptr = (const uint8_t*) ptr;
  while (to_write_count > 0) {
    int write_status = write(fd, byte_ptr, to_write_count);

    if (write_status < 0) {
      TORCH_CHECK(false, "an error happened when writing to socket");
      return -1;
    }

    byte_ptr += write_status;
    to_write_count -= write_status;
  }

  return byte_count;
}

// creates the domain sockets used to do cpu side exchanges
// (e.g. to exchange memory IPC handles)
muillm_comm_error_t __open_local_socket(
    int local_size,
    int local_rank,
    muillm_comm_local_socket_t* local_socket
) {
  local_socket->server_fd = -1;
  local_socket->client_to_server_fd = -1;
  local_socket->server_to_client_fds = nullptr;

  bool is_server;

  // try to become the server
  struct sockaddr_un server_addr;

  // Create socket
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    TORCH_CHECK(false, "an error happened when creating socket");
    return MUILLM_COMM_SOCKET_CREATION_FAILED;
  }

  // Try bindnig socket to address
  memset(&server_addr, 0, sizeof(struct sockaddr_un));
  server_addr.sun_family = AF_UNIX;
  strncpy(server_addr.sun_path, MUILLM_COMM_SOCKET_PATH, sizeof(server_addr.sun_path) - 1);

  // the server is always the rank 0
  is_server = local_rank == 0;

  if (is_server) {
    bool correctly_bound = bind(fd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_un)) == 0;

    if (!correctly_bound) {
      TORCH_CHECK(false, "an error happened when binding socket");
      return MUILLM_COMM_SOCKET_BIND_FAILED;
    }

    // if we managed to bind, we should connect to all other ranks and they will be the socket
    // clients
    local_socket->server_fd = fd;

    int num_clients = local_size -1;

    local_socket->server_to_client_fds = new int[local_size];
    local_socket->server_to_client_fds[local_rank] = -1;

    // Listen for connections
    if (listen(local_socket->server_fd, num_clients) == -1) {
      TORCH_CHECK(false, "an error happened when listening on socket");
      return MUILLM_COMM_SOCKET_LISTEN_FAILED;
    }

    struct pollfd poll_fd;
    poll_fd.fd = local_socket->server_fd;
    poll_fd.events = POLLIN;

    for (int c = 0; c < num_clients; c++) {

      int poll_count = poll(&poll_fd, 1, -1);
      if (poll_count < 0) {
        TORCH_CHECK(false, "an error happened when polling socket");
        return MUILLM_COMM_SOCKET_ACCEPT_FAILED;
      }

      if (poll_fd.revents & POLLIN) {
        int client_fd = accept(local_socket->server_fd, NULL, NULL);
        if (client_fd == -1) {
          TORCH_CHECK(false, "an error happened when accepting socket connection");
          return MUILLM_COMM_SOCKET_ACCEPT_FAILED;
        }
      
        // now read what rank this client corresponds to
        int client_rank = 0;

        if (full_read(client_fd, &client_rank, sizeof(int)) < 0) {
          TORCH_CHECK(false, "an error happened when reading from socket");
          return MUILLM_COMM_SOCKET_READ_ERROR;
        }

        local_socket->server_to_client_fds[client_rank] = client_fd;
      }
    }

    // we can already unlink as everyone has connected
    unlink(MUILLM_COMM_SOCKET_PATH);

  } else {
    // if we didn't manage to bind, we are a mere client
    // we just need to connect to the server
    local_socket->client_to_server_fd = fd;

    bool connected = false;

    while (!connected) {
      if (connect(local_socket->client_to_server_fd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_un)) == -1) {
        sleep(1); // wait before trying to connect again
        std::cout<<"rank "<<local_rank<<" retrying to connect to local socket..."<<std::endl;
        continue;
      }

      connected = true;
    }

    // send to the server our rank
    if (full_write(local_socket->client_to_server_fd, &local_rank, sizeof(int)) < 0) {
      TORCH_CHECK(false, "an error happened when writing to socket");
      return MUILLM_COMM_SOCKET_WRITE_ERROR;
    }
  }

  return MUILLM_COMM_SUCCESS;
}


// do a barrier using the local socket
muillm_comm_error_t __local_socket_barrier(
    muillm_comm_t* comm
) {
  bool is_server = comm->server_fd != -1;
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  int client_to_server_val = 1;
  int server_to_client_val = 2;

  // the server waits for all other ranks to send a value
  // then sends all ranks something
    if (is_server) {
      // read from all the other ranks
      for (int r = 0; r < local_size; r++) {
        if (r == local_rank) continue;

        int v = 0;
        if (full_read(comm->server_to_client_fds[r], &v, sizeof(int)) < 0) {
          TORCH_CHECK(false, "an error happened when reading from socket during barrier");
          return MUILLM_COMM_SOCKET_READ_ERROR;
        }

        if (v != client_to_server_val) {
          TORCH_CHECK(false, "an error happened when reading from socket during barrier");
          return MUILLM_COMM_SOCKET_READ_ERROR;
        }
      }

      // then send something to all other ranks
      for (int r = 0; r < local_size; r++) {
        if (r == local_rank) continue;

        int v = server_to_client_val;
        if (full_write(comm->server_to_client_fds[r], &v, sizeof(int)) < 0) {
          TORCH_CHECK(false, "an error happened when writing to socket during barrier");
          return MUILLM_COMM_SOCKET_WRITE_ERROR;
        }
      }
    } else {
      // need to send to the server, the server will wait for all
      int v = client_to_server_val;
      if (full_write(comm->client_to_server_fd, &v, sizeof(int)) < 0) {
        TORCH_CHECK(false, "an error happened when writing to socket during barrier");
        return MUILLM_COMM_SOCKET_WRITE_ERROR;
      }

      // wait for the reply from the server
      if (full_read(comm->client_to_server_fd, &v, sizeof(int)) < 0) {
        TORCH_CHECK(false, "an error happened when reading from socket during barrier");
        return MUILLM_COMM_SOCKET_READ_ERROR;
      }

      if (v != server_to_client_val) {
        return MUILLM_COMM_SOCKET_READ_ERROR;
      }
    }
  
  return MUILLM_COMM_SUCCESS;
}

// do a broadcast using the local socker
muillm_comm_error_t __local_socket_broadcast(
    muillm_comm_t* comm,
    int src_local_rank,
    void* ptr,
    size_t byte_count
) {
  bool is_server = comm->server_fd != -1;
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  if (src_local_rank == local_rank) {
    // we are the sender
    // we need to share the value with the other ranks
    if (is_server) {
      // just send to all other ranks
      for (int r = 0; r < local_size; r++) {
        if (r == local_rank) continue;
        if (full_write(comm->server_to_client_fds[r], ptr, byte_count) < 0) {
          TORCH_CHECK(false, "an error happened when writing to socket during broadcast");
          return MUILLM_COMM_SOCKET_WRITE_ERROR;
        }
      }
    } else {
      // need to send to the server, the server will broadcast
      if (full_write(comm->client_to_server_fd, ptr, byte_count) < 0) {
        TORCH_CHECK(false, "an error happened when writing to socket during broadcast");
        return MUILLM_COMM_SOCKET_WRITE_ERROR;
      }
    }
  } else {
    // we are a receiver
    // we need to receive the value from the src rank
    if (is_server) {
      // we need to receive it from the src rank first
      if (full_read(comm->server_to_client_fds[src_local_rank], ptr, byte_count) < 0) {
        TORCH_CHECK(false, "an error happened when reading from socket during broadcast");
        return MUILLM_COMM_SOCKET_READ_ERROR;
      }
      // then broadcast to the others
      for (int r = 0; r < local_size; r++) {
        if (r == local_rank || r == src_local_rank) continue;
        if (full_write(comm->server_to_client_fds[r], ptr, byte_count) < 0) {
          TORCH_CHECK(false, "an error happened when writing to socket during broadcast");
          return MUILLM_COMM_SOCKET_WRITE_ERROR;
        }
      }
    } else {
      // we need to receive from the server
      if (full_read(comm->client_to_server_fd, ptr, byte_count) < 0) {
        TORCH_CHECK(false, "an error happened when reading from socket during broadcast");
        return MUILLM_COMM_SOCKET_READ_ERROR;
      }
    }
  }

  return MUILLM_COMM_SUCCESS;
}

// do a all gather using the local socket
// out_ptr is expected to have enough space for LOCAL_SIZE * byte_count
muillm_comm_error_t __local_socket_all_gather(
    muillm_comm_t* comm,
    void* in_ptr,
    size_t byte_count,
    void* out_ptr
) {
  bool is_server = comm->server_fd != -1;
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  if (is_server) {
    // receive from all ranks
    for (int r = 0; r < local_size; r++) {
      size_t offset = byte_count * r;
      if (r == local_rank) {
        // copy as well from ourself to put the content in in_ptr into out_ptr
        memcpy((int8_t*)out_ptr + offset, in_ptr, byte_count);
      } else {
        // copy the content from the remote rank into out_ptr
        if (full_read(comm->server_to_client_fds[r], (int8_t*)out_ptr + offset, byte_count) < 0) {
          TORCH_CHECK(false, "an error happened when reading from socket during all_gather");
          return MUILLM_COMM_SOCKET_READ_ERROR;
        }
      }
    }

    // send to all other ranks
    for (int r = 0; r < local_size; r++) {
      if (r == local_rank) continue;
      if (full_write(comm->server_to_client_fds[r], out_ptr, byte_count * local_size) < 0) {
        TORCH_CHECK(false, "an error happened when writing to socket during all_gather");
        return MUILLM_COMM_SOCKET_WRITE_ERROR;
      }
    }
  } else {
    // need to send to the server
    if (full_write(comm->client_to_server_fd, in_ptr, byte_count) < 0) {
      TORCH_CHECK(false, "an error happened when writing to socket during all_gather");
      return MUILLM_COMM_SOCKET_WRITE_ERROR;
    }

    // need to receive from the server
    if (full_read(comm->client_to_server_fd, out_ptr, byte_count * local_size) < 0) {
      TORCH_CHECK(false, "an error happened when reading from socket during all_gather");
      return MUILLM_COMM_SOCKET_READ_ERROR;
    }
  }

  return MUILLM_COMM_SUCCESS;
}

void __allocate_locked_shared_cpu_mem(
    muillm_comm_t* comm,
    size_t size,
    void** shm_addr_ptr,
    void** device_ptr_ptr
  ) {
  int local_rank = comm->local_rank;

  int shm_id;
  void *shm_addr;

  *shm_addr_ptr = nullptr;
  *device_ptr_ptr = nullptr;

  if (local_rank == 0) {
    // rank 0 creates the shared memory

    shm_id = shmget(IPC_PRIVATE, size, IPC_CREAT | 0666);
    if (shm_id < 0) {
      // TODO: return error code
      TORCH_CHECK(false, "an error happened when creating shared memory");
      return;
    }
    
    shm_addr = shmat(shm_id, NULL, 0);
    if (shm_addr == (void *) -1) {
      // TODO: return error code
      TORCH_CHECK(false, "an error happened when attaching to shared memory");
      return;
    }

    // make the memory be deleted once all processes have detached from it
    // (memory is automatically detached on process exit)
    if (shmctl(shm_id, IPC_RMID, NULL) != 0) {
      // TODO: return error code
      TORCH_CHECK(false, "an error happened when marking shared memory for deletion");
      return;
    }

    if (mlock(shm_addr, size) != 0) {
      // TODO: return error code
      shm_id = - 1;
      // go to the broadcast
    }
  }

  // get the share memory ID on all ranks
  __local_socket_broadcast(comm, /*src*/ 0, &shm_id, sizeof(int));

  if (shm_id < 0) {
    // TODO: return error code
    TORCH_CHECK(false, "an error happened when getting shared memory id");
    return;
  }

  if (local_rank != 0) {
    shm_addr = shmat(shm_id, NULL, 0);
    if (shm_addr == (void *) -1) {
      // TODO: return error code
      TORCH_CHECK(false, "an error happened when attaching to shared memory");
      return;
    }
  }

  // register the memory for use with HIP
  if (hipHostRegister(shm_addr, size, hipHostRegisterPortable | hipHostRegisterMapped) != hipSuccess) {
    // TODO: return error code
    TORCH_CHECK(false, "an error happened when registering shared memory with HIP");
    return;
  }

  // get the device pointer after registration
  if (hipHostGetDevicePointer((void**)device_ptr_ptr, shm_addr, 0) != hipSuccess) {
    // TODO: return error code
    TORCH_CHECK(false, "an error happened when getting device pointer for shared memory");
    return;
  }
  
  // return
  *shm_addr_ptr = shm_addr;
}

void __deallocate_locked_shared_cpu_mem(
    muillm_comm_t* comm,
    void* host_addr
  ) {
  int local_rank = comm->local_rank;

  if (hipHostUnregister(host_addr) != hipSuccess) {
    // TODO: return error code
    TORCH_CHECK(false, "an error happened when unregistering shared memory from HIP");
    return;
  }

  if (shmdt(host_addr) != 0) {
    TORCH_CHECK(false, "an error happened when detaching from shared memory");
    return;
  }
}

#ifndef __MUILLM_COMM_P2P_HPP__
#define __MUILLM_COMM_P2P_HPP__

typedef struct muillm_comm_p2p_buffer_set {
  void* buffers[MUILLM_COMM_MAX_GPUS];
  uint32_t* counters_host;
  uint32_t* counters;
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
  hipEvent_t cache_flush_event;

  // indicator whether we can skip the cache flush event
  bool cant_skip_cache_flush_event;

  muillm_gpu_info_t* gpu_info;
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

muillm_comm_error_t muillm_comm_p2p_get_buffers(
  muillm_comm_p2p_t* comm,
  size_t count,
  muillm_comm_datatype_t datatype,
  void*** buffers,
  hipStream_t stream
);

#endif // __MUILLM_COMM_P2P_HPP__

#include <hip/hip_runtime.h>

#include <stdint.h>

#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <errno.h>
#include <poll.h>

#include <iostream>

static muillm_comm_error_t __mui_gpu_barrier(
  muillm_comm_p2p_t* comm,
  hipStream_t stream
);

#define MUILLM_COMM_INITIAL_BUFFER_CAPACITY (256 * 1024 * 1024) // 256MiB

static muillm_comm_error_t __free_buffer_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_buffer_set_t* buffer_set
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  muillm_comm_error_t error;

  // we need to synchronize the ranks and block the  CPU so that we can deallocate
  // the previous receive buffers

  // synchronize to make sure no GPU is going to reference the previous memory
  if (hipDeviceSynchronize() != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // make sure all CPUs have synchronized their GPUs
  if ((error =__local_socket_barrier(comm)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when doing barrier");
    return error;
  }

  // close all the previous mappings
  for (int d = 0; d < local_size; d++) {
    if (d == local_rank) continue;
    if (buffer_set->buffers[d] == nullptr) continue;

    if (hipIpcCloseMemHandle(buffer_set->buffers[d]) != hipSuccess) {
      // failed
      TORCH_CHECK(false, "an error happened when closing IPC memory handle");
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }

  // make sure all memory mappings are closed before we free the memory
  if ((error =__local_socket_barrier(comm)) != MUILLM_COMM_SUCCESS) {
    return error;
  }

  // deallocate the previous memory
  if (buffer_set->buffers[local_rank] != nullptr) {
    if (hipFree(buffer_set->buffers[local_rank]) != hipSuccess) {
      TORCH_CHECK(false, "an error happened when freeing GPU memory");
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }

  // free the counters memory as well
  if (buffer_set->counters_host != nullptr) {
    __deallocate_locked_shared_cpu_mem(
      comm,
      buffer_set->counters_host
    );
  }

  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __allocate_shared_gpu_mem(
  muillm_comm_p2p_t* comm,
  size_t capacity,
  void** ptrs
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  void* ptr = nullptr;
  if (hipMalloc((void**)&ptr, capacity) != hipSuccess || ptr == nullptr) {
    TORCH_CHECK(false, "an error happened when allocating shared GPU memory");
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
  
  ptrs[local_rank] = ptr;

  // get the memory pointers from other processes

  hipIpcMemHandle_t ipcHandle;
  if (hipIpcGetMemHandle(&ipcHandle, ptr) != hipSuccess) {
    TORCH_CHECK(false, "an error happened when getting IPC memory handle");
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  hipIpcMemHandle_t* allMemHandles = new hipIpcMemHandle_t[local_size];

  // gather all memory handles
  __local_socket_all_gather(comm, &ipcHandle, sizeof(hipIpcMemHandle_t), allMemHandles);

  // get the remote pointers
  for (int d = 0; d < local_size; d++) {
    if (d != local_rank) {
      // need to open the memory handle
      void* recv_ptr = nullptr;
      // import the memory mapping on the current GPU
      if (hipIpcOpenMemHandle(&recv_ptr, allMemHandles[d], hipIpcMemLazyEnablePeerAccess) != hipSuccess) {
        // failed
        TORCH_CHECK(false, "an error happened when opening IPC memory handle");
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
      if (recv_ptr == nullptr) {
        TORCH_CHECK(false, "an error happened when opening IPC memory handle");
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
      ptrs[d] = recv_ptr;
    }
  }

  // we don't need this array anymore
  delete[] allMemHandles;

  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __ensure_buffer_set_capacity(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_buffer_set_t* buffer_set,
  size_t capacity,
  hipStream_t stream
) {
  if (capacity <= buffer_set->capacity) {
    // the buffers are big enough
    return MUILLM_COMM_SUCCESS;
  }

  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  std::cout<<"rank "<<local_rank<<" reallocating buffers for capacity "<<capacity<<"..."<<std::endl;

  muillm_comm_error_t error;

  // we will import the memory mappings for that specific GPU
  if (hipSetDevice(local_rank) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // free the memory mappings and so on
  if ((error =__free_buffer_set(comm, buffer_set)) != MUILLM_COMM_SUCCESS) {
    return error;
  }

  // allocate counters memory
  // we need it to be on the CPU side so that there is no coherency issues between GPUs
  // (need correct fine-grained atomic operations)
  __allocate_locked_shared_cpu_mem(
    comm,
    sizeof(uint64_t) * MUILLM_MAX_GPUS, // alloc 8 bytes even though we use only 4
    (void**) &buffer_set->counters_host,
    (void**) &buffer_set->counters
  );

  // initialize the counters to 0
  if (local_rank == 0) {
    // the counters are shared, so only one rank needs to initialize them
    // __allocate_shared_gpu_mem after will guarantee all ranks see the updated value
    if (hipMemset(buffer_set->counters, 0, sizeof(uint64_t) * MUILLM_MAX_GPUS) != hipSuccess) {
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }

  // allocate new buffers
  capacity = __next_power_of_2(capacity);

  if ((error = __allocate_shared_gpu_mem(comm, capacity, (void**)buffer_set->buffers)) != MUILLM_COMM_SUCCESS) {
    return error;
  }

  // synchronize the device
  if (hipDeviceSynchronize() != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // all buffer allocations succeeded
  buffer_set->capacity = capacity;

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t muillm_comm_p2p_get_buffer_set(
  muillm_comm_p2p_t* comm,
  size_t capacity,
  muillm_comm_p2p_buffer_set_t** buffer_set,
  hipStream_t stream
) {
  
  muillm_comm_error_t muillm_error;

  if ((muillm_error = __ensure_buffer_set_capacity(comm, comm->first_buffers, capacity, stream)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  // always return the current first buffer set
  *buffer_set = comm->first_buffers;

  // swap buffer sets for next time
  muillm_comm_p2p_buffer_set_t* tmp = comm->first_buffers;
  comm->first_buffers = comm->second_buffers;
  comm->second_buffers = tmp;

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t muillm_comm_p2p_get_buffer_set(
  muillm_comm_p2p_t* comm,
  size_t count,
  muillm_comm_datatype_t datatype,
  muillm_comm_p2p_buffer_set_t** buffer_set,
  hipStream_t stream
) {
  size_t size = __comm_size(datatype, count);
  return muillm_comm_p2p_get_buffer_set(comm, size, buffer_set, stream);
}

muillm_comm_error_t muillm_comm_p2p_get_next_buffer_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_buffer_set_t** buffer_set
) {
  // always return the second buffer set
  *buffer_set = comm->second_buffers;

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t muillm_comm_p2p_get_buffers(
  muillm_comm_p2p_t* comm,
  size_t count,
  muillm_comm_datatype_t datatype,
  void*** buffers,
  hipStream_t stream
) {

  muillm_comm_p2p_buffer_set_t* buffer_set;
  muillm_comm_error_t error = muillm_comm_p2p_get_buffer_set(
    comm,
    count,
    datatype,
    &buffer_set,
    stream
  );

  if (error != MUILLM_COMM_SUCCESS) {
    return error;
  }

  *buffers = (void**) buffer_set->buffers;

  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __init_buffer_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_buffer_set_t** buffer_set_ptr,
  hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  muillm_comm_p2p_buffer_set_t* buffer_set = new muillm_comm_p2p_buffer_set_t;
  buffer_set->capacity = 0;

  if (buffer_set == nullptr) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  for (int i = 0; i < MUILLM_COMM_MAX_GPUS; i++) {
    buffer_set->buffers[i] = nullptr;
  }
  buffer_set->counters_host = nullptr;
  buffer_set->counters = nullptr;

  if (hipSetDevice(local_rank) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // ensure a certain good initial size
  muillm_comm_error_t muillm_error;
  if ((muillm_error = __ensure_buffer_set_capacity(comm, buffer_set, MUILLM_COMM_INITIAL_BUFFER_CAPACITY, stream)) != MUILLM_COMM_SUCCESS) {
    *buffer_set_ptr = nullptr;
    return muillm_error;
  }

  *buffer_set_ptr = buffer_set;
  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __init_p2p_recv(
  muillm_comm_p2p_t* comm
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  // enable peer to peer
  if (hipSetDevice(local_rank) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  for (int d = 0; d < local_size; d++) {
    if (d == local_rank) continue;
    if (hipDeviceEnablePeerAccess(d, 0) != hipSuccess) {
      // TODO: return error
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t muillm_comm_p2p_init_comm(
  int world_size,
  int local_size,
  int rank,
  int local_rank,
  const muillm_comm_local_socket_t* local_socket,
  muillm_comm_p2p_t** comm_ptr,
  hipStream_t stream
) {
  if (world_size != local_size) {
    // we currently ony support single machine, so
    // we should fail
    return MUILLM_COMM_UNSUPPORTED_SIZE;
  }

  muillm_comm_error_t muillm_error;

  muillm_comm_method_t transfer_method = MUILLM_COMM_METHOD_P2P_TRANSFER;

  // create the comm object
  muillm_comm_p2p_t* comm = new muillm_comm_p2p_t;
  comm->transfer_method = transfer_method;

  comm->world_size = world_size;
  comm->local_size = local_size;
  comm->rank = rank;
  comm->local_rank = local_rank;

  comm->signal_host = nullptr;
  comm->signal = nullptr;
  comm->signal_seq_no = 0;

  // merge in local socket
  comm->server_fd = local_socket->server_fd;
  comm->client_to_server_fd = local_socket->client_to_server_fd;
  comm->server_to_client_fds = local_socket->server_to_client_fds;

  // set the device
  if (hipSetDevice(local_rank) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // get the gpu info
  muillm_gpu_info_t* gpu_info = new muillm_gpu_info_t;
  if (muillm_detect_gpu_properties(local_rank, gpu_info) != MUILLM_SUCCESS) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  comm->gpu_info = gpu_info;

  // check that signal memory is supported
  int signals_supported;
  if (hipDeviceGetAttribute(&signals_supported, hipDeviceAttributeCanUseStreamWaitValue, 0) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  if (!signals_supported) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // setup p2p 
  __init_p2p_recv(comm);

  // by default, do not skip the cache flush
  // but MI300 and successors don't need it apparently
  comm->cant_skip_cache_flush_event = comm->gpu_info->arch < MUILLM_GPU_ARCH_MI300;

  // allocate cache flush event
  if (hipEventCreateWithFlags(&comm->cache_flush_event, hipEventDisableTiming | hipEventReleaseToSystem) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // allocate signal memory
  __allocate_locked_shared_cpu_mem(
    comm,
    sizeof(uint64_t), // alloc 8 bytese even though we use only 4
    (void**) &comm->signal_host,
    (void**) &comm->signal
  );

  if (comm->signal_host == nullptr || comm->signal == nullptr) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
  // initialize to 0
  if (hipMemset(comm->signal, 0, sizeof(uint64_t)) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // initialize buffer sets
  if ((muillm_error = __init_buffer_set(comm, &comm->first_buffers, stream)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  if ((muillm_error = __init_buffer_set(comm, &comm->second_buffers, stream)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  // set the device
  if (hipSetDevice(local_rank) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  if (hipDeviceSynchronize() != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // make sure every GPU has opened the memory before returning
  if ((muillm_error =__local_socket_barrier(comm)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  // return the comm object
  *comm_ptr = comm;

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t __mui_stream_inc_value(hipStream_t stream, uint32_t* signal);

muillm_comm_error_t __mui_stream_inc_wait_value(hipStream_t stream, uint32_t* signal, uint32_t seq_no);

static muillm_comm_error_t __mui_gpu_barrier(muillm_comm_p2p_t* comm, hipStream_t stream) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  if (comm->signal != nullptr) {
    comm->signal_seq_no += local_size;
    uint64_t seq_no = comm->signal_seq_no;

    // GPU barrier: all GPUs wait on each other
    if (comm->cant_skip_cache_flush_event) {
      // on MI100, we get a crash if not putting this event here
      // record an event to flush caches
      if (hipEventRecord(comm->cache_flush_event, stream) != hipSuccess) {
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }

    // write the values
    if ((muillm_error = __mui_stream_inc_wait_value(stream, comm->signal, seq_no)) != MUILLM_COMM_SUCCESS) {
      return muillm_error;
    }
  } else {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t __muillm_gpu_copy(void* dst, const void* src, size_t count, hipStream_t stream);

// torch extension

#include <tuple>

#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_fp16.h>

#define META_DIM 4

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void* all2all_comm_init(int world_size, int rank) {
  // assumme local_size == world_size for now
  int local_size = world_size;
  int local_rank = rank;

  muillm_comm_error_t muillm_error;

  // establish the local socket connection
  muillm_comm_local_socket_t local_socket;
  if ((muillm_error = __open_local_socket(local_size, local_rank, &local_socket)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when opening local socket");
    return (void*) nullptr;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  muillm_comm_p2p_t* comm_ptr = nullptr;
  muillm_error = muillm_comm_p2p_init_comm(
    world_size,
    local_size,
    rank,
    local_rank,
    &local_socket,
    (muillm_comm_p2p_t**) &comm_ptr,
    stream
  );

  TORCH_CHECK(muillm_error == MUILLM_COMM_SUCCESS, "an error happened when initializing mui comm");

  return (void*) comm_ptr;
}

void all2all_dispatch_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ indices,
    uint32_t* __restrict__ send_counts,
    uint32_t* __restrict__ send_offsets,
    // counters for the different ranks
    uint32_t* counters,
    // local counter to clear for next use
    uint32_t* next_local_counters,
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int local_size,
    int local_rank
);

void all2all_dispatch_pack_send_buffers_fp32(
    hipStream_t stream,
    const float* __restrict__ x,
    const int32_t* __restrict__ indices,
    uint32_t* __restrict__ send_offsets,
    float* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int hidden_dim,
    int buff_meta_offset,
    int local_size,
    int local_rank
);

void all2all_dispatch_pack_send_buffers_fp16(
    hipStream_t stream,
    const half* __restrict__ x,
    const int32_t* __restrict__ indices,
    uint32_t* __restrict__ send_offsets,
    half* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int hidden_dim,
    int buff_meta_offset,
    int local_size,
    int local_rank
);

void all2all_dispatch_unpack_fp16(
    hipStream_t stream,
    const half* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    half* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset,
    int max_total_recv
);

void all2all_dispatch_unpack_fp32(
    hipStream_t stream,
    const float* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    float* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset,
    int max_total_recv
);

// dispatch
// outputs:
// expert_num_tokens: shape [num_local_experts]
// expert_y: shape [num_local_experts, max_recv, hidden_dim]
// expert_meta: shape [num_local_experts, max_recv, META_DIM] (expert_id, src_rank, src_token_id, topk_offset)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> all2all_comm_dispatch(
  void* comms,
  torch::Tensor& x, // shape [num_tokens, hidden_dim]
  torch::Tensor& indices, // shape [num_tokens, experts_per_token]
  int num_local_experts,
  int max_recv
) {
  muillm_comm_p2p_t* comm = (muillm_comm_p2p_t*) comms;

  CHECK_INPUT(x);
  CHECK_INPUT(indices);

  auto device = x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  int local_size = comm->local_size;
  int local_rank = comm->local_rank;


  int num_tokens = x.size(0);
  int hidden_dim = x.size(1);
  int num_experts_per_token = indices.size(1);

  // total number of tokens a rank has to combine is at most this much.
  // we use this to allocate the send buffer with the same size on all ranks
  int max_total_recv = max_recv * num_experts_per_token;
  // but for this rank, the actual number of tokens to send is:
  int total_send = num_tokens * num_experts_per_token;

  auto dtype = x.dtype();

  size_t dtype_size = 0;
  if (dtype == torch::kFloat16) {
    dtype_size = 2;
  } else if (dtype == torch::kBFloat16) {
    dtype_size = 2;
  } else if (dtype == torch::kFloat32) {
    dtype_size = 4;
  } else {
    TORCH_CHECK(false, "datatype must be float16, bfloat16 or float32");
  }

  //
  // First we compute the send offsets for each rank
  //

  auto send_counts_options = at::TensorOptions()
                            .dtype(torch::kInt32)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // TODO: do not allocate send_counts, we can just use shared memory in the kernel
  // send_counts: shape [local_size]
  auto send_counts = torch::zeros({local_size}, send_counts_options);
  auto send_offsets = torch::empty({local_size}, send_counts_options);

  // we align the metadata pointer to 4k for better performance
  // so we align up the data size as such as the metadata pointer will be right after the data pointer
  size_t size_data = max_total_recv * hidden_dim * dtype_size;
  size_t aligned_size_data = ALIGN_UP(size_data, 4096);
  size_t size_metadata = max_total_recv * META_DIM * sizeof(int32_t);
  size_t capacity = aligned_size_data + size_metadata;

  muillm_comm_error_t muillm_error;

  // get the next reduction buffer to get the counters to clear
  muillm_comm_p2p_buffer_set_t* next_buffer_set = nullptr;

  // we have to do this call before flipping the buffer sets with muillm_comm_p2p_get_buffer_set
  if (muillm_comm_p2p_get_next_buffer_set(comm, &next_buffer_set) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting next buffer set");
  }
  // get reduction buffer set
  muillm_comm_p2p_buffer_set_t* buffer_set = nullptr;

  // this call flips the buffer sets
  if ((muillm_error = muillm_comm_p2p_get_buffer_set(comm, capacity, &buffer_set, stream)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting buffer set");
  }

  //
  // We wait for all the GPUs to be done with sending data
  //
  if ((muillm_error = __mui_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when doing barrier");
  }

  uint32_t* counters = (uint32_t*)buffer_set->counters;
  uint32_t* next_counters = (uint32_t*) next_buffer_set->counters;

  all2all_dispatch_compute_send_counts(
    stream,
    (const int32_t*)indices.data_ptr(),
    (uint32_t*)send_counts.data_ptr(),
    (uint32_t*)send_offsets.data_ptr(),
    counters,
    next_counters,
    num_local_experts,
    num_tokens,
    num_experts_per_token,
    local_size,
    local_rank
  );

  //
  // Second we pack and send the data to the experts
  //
  int buff_meta_offset = aligned_size_data;

  if (dtype == torch::kFloat32) {
    // buffers will be nullptr if local_size < 8, but it's ok to pass nullptr to the kernel
    all2all_dispatch_pack_send_buffers_fp32(
      stream,
      (const float*)x.data_ptr(),
      (const int32_t*)indices.data_ptr(),
      (uint32_t*)send_offsets.data_ptr(),
      (float*)buffer_set->buffers[0], // send_buf0
      (float*)buffer_set->buffers[1], // send_buf1
      (float*)buffer_set->buffers[2], // send_buf2
      (float*)buffer_set->buffers[3], // send_buf3
      (float*)buffer_set->buffers[4], // send_buf4
      (float*)buffer_set->buffers[5], // send_buf5
      (float*)buffer_set->buffers[6], // send_buf6
      (float*)buffer_set->buffers[7], // send_buf7
      num_local_experts,
      num_tokens,
      num_experts_per_token,
      hidden_dim,
      buff_meta_offset,
      local_size,
      local_rank
    );
  } else if (dtype == torch::kFloat16) {
    // buffers will be nullptr if local_size < 8, but it's ok to pass nullptr to the kernel
    all2all_dispatch_pack_send_buffers_fp16(
      stream,
      (const half*)x.data_ptr(),
      (const int32_t*)indices.data_ptr(),
      (uint32_t*)send_offsets.data_ptr(),
      (half*)buffer_set->buffers[0], // send_buf0
      (half*)buffer_set->buffers[1], // send_buf1
      (half*)buffer_set->buffers[2], // send_buf2
      (half*)buffer_set->buffers[3], // send_buf3
      (half*)buffer_set->buffers[4], // send_buf4
      (half*)buffer_set->buffers[5], // send_buf5
      (half*)buffer_set->buffers[6], // send_buf6
      (half*)buffer_set->buffers[7], // send_buf7
      num_local_experts,
      num_tokens,
      num_experts_per_token,
      hidden_dim,
      buff_meta_offset,
      local_size,
      local_rank
    );
  } else {
    TORCH_CHECK(false, "unsupported data type");
  }

  //
  // We wait for all the GPUs to be done with sending data
  //
  if ((muillm_error = __mui_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when doing barrier");
  }

  //
  // Third, we unpack into the output tensors
  //
  auto expert_num_tokens_options = at::TensorOptions()
                            .dtype(torch::kInt32)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto expert_meta_options = at::TensorOptions()
                            .dtype(torch::kInt32)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto expert_y_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // TODO: do not zero, we can just use shared memory in the kernel
  torch::Tensor expert_num_tokens = torch::zeros({num_local_experts}, expert_num_tokens_options);
  torch::Tensor expert_meta = torch::empty({num_local_experts, max_recv, META_DIM}, expert_meta_options);
  torch::Tensor expert_x = torch::empty({num_local_experts, max_recv, hidden_dim}, expert_y_options);

  int local_expert_offset = local_rank * num_local_experts;

  void* recv_buf = buffer_set->buffers[local_rank];
  int32_t* recv_meta = (int32_t*) ((uint8_t*)recv_buf + buff_meta_offset);

  if (dtype == at::kFloat) {
    all2all_dispatch_unpack_fp32(
      stream,
      (const float*) recv_buf,
      (const int32_t*) recv_meta,
      (int32_t*) expert_num_tokens.data_ptr(),
      (float*) expert_x.data_ptr(),
      (int32_t*) expert_meta.data_ptr(),
      &counters[local_rank], // total_recv
      hidden_dim,
      max_recv,
      local_expert_offset,
      max_total_recv
    );
  } else if (dtype == at::kHalf) {
    all2all_dispatch_unpack_fp16(
      stream,
      (const half*) recv_buf,
      (const int32_t*) recv_meta,
      (int32_t*) expert_num_tokens.data_ptr(),
      (half*) expert_x.data_ptr(),
      (int32_t*) expert_meta.data_ptr(),
      &counters[local_rank], // total_recv
      hidden_dim,
      max_recv,
      local_expert_offset,
      max_total_recv
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type");
  }

  // return expert_num_tokens, expert_x, expert_meta
  return std::make_tuple(expert_num_tokens, expert_x, expert_meta);
}


void all2all_compute_fp32(
  hipStream_t stream,
  const int32_t* __restrict__ expert_num_tokens,
  const float* __restrict__ expert_x,
  float* __restrict__ expert_y,
  int num_local_experts,
  int max_recv,
  int hidden_dim,
  int rank
);

void all2all_compute_fp16(
  hipStream_t stream,
  const int32_t* __restrict__ expert_num_tokens,
  const half* __restrict__ expert_x,
  half* __restrict__ expert_y,
  int num_local_experts,
  int max_recv,
  int hidden_dim,
  int rank
);

// output: expert_y shape [num_local_experts, max_recv, hidden_dim]
at::Tensor all2all_compute(
  torch::Tensor& expert_num_tokens, // shape [num_local_experts]
  torch::Tensor& expert_x, // shape [num_local_experts, max_recv, hidden_dim]
  int rank
) {
  CHECK_INPUT(expert_x);

  auto device = expert_x.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  int num_local_experts = expert_x.size(0);
  int max_recv = expert_x.size(1);
  int hidden_dim = expert_x.size(2);

  auto dtype = expert_x.dtype();

  auto expert_y_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto expert_y = torch::empty({num_local_experts, max_recv, hidden_dim}, expert_y_options);

  if (dtype == at::kFloat) {
    all2all_compute_fp32(
      stream,
      (const int32_t*) expert_num_tokens.data_ptr(),
      (const float*) expert_x.data_ptr(),
      (float*) expert_y.data_ptr(),
      num_local_experts,
      max_recv,
      hidden_dim,
      rank
    );
  } else if (dtype == at::kHalf) {
    all2all_compute_fp16(
      stream,
      (const int32_t*) expert_num_tokens.data_ptr(),
      (const half*) expert_x.data_ptr(),
      (half*) expert_y.data_ptr(),
      num_local_experts,
      max_recv,
      hidden_dim,
      rank
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type");
  }

  return expert_y;
}

void all2all_combine_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    uint32_t* __restrict__ send_counts, // shape [world_size]
    uint32_t* __restrict__ send_offsets, // shape [world_size]
    // counters for the different ranks
    uint32_t* counters,
    // local counter to clear for next use
    uint32_t* next_counters,
    int num_local_experts,
    int max_recv,
    int local_size,
    int local_rank
);

void all2all_combine_pack_send_buffers_fp16(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const half* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int32_t* __restrict__ send_offsets, // shape [world_size]
    half* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int max_recv,
    int hidden_dim,
    int buff_meta_offset
);

void all2all_combine_pack_send_buffers_fp32(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const float* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int32_t* __restrict__ send_offsets, // shape [world_size]
    float* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int max_recv,
    int hidden_dim,
    int buff_meta_offset
);

void all2all_combine_unpack_fp32(
    hipStream_t stream,
    const float* __restrict__ recv_buf, // shape [total_recv, hidden_sim]
    const int32_t* __restrict__ recv_meta, // shape [total_recv, META_DIM]
    const float* __restrict__ weights, // shape [num_tokens, experts_per_token]
    float* __restrict__ scaled_expert_outputs, // shape [num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ output, // shape [max_num_tokens, hidden_dim]
    int hidden_dim,
    int total_recv,
    int num_tokens,
    int experts_per_token
);

void all2all_combine_unpack_fp16(
    hipStream_t stream,
    const half* __restrict__ recv_buf, // shape [total_recv, hidden_sim]
    const int32_t* __restrict__ recv_meta, // shape [total_recv, META_DIM]
    const float* __restrict__ weights, // shape [num_tokens, experts_per_token]
    float* __restrict__ scaled_expert_outputs, // shape [num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ output, // shape [max_num_tokens, hidden_dim]
    int hidden_dim,
    int total_recv,
    int num_tokens,
    int experts_per_token
);

// combine
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> all2all_comm_combine(
  void* comms,
  torch::Tensor& weights, // shape [num_tokens, experts_per_token]
  torch::Tensor& expert_meta, // shape [num_local_experts, max_recv, meta_dim] (expert_id, src_rank, src_token_id, topk_offset)
  torch::Tensor& expert_y, // shape [num_local_experts, max_recv, hidden_dim]
  torch::Tensor& expert_num_tokens, // shape [num_local_experts]
  torch::Tensor& out_tokens // shape [max_num_tokens, hidden_dim]
) {
  muillm_comm_p2p_t* comm = (muillm_comm_p2p_t*) comms;

  CHECK_INPUT(weights);
  CHECK_INPUT(expert_meta);
  CHECK_INPUT(expert_y);
  CHECK_INPUT(expert_num_tokens);
  CHECK_INPUT(out_tokens);

  auto device = expert_meta.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  int num_tokens = weights.size(0);
  int num_experts_per_token = weights.size(1);
  int num_local_experts = expert_num_tokens.size(0);
  int max_recv = expert_meta.size(1);
  int meta_dim = expert_meta.size(2);
  int hidden_dim = expert_y.size(2);
  int max_num_tokens = out_tokens.size(0);

  // total number of tokens a rank has to combine is at most this much.
  // we use this to allocate the send buffer with the same size on all ranks
  int max_total_send = max_num_tokens * num_experts_per_token * local_size;
  // but for this rank, the actual number of tokens to combine is:
  int total_recv = num_tokens * num_experts_per_token;

  if (meta_dim != META_DIM) {
    TORCH_CHECK(false, "meta_dim must be ", META_DIM);
  }
  auto dtype = expert_y.dtype();
  auto meta_dtype = expert_meta.dtype();

  if (meta_dtype != torch::kInt32) {
    TORCH_CHECK(false, "meta_dtype must be int32");
  }

  if (expert_num_tokens.dtype() != torch::kInt32) {
    TORCH_CHECK(false, "expert_num_tokens dtype must be int32");
  }

  size_t dtype_size = 0;
  if (dtype == torch::kFloat16) {
    dtype_size = 2;
  } else if (dtype == torch::kBFloat16) {
    dtype_size = 2;
  } else if (dtype == torch::kFloat32) {
    dtype_size = 4;
  } else {
    TORCH_CHECK(false, "datatype must be float16, bfloat16 or float32");
  }

  //
  // First we compute the send offsets for each rank
  //

  auto send_counts_options = at::TensorOptions()
                            .dtype(torch::kInt32)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // TODO: do not allocate send_counts, we can just use shared memory in the kernel
  // send_counts: shape [local_size]
  auto send_counts = torch::zeros({local_size}, send_counts_options);
  auto send_offsets = torch::empty({local_size}, send_counts_options);

  // we align the metadata pointer to 4k for better performance
  // so we align up the data size as such as the metadata pointer will be right after the data pointer
  size_t size_data = max_total_send * hidden_dim * dtype_size;
  size_t aligned_size_data = ALIGN_UP(size_data, 4096);
  size_t size_metadata = max_total_send * META_DIM * sizeof(int32_t);
  size_t capacity = aligned_size_data + size_metadata;

  muillm_comm_error_t muillm_error;

  // get the next reduction buffer to get the counters to clear
  muillm_comm_p2p_buffer_set_t* next_buffer_set = nullptr;

  // we have to do this call before flipping the buffer sets with muillm_comm_p2p_get_buffer_set
  if (muillm_comm_p2p_get_next_buffer_set(comm, &next_buffer_set) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting next buffer set");
  }
  // get reduction buffer set
  muillm_comm_p2p_buffer_set_t* buffer_set = nullptr;

  // this call flips the buffer sets
  if ((muillm_error = muillm_comm_p2p_get_buffer_set(comm, capacity, &buffer_set, stream)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting buffer set");
  }

  //
  // We wait for all the GPUs to be done with sending data
  //
  if ((muillm_error = __mui_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when doing barrier");
  }

  uint32_t* counters = (uint32_t*)buffer_set->counters;
  uint32_t* next_counters = (uint32_t*) next_buffer_set->counters;
  
  all2all_combine_compute_send_counts(
    stream,
    (const int32_t*) expert_num_tokens.data_ptr(),
    (const int32_t*) expert_meta.data_ptr(),
    (uint32_t*) send_counts.data_ptr(),
    (uint32_t*) send_offsets.data_ptr(),
    counters,
    next_counters,
    num_local_experts,
    max_recv,
    local_size,
    local_rank
  );

  //
  // then we send/receive the data and meta data
  //
  int buff_meta_offset = aligned_size_data;

  if (dtype == torch::kFloat16) {
    // buffers will be nullptr if local_size < 8, but it's ok to pass nullptr to the kernel
    all2all_combine_pack_send_buffers_fp16(
      stream,
      (const int32_t*) expert_num_tokens.data_ptr(),
      (const int32_t*) expert_meta.data_ptr(),
      (const half*) expert_y.data_ptr(),
      (int32_t*) send_offsets.data_ptr(),
      (half*) buffer_set->buffers[0],
      (half*) buffer_set->buffers[1],
      (half*) buffer_set->buffers[2],
      (half*) buffer_set->buffers[3],
      (half*) buffer_set->buffers[4],
      (half*) buffer_set->buffers[5],
      (half*) buffer_set->buffers[6],
      (half*) buffer_set->buffers[7],
      num_local_experts,
      max_recv,
      hidden_dim,
      buff_meta_offset
    );
  } else if (dtype == torch::kFloat32) {
    // buffers will be nullptr if local_size < 8, but it's ok to pass nullptr to the kernel
    all2all_combine_pack_send_buffers_fp32(
      stream,
      (const int32_t*) expert_num_tokens.data_ptr(),
      (const int32_t*) expert_meta.data_ptr(),
      (const float*) expert_y.data_ptr(),
      (int32_t*) send_offsets.data_ptr(),
      (float*) buffer_set->buffers[0],
      (float*) buffer_set->buffers[1],
      (float*) buffer_set->buffers[2],
      (float*) buffer_set->buffers[3],
      (float*) buffer_set->buffers[4],
      (float*) buffer_set->buffers[5],
      (float*) buffer_set->buffers[6],
      (float*) buffer_set->buffers[7],
      num_local_experts,
      max_recv,
      hidden_dim,
      buff_meta_offset
    );
  } else {
    TORCH_CHECK(false, "datatype must be float16 for now");
  }

  //
  // We wait for all the GPUs to be done with sending data
  //
  if ((muillm_error = __mui_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when doing barrier");
  }

  //
  // Finally, we need to combine the received data
  //

  auto scaled_expert_output_options = at::TensorOptions()
                            .dtype(at::kFloat)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  // allocate empty tensor to hold scaled expert outputs shape [num_tokens, experts_per_token, hidden_dim]
  // makes it possible to avoid accumulations with atomicAdd
  torch::Tensor scaled_expert_outputs = torch::empty({num_tokens, num_experts_per_token, hidden_dim}, scaled_expert_output_options);

  void* recv_buf = buffer_set->buffers[local_rank];
  int32_t* recv_meta = (int32_t*) ((uint8_t*)recv_buf + buff_meta_offset);

  if (dtype == at::kFloat) {
    all2all_combine_unpack_fp32(
      stream,
      (const float*) recv_buf,
      recv_meta,
      (const float*) weights.data_ptr(),
      (float*) scaled_expert_outputs.data_ptr(),
      (float*) out_tokens.data_ptr(),
      hidden_dim,
      total_recv,
      num_tokens,
      num_experts_per_token
    );
  } else if (dtype == at::kHalf) {
    all2all_combine_unpack_fp16(
      stream,
      (const half*) recv_buf,
      recv_meta,
      (const float*) weights.data_ptr(),
      (float*) scaled_expert_outputs.data_ptr(),
      (half*) out_tokens.data_ptr(),
      hidden_dim,
      total_recv,
      num_tokens,
      num_experts_per_token
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type");
  }

  // return tuple with send_counts, send_offsets, out_tokens
  return std::make_tuple(send_counts, send_offsets, out_tokens);
}
"""

COMM_KERNELS_CUDA_CODE = """
#include <stdint.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <iostream>

typedef enum muillm_comm_error {
  MUILLM_COMM_SUCCESS = 0,

  MUILLM_COMM_UNSUPPORTED_SIZE,

  MUILLM_COMM_SOCKET_CREATION_FAILED,
  MUILLM_COMM_SOCKET_BIND_FAILED,
  MUILLM_COMM_SOCKET_LISTEN_FAILED,
  MUILLM_COMM_SOCKET_ACCEPT_FAILED,
  MUILLM_COMM_SOCKET_CONNECT_FAILED,

  MUILLM_COMM_SOCKET_READ_ERROR,
  MUILLM_COMM_SOCKET_WRITE_ERROR,

  MUILLM_COMM_UNKNOWN_ERROR = -1
} muillm_comm_error_t;

#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_ULL 512

#define FULL_MASK32 0xffffffff
#define FULL_MASK64 0xffffffffffffffff

#ifdef  __CUDA_ARCH__
#define __xx_shfl(mask, val, offset) __shfl_sync(mask, val, offset)
#elif defined(__HIP_PLATFORM_AMD__) // AMD
#define __xx_shfl(mask, val, offset) __shfl(val, offset)
#else
#error "Unsupported compiler"
#endif


// warp broadcast using shuffle
__inline__ __device__ int __warp_broadcast(int val) {
  if (warpSize == 32) {
    val = __xx_shfl(FULL_MASK32, val, 0);
  }
  if (warpSize == 64) {
    val = __xx_shfl(FULL_MASK64, val, 0);
  }
  return val;
}

__inline__ __device__ int __block_broadcast(int val) {
  int lane_id = threadIdx.x % warpSize;

  if (THREADS_PER_BLOCK > warpSize) {
    int __shared__ val_shared;
    if (threadIdx.x == 0) {
      val_shared = val;
    }
    __syncthreads();
    if (lane_id == 0) {
      val = val_shared;
    }
  }
  return __warp_broadcast(val);
}

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

__global__ void __muillm_inc_value_p2p_kernel(
  uint32_t* signal
) {
  if (threadIdx.x == 0) {
    atomicAdd_system(signal, 1);
    __threadfence_system();
  }
}

muillm_comm_error_t __mui_stream_inc_value(hipStream_t stream, uint32_t* signal) {
  __muillm_inc_value_p2p_kernel<<<1, 1, 0, stream>>>(signal);
  return MUILLM_COMM_SUCCESS;
}

__device__ void __do_inc_wait_value_p2p(
  volatile uint32_t* signal,
  uint32_t seq_no
) {
  if (threadIdx.x == 0) {
    // increment the value
    uint32_t value = atomicAdd_system((uint32_t*) signal, 1) + 1;
    __threadfence_system();

    // wait for the other ranks
    // we need the comparison to be >= as one GPU might already increment the value before all the other GPUs
    // have seen the previous one
    if (value < seq_no) {
      while (*signal < seq_no) __threadfence_system();
    }
  }
}

__global__ void __muillm_inc_wait_value_p2p_kernel(
  volatile uint32_t* signal,
  uint32_t seq_no
) {
  __do_inc_wait_value_p2p(signal, seq_no);
}

muillm_comm_error_t __mui_stream_inc_wait_value(hipStream_t stream, uint32_t* signal, uint32_t seq_no) {
  __muillm_inc_wait_value_p2p_kernel<<<1, 1, 0, stream>>>(signal, seq_no);
  return MUILLM_COMM_SUCCESS;
}



// each threads can copy 16 bytes
#define BYTES_PER_THREAD 16
#define BYTES_PER_BLOCK (THREADS_PER_BLOCK * BYTES_PER_THREAD)

typedef struct uint32x4{
uint32_t x, y, z, w;
} uint32x4_t;

__global__ void __muillm_copy_p2p_kernel(
  const uint8_t* src_ptr,
  uint8_t* dst_ptr,
  unsigned N
) {
  unsigned i = blockIdx.x * BYTES_PER_BLOCK + (threadIdx.x * BYTES_PER_THREAD);
  if (i + (BYTES_PER_THREAD - 1) < N) {
    // can copy 16 bytes

    const uint32x4_t* src_x16_ptr = (const uint32x4_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr = (uint32x4_t*)(&dst_ptr[i]);
    *dst_x16_ptr = *src_x16_ptr;

    i += BYTES_PER_THREAD;
  } else {
    // non vectorized copy
    for (unsigned b = 0; b < BYTES_PER_THREAD; b++) {
      if (i < N) {
        dst_ptr[i] = src_ptr[i];
        i++;
      }
    }
  }
}

muillm_comm_error_t __muillm_gpu_copy(void* dst, const void* src, size_t count, hipStream_t stream) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(count, BYTES_PER_BLOCK);

  // a copy kernel is faster than a hipMemcpyAsync
  __muillm_copy_p2p_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint8_t*) src,
    (uint8_t*) dst,
    count
  );

  if (hipPeekAtLastError() != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}

#define META_DIM 4

//
// All2All dispatch kernels
//

// TODO: avoid send_counts, do per thread reductions + block reductions
void __global__ all2all_dispatch_compute_send_counts_kernel(
  const int32_t* __restrict__ indices,
  uint32_t* __restrict__ send_counts,
  uint32_t* __restrict__ send_offsets,
  // counters for the different ranks
  uint32_t* counters,
  // local counter to clear for next use
  uint32_t* next_counters,
  int num_local_experts,
  int total_send,
  int local_size,
  int local_rank
) {
  if (local_rank == 0 && threadIdx.x < local_size) {
    // clear the counters for the next use
    next_counters[threadIdx.x] = 0;
  }

  // each thread handles one token expert
  for (int i = threadIdx.x; i < total_send; i += THREADS_PER_BLOCK) {
    // TODO: avoid division if num_local_experts is power of 2
    int32_t dst_expert = indices[i];
    int32_t dst_rank = dst_expert / num_local_experts;
    atomicAdd((uint32_t*)&send_counts[dst_rank], 1);
  }

  __syncthreads();

  // do a single thread prefix sum to compute send offsets
  if (threadIdx.x < local_size) {
    int rank = threadIdx.x;
    uint32_t send_count = send_counts[rank];
    uint32_t send_offset = atomicAdd_system((uint32_t*) &counters[rank], send_count);
    send_offsets[rank] = send_offset;
  }
}

void all2all_dispatch_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ indices,
    uint32_t* __restrict__ send_counts,
    uint32_t* __restrict__ send_offsets,
    // counters for the different ranks
    uint32_t* counters,
    // local counter to clear for next use
    uint32_t* next_counters,
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int local_size,
    int local_rank
) {
  // call kernel to compute send_counts using a single block to do the send offset reduction as well
  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = 1;

  int total_send = num_tokens * num_experts_per_token;

  all2all_dispatch_compute_send_counts_kernel<<<blocks, threads_per_block, 0, stream>>>(
    indices,
    send_counts,
    send_offsets,
    counters,
    next_counters,
    num_local_experts,
    total_send,
    local_size,
    local_rank
  );
}

void __global__ all2all_dispatch_pack_send_buffers_fp32_kernel(
  const float* __restrict__ x,
  const int32_t* __restrict__ indices,
  uint32_t* __restrict__ send_offsets,
  float* __restrict__ send_buf0,
  float* __restrict__ send_buf1,
  float* __restrict__ send_buf2,
  float* __restrict__ send_buf3,
  float* __restrict__ send_buf4,
  float* __restrict__ send_buf5,
  float* __restrict__ send_buf6,
  float* __restrict__ send_buf7,
  int num_local_experts,
  int num_tokens,
  int num_experts_per_token,
  int hidden_dim,
  int buff_meta_offset,
  int rank
) {
  int expert_idx = blockIdx.x;
  int token_idx = blockIdx.y;

  x = &x[token_idx * hidden_dim];

  // determine where to write out
  // TODO: avoid division if num_local_experts is power of 2
  int32_t dst_expert = indices[token_idx * num_experts_per_token + expert_idx];
  int32_t dst_rank = dst_expert / num_local_experts;

  int32_t send_pos = -1;
  if (threadIdx.x == 0) {
    send_pos = (int32_t)atomicAdd((uint32_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos);

  float* send_buffs[8] = {send_buf0, send_buf1, send_buf2, send_buf3, send_buf4, send_buf5, send_buf6, send_buf7};
  float* send_buf = send_buffs[dst_rank];

  // write to send_buf
  float* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  for (int i = threadIdx.x; i < hidden_dim; i += THREADS_PER_BLOCK) {
    send_buf_ptr[i] = x[i];
  }

  // write to send_meta
  int32_t* send_meta = (int32_t*)(((uint8_t*)send_buf) + buff_meta_offset);
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  if (threadIdx.x == 0) {
    send_meta_ptr[0] = dst_expert;
    send_meta_ptr[1] = rank;
    send_meta_ptr[2] = token_idx;
    send_meta_ptr[3] = expert_idx;
  }
}

void all2all_dispatch_pack_send_buffers_fp32(
    hipStream_t stream,
    const float* __restrict__ x,
    const int32_t* __restrict__ indices,
    uint32_t* __restrict__ send_offsets,
    float* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int hidden_dim,
    int buff_meta_offset,
    int local_size,
    int local_rank
) {
  // call kernel to do the packing, with a 2D grid of size (num_experts_per_token, num_tokens)
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(num_experts_per_token, num_tokens);

  all2all_dispatch_pack_send_buffers_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    x,
    indices,
    send_offsets,
    send_buf0,
    send_buf1,
    send_buf2,
    send_buf3,
    send_buf4,
    send_buf5,
    send_buf6,
    send_buf7,
    num_local_experts,
    num_tokens,
    num_experts_per_token,
    hidden_dim,
    buff_meta_offset,
    local_rank
  );
}

void __global__ all2all_dispatch_pack_send_buffers_fp16_kernel(
  const half* __restrict__ x,
  const int32_t* __restrict__ indices,
  uint32_t* __restrict__ send_offsets,
  half* __restrict__ send_buf0,
  half* __restrict__ send_buf1,
  half* __restrict__ send_buf2,
  half* __restrict__ send_buf3,
  half* __restrict__ send_buf4,
  half* __restrict__ send_buf5,
  half* __restrict__ send_buf6,
  half* __restrict__ send_buf7,
  int num_local_experts,
  int num_tokens,
  int num_experts_per_token,
  int hidden_dim,
  int buff_meta_offset,
  int rank
) {
  int expert_idx = blockIdx.x;
  int token_idx = blockIdx.y;

  x = &x[token_idx * hidden_dim];

  // determine where to write out
  // TODO: avoid division if num_local_experts is power of 2
  int32_t dst_expert = indices[token_idx * num_experts_per_token + expert_idx];
  int32_t dst_rank = dst_expert / num_local_experts;

  int32_t send_pos = -1;
  if (threadIdx.x == 0) {
    send_pos = (int32_t)atomicAdd((uint32_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos);

  half* send_buffs[8] = {send_buf0, send_buf1, send_buf2, send_buf3, send_buf4, send_buf5, send_buf6, send_buf7};
  half* send_buf = send_buffs[dst_rank];

  // write to send_buf
  half* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  for (int i = threadIdx.x; i < hidden_dim; i += THREADS_PER_BLOCK) {
    send_buf_ptr[i] = x[i];
  }

  // write to send_meta
  int32_t* send_meta = (int32_t*)(((uint8_t*)send_buf) + buff_meta_offset);
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  if (threadIdx.x == 0) {
    send_meta_ptr[0] = dst_expert;
    send_meta_ptr[1] = rank;
    send_meta_ptr[2] = token_idx;
    send_meta_ptr[3] = expert_idx;
  }
}

void all2all_dispatch_pack_send_buffers_fp16(
    hipStream_t stream,
    const half* __restrict__ x,
    const int32_t* __restrict__ indices,
    uint32_t* __restrict__ send_offsets,
    half* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int num_tokens,
    int num_experts_per_token,
    int hidden_dim,
    int buff_meta_offset,
    int local_size,
    int local_rank
) {
  // call kernel to do the packing, with a 2D grid of size (num_experts_per_token, num_tokens)
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(num_experts_per_token, num_tokens);

  all2all_dispatch_pack_send_buffers_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    x,
    indices,
    send_offsets,
    send_buf0,
    send_buf1,
    send_buf2,
    send_buf3,
    send_buf4,
    send_buf5,
    send_buf6,
    send_buf7,
    num_local_experts,
    num_tokens,
    num_experts_per_token,
    hidden_dim,
    buff_meta_offset,
    local_rank
  );
}

// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_dispatch_unpack_fp16_kernel(
    const half* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    half* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    // TODO recv_counts is on CPU so uncached
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset) {
  
  int lane_id = threadIdx.x % warpSize;

  int token_idx = blockIdx.x;

  if (token_idx >= recv_counts[0]) { // TODO: remove uncached access somehow
    return;
  }

  int global_expert_idx = recv_meta[token_idx * META_DIM + 0]; // expert_id
  int local_expert_idx = global_expert_idx - local_expert_offset;

  int local_num_expert_tokens = -1; // needs to be initialized to avoid compiler bug?

  //
  // reserve space in expert_x and expert_meta
  //

  // do a single atomic add per block and broadcast the result to the warp
  if (threadIdx.x == 0) {
    local_num_expert_tokens = atomicAdd(&expert_num_tokens[local_expert_idx], 1);
  }
  local_num_expert_tokens = __block_broadcast(local_num_expert_tokens);


  //
  // do the copies
  //

  // realign the pointers based on where to read/write
  recv_buf = &recv_buf[token_idx * hidden_dim];
  recv_meta = &recv_meta[token_idx * META_DIM];

  expert_x = &expert_x[(local_expert_idx * max_recv + local_num_expert_tokens) * hidden_dim];
  expert_meta = &expert_meta[(local_expert_idx * max_recv + local_num_expert_tokens) * META_DIM];

  // copy the token to expert_x
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    expert_x[d] = recv_buf[d];
  }

  // copy the meta to expert_meta
  if (threadIdx.x < META_DIM) {
    expert_meta[threadIdx.x] = recv_meta[threadIdx.x];
  }
}

void all2all_dispatch_unpack_fp16(
    hipStream_t stream,
    const half* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    half* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset,
    int max_total_recv) {

  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = max_total_recv;

  all2all_dispatch_unpack_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    recv_buf,
    recv_meta,
    expert_num_tokens,
    expert_x,
    expert_meta,
    recv_counts,
    hidden_dim,
    max_recv,
    local_expert_offset
  );
}

// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_dispatch_unpack_fp32_kernel(
    const float* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    float* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    // TODO recv_counts is on CPU so uncached
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset) {
 
  int lane_id = threadIdx.x % warpSize;

  int token_idx = blockIdx.x;

  if (token_idx >= recv_counts[0]) { // TODO: remove uncached access somehow
    return;
  }

  int global_expert_idx = recv_meta[token_idx * META_DIM + 0]; // expert_id
  int local_expert_idx = global_expert_idx - local_expert_offset;

  int local_num_expert_tokens = -1; // needs to be initialized to avoid compiler bug?

  //
  // reserve space in expert_x and expert_meta
  //

  // do a single atomic add per block and broadcast the result to the warp
  if (threadIdx.x == 0) {
    local_num_expert_tokens = atomicAdd(&expert_num_tokens[local_expert_idx], 1);
  }
  local_num_expert_tokens = __block_broadcast(local_num_expert_tokens);


  //
  // do the copies
  //

  // realign the pointers based on where to read/write
  recv_buf = &recv_buf[token_idx * hidden_dim];
  recv_meta = &recv_meta[token_idx * META_DIM];

  expert_x = &expert_x[(local_expert_idx * max_recv + local_num_expert_tokens) * hidden_dim];
  expert_meta = &expert_meta[(local_expert_idx * max_recv + local_num_expert_tokens) * META_DIM];

  // copy the token to expert_x
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    expert_x[d] = recv_buf[d];
  }

  // copy the meta to expert_meta
  if (threadIdx.x < META_DIM) {
    expert_meta[threadIdx.x] = recv_meta[threadIdx.x];
  }
}

void all2all_dispatch_unpack_fp32(
    hipStream_t stream,
    const float* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    float* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset,
    int max_total_recv) {

  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = max_total_recv;

  all2all_dispatch_unpack_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    recv_buf,
    recv_meta,
    expert_num_tokens,
    expert_x,
    expert_meta,
    recv_counts,
    hidden_dim,
    max_recv,
    local_expert_offset
  );
}

#define COMPUTE_PER_THREAD_FP32 4

void __global__ all2all_compute_fp32_kernel(
  const int32_t* __restrict__ expert_num_tokens,
  const float* __restrict__ expert_x,
  float* __restrict__ expert_y,
  int max_recv,
  int hidden_dim,
  int rank
) {

  int token_idx = blockIdx.x;
  int local_expert_idx = blockIdx.y;

  int num_local_tokens = expert_num_tokens[local_expert_idx];
  if (token_idx >= num_local_tokens) {
    return;
  }

  int element_idx = threadIdx.x * COMPUTE_PER_THREAD_FP32;

  // realign the pointers
  expert_x = &expert_x[(local_expert_idx * max_recv + token_idx) * hidden_dim + element_idx];
  expert_y = &expert_y[(local_expert_idx * max_recv + token_idx) * hidden_dim + element_idx];

  for (; element_idx < hidden_dim; element_idx += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP32) {
    if (element_idx + (COMPUTE_PER_THREAD_FP32 - 1) < hidden_dim) {
      // all elements are within range
      #pragma unroll
      for (int i = 0; i < COMPUTE_PER_THREAD_FP32; i++) {
        expert_y[i] = expert_x[i] * (1.0f + rank);
      }
    } else {
      // not all elements are within range, use a loop
      for (int i = 0; i < COMPUTE_PER_THREAD_FP32 && element_idx < hidden_dim; i++, element_idx++) {
        expert_y[i] = expert_x[i] * (1.0f + rank);
      }
    }
    // realign the pointers for the next iteration
    expert_x += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP32;
    expert_y += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP32;
  }
}

void all2all_compute_fp32(
  hipStream_t stream,
  const int32_t* __restrict__ expert_num_tokens,
  const float* __restrict__ expert_x,
  float* __restrict__ expert_y,
  int num_local_experts,
  int max_recv,
  int hidden_dim,
  int rank
) {
  // call the kernel to compute expert_y by processing COMPUTE_PER_THREADS_FP16 elements per thread
  // and THREADS_PER_BLOCK threads per block
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_compute_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_x,
    expert_y,
    max_recv,
    hidden_dim,
    rank
  );
}

#define COMPUTE_PER_THREAD_FP16 8

void __global__ all2all_compute_fp16_kernel(
  const int32_t* __restrict__ expert_num_tokens,
  const half* __restrict__ expert_x,
  half* __restrict__ expert_y,
  int max_recv,
  int hidden_dim,
  int rank
) {

  int token_idx = blockIdx.x;
  int local_expert_idx = blockIdx.y;

  int num_local_tokens = expert_num_tokens[local_expert_idx];
  if (token_idx >= num_local_tokens) {
    return;
  }

  int element_idx = threadIdx.x * COMPUTE_PER_THREAD_FP16;

  // realign the pointers
  expert_x = &expert_x[(local_expert_idx * max_recv + token_idx) * hidden_dim + element_idx];
  expert_y = &expert_y[(local_expert_idx * max_recv + token_idx) * hidden_dim + element_idx];

  for (; element_idx < hidden_dim; element_idx += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP16) {
    if (element_idx + (COMPUTE_PER_THREAD_FP16 - 1) < hidden_dim) {
      // all elements are within range
      #pragma unroll
      for (int i = 0; i < COMPUTE_PER_THREAD_FP16; i++) {
        expert_y[i] = __float2half_rn(__half2float(expert_x[i]) * (1.0f + rank));
      }
    } else {
      // not all elements are within range, use a loop
      for (int i = 0; i < COMPUTE_PER_THREAD_FP16 && element_idx < hidden_dim; i++, element_idx++) {
        expert_y[i] = __float2half_rn(__half2float(expert_x[i]) * (1.0f + rank));
      }
    }
    // realign the pointers for the next iteration
    expert_x += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP16;
    expert_y += THREADS_PER_BLOCK * COMPUTE_PER_THREAD_FP16;
  }
}

void all2all_compute_fp16(
  hipStream_t stream,
  const int32_t* __restrict__ expert_num_tokens,
  const half* __restrict__ expert_x,
  half* __restrict__ expert_y,
  int num_local_experts,
  int max_recv,
  int hidden_dim,
  int rank
) {
  // call the kernel to compute expert_y by processing COMPUTE_PER_THREADS_FP16 elements per thread
  // and THREADS_PER_BLOCK threads per block
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_compute_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_x,
    expert_y,
    max_recv,
    hidden_dim,
    rank
  );
}

//
// All2All combine kernels
//

// Compute send count kernels

// TODO: remove send_counts, use shared memory for it instead
void __global__ all2all_combine_compute_send_counts_kernel(
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    uint32_t* __restrict__ send_counts, // shape [world_size]
    uint32_t* __restrict__ send_offsets, // shape [world_size]
    // counters for the different ranks
    uint32_t* counters,
    // local counter to clear for next use
    uint32_t* next_counters,
    int num_local_experts,
    int max_recv,
    int local_size,
    int local_rank
) {
  if (local_rank == 0 && threadIdx.x < local_size) {
    // clear the counters for the next use
    next_counters[threadIdx.x] = 0;
  }

  // each thread handles one token expert
  for (int local_expert_idx = 0; local_expert_idx < num_local_experts; local_expert_idx++) {
    int num_local_tokens = expert_num_tokens[local_expert_idx];

    const int32_t* local_expert_meta = &expert_meta[local_expert_idx * max_recv * META_DIM + 1];

    // for each token of this expert, figure out which rank to send to
    for (int i = threadIdx.x; i < num_local_tokens; i += THREADS_PER_BLOCK) {
      int32_t dst_rank = local_expert_meta[i * META_DIM]; // dst_rank
      atomicAdd((uint32_t*)&send_counts[dst_rank], 1);
    }
  }

  __syncthreads();

  // do a single thread prefix sum to compute send offsets
  if (threadIdx.x < local_size) {
    int rank = threadIdx.x;
    uint32_t send_count = send_counts[rank];
    uint32_t send_offset = atomicAdd_system((uint32_t*) &counters[rank], send_count);
    send_offsets[rank] = send_offset;
  }
}

void all2all_combine_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    uint32_t* __restrict__ send_counts, // shape [world_size]
    uint32_t* __restrict__ send_offsets, // shape [world_size]
    // counters for the different ranks
    uint32_t* counters,
    // local counter to clear for next use
    uint32_t* next_counters,
    int num_local_experts,
    int max_recv,
    int local_size,
    int local_rank
) {

  // call kernel to compute send_counts using a single block to do the send offset reduction as well
  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks = 1;

  all2all_combine_compute_send_counts_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_meta,
    send_counts,
    send_offsets,
    counters,
    next_counters,
    num_local_experts,
    max_recv,
    local_size,
    local_rank
  );
}

// Send kernels


void __global__ all2all_combine_pack_send_buffers_fp16_kernel(
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const half* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int32_t* __restrict__ send_offsets, // shape [world_size]
    half* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int max_recv,
    int hidden_dim,
    int buff_meta_offset
) {
  int token_idx = blockIdx.x;
  int local_expert_idx = blockIdx.y;

  int num_local_tokens = expert_num_tokens[local_expert_idx];
  if (token_idx >= num_local_tokens) {
    return;
  }

  const int32_t* local_expert_meta = &expert_meta[(local_expert_idx * max_recv + token_idx) * META_DIM];
  const half* local_expert_y = &expert_y[(local_expert_idx * max_recv + token_idx) * hidden_dim];

  // determine where to write out
  int32_t dst_rank = local_expert_meta[1]; // dst_rank

  int32_t send_pos = -1;
  if (threadIdx.x == 0) {
    send_pos = (int32_t)atomicAdd((uint32_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos);

  half* send_buffs[8] = {send_buf0, send_buf1, send_buf2, send_buf3, send_buf4, send_buf5, send_buf6, send_buf7};
  half* send_buf = send_buffs[dst_rank];

  // write to send_buf
  half* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  for (int i = threadIdx.x; i < hidden_dim; i += THREADS_PER_BLOCK) {
    send_buf_ptr[i] = local_expert_y[i];
  }

  // write to send_meta
  int32_t* send_meta = (int32_t*)(((uint8_t*)send_buf) + buff_meta_offset);
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  for (int i = threadIdx.x; i < META_DIM; i += THREADS_PER_BLOCK) {
    send_meta_ptr[i] = local_expert_meta[i];
  }
}

void all2all_combine_pack_send_buffers_fp16(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const half* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int32_t* __restrict__ send_offsets, // shape [world_size]
    half* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    half* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int max_recv,
    int hidden_dim,
    int buff_meta_offset
) {
  // call kernel to do the packing, with a 2D grid of size (max_recv, num_local_experts)
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_combine_pack_send_buffers_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_meta,
    expert_y,
    send_offsets,
    send_buf0,
    send_buf1,
    send_buf2,
    send_buf3,
    send_buf4,
    send_buf5,
    send_buf6,
    send_buf7,
    max_recv,
    hidden_dim,
    buff_meta_offset
  );
}

void __global__ all2all_combine_pack_send_buffers_fp32_kernel(
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const float* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int32_t* __restrict__ send_offsets, // shape [world_size]
    float* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int max_recv,
    int hidden_dim,
    int buff_meta_offset
) {
  int token_idx = blockIdx.x;
  int local_expert_idx = blockIdx.y;

  int num_local_tokens = expert_num_tokens[local_expert_idx];
  if (token_idx >= num_local_tokens) {
    return;
  }

  const int32_t* local_expert_meta = &expert_meta[(local_expert_idx * max_recv + token_idx) * META_DIM];
  const float* local_expert_y = &expert_y[(local_expert_idx * max_recv + token_idx) * hidden_dim];

  // determine where to write out
  int32_t dst_rank = local_expert_meta[1]; // dst_rank

  int32_t send_pos = -1;
  if (threadIdx.x == 0) {
    send_pos = (int32_t)atomicAdd((uint32_t*)&send_offsets[dst_rank], 1);
  }
  send_pos = __block_broadcast(send_pos);

  float* send_buffs[8] = {send_buf0, send_buf1, send_buf2, send_buf3, send_buf4, send_buf5, send_buf6, send_buf7};
  float* send_buf = send_buffs[dst_rank];

  // write to send_buf
  float* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  for (int i = threadIdx.x; i < hidden_dim; i += THREADS_PER_BLOCK) {
    send_buf_ptr[i] = local_expert_y[i];
  }

  // write to send_meta
  int32_t* send_meta = (int32_t*)(((uint8_t*)send_buf) + buff_meta_offset);
  int32_t* send_meta_ptr = &send_meta[send_pos * META_DIM];
  for (int i = threadIdx.x; i < META_DIM; i += THREADS_PER_BLOCK) {
    send_meta_ptr[i] = local_expert_meta[i];
  }
}

void all2all_combine_pack_send_buffers_fp32(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const float* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    int32_t* __restrict__ send_offsets, // shape [world_size]
    float* __restrict__ send_buf0, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf1, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf2, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf3, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf4, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf5, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf6, // shape [total_send, hidden_dim]
    float* __restrict__ send_buf7, // shape [total_send, hidden_dim]
    int num_local_experts,
    int max_recv,
    int hidden_dim,
    int buff_meta_offset
) {
  // call kernel to do the packing, with a 2D grid of size (max_recv, num_local_experts)
  const int threads_per_block = THREADS_PER_BLOCK;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_combine_pack_send_buffers_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_meta,
    expert_y,
    send_offsets,
    send_buf0,
    send_buf1,
    send_buf2,
    send_buf3,
    send_buf4,
    send_buf5,
    send_buf6,
    send_buf7,
    max_recv,
    hidden_dim,
    buff_meta_offset
  );
}

// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_combine_scale_experts_fp32_kernel(
  const float* __restrict__ recv_buf,
  const int32_t* __restrict__ recv_meta,
  const float* __restrict__ weights,
  float* __restrict__ scaled_expert_outputs,
  int hidden_dim,
  int experts_per_token
) {
  int recv_token_idx = blockIdx.x;

  // realign input pointers
  recv_buf = &recv_buf[recv_token_idx * hidden_dim];
  recv_meta = &recv_meta[recv_token_idx * META_DIM];

  int src_token_idx = recv_meta[2];
  int src_k = recv_meta[3]; // src_k

  float w = weights[src_token_idx * experts_per_token + src_k];

  // realign output pointer
  scaled_expert_outputs = &scaled_expert_outputs[(src_token_idx * experts_per_token + src_k) * hidden_dim];

  // scale and write the expert output from the receive buffer
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    scaled_expert_outputs[d] = recv_buf[d] * w;
  }
}

void __global__ all2all_combine_write_back_fp32_kernel(
  const float* __restrict__ scaled_expert_outputs,
  float* __restrict__ output,
  int hidden_dim,
  int experts_per_token
) {
  int token_idx = blockIdx.x;

  // realign input pointer
  scaled_expert_outputs = &scaled_expert_outputs[token_idx * experts_per_token * hidden_dim];

  // realign output pointer
  output = &output[token_idx * hidden_dim];

  // combine the expert outputs into output
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    float sum = 0.0f;
    for (int k = 0; k < experts_per_token; k++) {
      sum += scaled_expert_outputs[k * hidden_dim + d];
    }
    output[d] = sum;
  }
}

void all2all_combine_unpack_fp32(
    hipStream_t stream,
    const float* __restrict__ recv_buf, // shape [total_recv, hidden_sim]
    const int32_t* __restrict__ recv_meta, // shape [total_recv, META_DIM]
    const float* __restrict__ weights, // shape [num_tokens, experts_per_token]
    float* __restrict__ scaled_expert_outputs, // shape [num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ output, // shape [max_num_tokens, hidden_dim]
    int hidden_dim,
    int total_recv,
    int num_tokens,
    int experts_per_token
) {
  // First kernel to scale expert outputs by weights and write to scaled_expert_outputs
  {
    const int threads_per_block = THREADS_PER_BLOCK;
    const int blocks = total_recv;

    all2all_combine_scale_experts_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
      recv_buf,
      recv_meta,
      weights,
      scaled_expert_outputs,
      hidden_dim,
      experts_per_token
    );
  }
  // Second kernel to combine scaled_expert_outputs into output
  {
    const int threads_per_block = THREADS_PER_BLOCK;
    const int blocks = num_tokens;

    all2all_combine_write_back_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
      scaled_expert_outputs,
      output,
      hidden_dim,
      experts_per_token
    );
  }
}


// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_combine_scale_experts_fp16_kernel(
  const half* __restrict__ recv_buf,
  const int32_t* __restrict__ recv_meta,
  const float* __restrict__ weights,
  float* __restrict__ scaled_expert_outputs,
  int hidden_dim,
  int experts_per_token
) {
  int recv_token_idx = blockIdx.x;

  // realign input pointers
  recv_buf = &recv_buf[recv_token_idx * hidden_dim];
  recv_meta = &recv_meta[recv_token_idx * META_DIM];

  // int dst_expert = recv_meta[0];
  // int dst_rank = recv_meta[1];
  int src_token_idx = recv_meta[2];
  int src_k = recv_meta[3]; // src_k

  float w = weights[src_token_idx * experts_per_token + src_k];

  // realign output pointer
  scaled_expert_outputs = &scaled_expert_outputs[(src_token_idx * experts_per_token + src_k) * hidden_dim];

  // scale and write the expert output from the receive buffer
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    scaled_expert_outputs[d] = __half2float(recv_buf[d]) * w;
  }
}

void __global__ all2all_combine_write_back_fp16_kernel(
  const float* __restrict__ scaled_expert_outputs,
  half* __restrict__ output,
  int hidden_dim,
  int experts_per_token
) {
  int token_idx = blockIdx.x;

  // realign input pointer
  scaled_expert_outputs = &scaled_expert_outputs[token_idx * experts_per_token * hidden_dim];

  // realign output pointer
  output = &output[token_idx * hidden_dim];

  // combine the expert outputs into output
  for (int d = threadIdx.x; d < hidden_dim; d += THREADS_PER_BLOCK) {
    float sum = 0.0f;
    for (int k = 0; k < experts_per_token; k++) {
      sum += scaled_expert_outputs[k * hidden_dim + d];
    }
    output[d] = __float2half(sum);
  }
}

void all2all_combine_unpack_fp16(
    hipStream_t stream,
    const half* __restrict__ recv_buf, // shape [total_recv, hidden_sim]
    const int32_t* __restrict__ recv_meta, // shape [total_recv, META_DIM]
    const float* __restrict__ weights, // shape [num_tokens, experts_per_token]
    float* __restrict__ scaled_expert_outputs, // shape [num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ output, // shape [max_num_tokens, hidden_dim]
    int hidden_dim,
    int total_recv,
    int num_tokens,
    int experts_per_token
) {
  // First kernel to scale expert outputs by weights and write to scaled_expert_outputs
  {
    const int threads_per_block = THREADS_PER_BLOCK;
    const int blocks = total_recv;

    all2all_combine_scale_experts_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
      recv_buf,
      recv_meta,
      weights,
      scaled_expert_outputs,
      hidden_dim,
      experts_per_token
    );
  }
  // Second kernel to combine scaled_expert_outputs into output
  {
    const int threads_per_block = THREADS_PER_BLOCK;
    const int blocks = num_tokens;

    all2all_combine_write_back_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
      scaled_expert_outputs,
      output,
      hidden_dim,
      experts_per_token
    );
  }
}
"""


class All2AllCommKernels:
    def __init__(self):
        from torch.utils.cpp_extension import load_inline

        # limit the architectures to avoid long compile time
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx908,gfx942"

        # measure the time to compile the kernels
        import time

        start_time = time.time()
        print("Loading All2All comm kernels...")

        self.comm_kernels = load_inline(
            name="all2all_comm_kernels",
            cpp_sources=[COMM_KERNELS_CPP_CODE],
            cuda_sources=[COMM_KERNELS_CUDA_CODE],
            functions=[
                "all2all_comm_init",
                "all2all_comm_dispatch",
                "all2all_compute",
                "all2all_comm_combine",
            ],
            with_cuda=True,
            no_implicit_headers=True,
        )
        end_time = time.time()
        print(
            f"All2All comm kernels loaded in {end_time - start_time:.2f} seconds.",
            flush=True,
        )


_global_comm_kernels = None


def get_global_comm_kernels():
    global _global_comm_kernels
    if _global_comm_kernels is None:
        _global_comm_kernels = All2AllCommKernels()
    return _global_comm_kernels.comm_kernels


class All2AllComm:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

        # measure the time to initialize the comms
        import time

        start_time = time.time()
        print(f"Initializing All2AllComm on rank {rank}...")

        self.comm_kernels = get_global_comm_kernels()

        if self.comm_kernels is not None:
            # use custom kernel
            self.comms = self.comm_kernels.all2all_comm_init(
                world_size,
                rank,
            )
        else:
            self.comms = None

        if self.comms is None:
            raise ValueError("All2AllComm initialization failed.")

        end_time = time.time()
        print(
            f"All2AllComm initialized in {end_time - start_time:.2f} seconds on rank {rank}.",
            flush=True,
        )

    def dispatch(
        self,
        dp_x: torch.Tensor,  # input shape (num_tokens, token_dim)
        indices: torch.Tensor,  # input shape (num_tokens, num_experts_per_token)
        num_local_experts: int,
        max_recv: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # returns (expert_num_tokens, expert_x, expert_meta)
        expert_num_tokens, expert_x, expert_meta = (
            self.comm_kernels.all2all_comm_dispatch(
                self.comms,
                dp_x,
                indices,
                num_local_experts,
                max_recv,
            )
        )

        return expert_num_tokens, expert_x, expert_meta

    def compute(self, expert_num_tokens: torch.Tensor, expert_x: torch.Tensor):
        return self.comm_kernels.all2all_compute(expert_num_tokens, expert_x, self.rank)

    def combine(
        self,
        weights: torch.Tensor,  # topk weight shape (num tokens, num_experts_per_token)
        expert_meta: torch.Tensor,  # input shape (num_local_experts, max_recv, META_DIM)
        expert_y: torch.Tensor,  # input shape (num_local_experts, max_recv, hidden_dim)
        expert_num_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> torch.Tensor:  # output shape (max num tokens, token dim)
        return self.comm_kernels.all2all_comm_combine(
            self.comms,
            weights,
            expert_meta,
            expert_y,
            expert_num_tokens,
            out_tokens,
        )


_global_all2all_comm = None


def get_global_all2all_comm(rank: int, world_size: int):
    global _global_all2all_comm
    if _global_all2all_comm is None:
        _global_all2all_comm = All2AllComm(rank, world_size)
    return _global_all2all_comm


# ---------------- All2All pytorch impl ----------------
class PyTorchAllToAll:
    META_DIM = 4  # global_exp, src_rank, src_token, src_k

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        # num experts per rank
        self.num_local_experts = cfg.num_experts // world_size
        # max recv tokens per rank
        self.max_recv = cfg.max_num_tokens * world_size

        self.comms = get_global_all2all_comm(rank, world_size)

    # ---------- dispatch ----------

    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        if self.comms is not None:
            return self.comms.dispatch(
                dp_x, indices, self.num_local_experts, self.max_recv
            )
        else:
            return self._dispatch_no_comms(dp_x, indices)

    def _dispatch_no_comms(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg

        # ---------1. get counts of send and recv for each rank -----------
        # 1.1 token nums to send to each rank
        send_counts = [0] * self.world_size
        # 1.2 token id to send to each rank
        token_map = [[] for _ in range(self.world_size)]
        # 1.3 token meta data, need update for combine
        meta_map = [[] for _ in range(self.world_size)]
        for t, expert_list in enumerate(indices.tolist()):
            for k, e in enumerate(expert_list):
                dst_rank = e // self.num_local_experts
                send_counts[dst_rank] += 1
                token_map[dst_rank].append(t)
                meta_map[dst_rank].extend(
                    [e, self.rank, t, k]
                )  # srcGobalExpert, srcRank, srcIndex, expert index

        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        # 1.3 token nums to recv from each rank
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        # ---------2. send and recv buffer, order by tokens on each rank ----------
        send_buf = torch.cat([dp_x[idx_list] for idx_list in token_map], dim=0)
        total_recv = int(recv_counts_t.sum().item())
        recv_buf = torch.empty(
            total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device
        )

        # 2.1 meta buf for send and recv
        send_meta = torch.tensor(
            [v for sub in meta_map for v in sub], dtype=torch.int32, device=device
        ).view(-1, self.META_DIM)
        recv_meta = torch.empty(
            total_recv, self.META_DIM, dtype=torch.int32, device=device
        )
        # ---------3. dispatch send_buf to recv_buf by recv and send counts--------------
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )

        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
        )
        recv_meta = recv_meta.view(-1, self.META_DIM)
        # ---------4. define output tensor of dispatch ------------
        # 4.1 num tokens per expert
        expert_num_tokens = torch.zeros(
            self.num_local_experts, dtype=torch.int32, device=device
        )
        # 4.2 token tensor on each expert
        expert_x = torch.empty(
            (self.num_local_experts, self.max_recv, cfg.hidden_dim),
            dtype=cfg.in_dtype,
            device=device,
        )
        expert_meta = torch.empty(
            (self.num_local_experts, self.max_recv, self.META_DIM),
            dtype=torch.int32,
            device=device,
        )
        # ---------5. dispatch send_meta to recv_meta by recv and send counts------
        # ---------6. write tokens to each expert on each rank ------
        # 6.1 fetch the local expert id of corresponding token i
        for i in range(total_recv):
            global_eid = int(recv_meta[i, 0].item())
            local_eid = global_eid % self.num_local_experts
            # output, store token buf and token meta and token nums of each expert
            expert_x[local_eid, expert_num_tokens[local_eid]] = recv_buf[i]
            expert_meta[local_eid, expert_num_tokens[local_eid]] = recv_meta[i]
            expert_num_tokens[local_eid] += 1
        # 6.2 after dispatch, token nums and token and meta of token on expert
        return expert_num_tokens, expert_x, expert_meta

    # ---------- combine ----------
    def combine(
        self,
        out_tokens: torch.Tensor,  # output, (max num tokens, token dim)
        weights: torch.Tensor,  # topk weight
        expert_meta: torch.Tensor,  # input
        expert_y: torch.Tensor,  # input, (num_local_experts, max_num_tokens * num_dp, token_dim)
        expert_num_tokens: torch.Tensor,
    ):
        if self.comms is not None:
            send_counts, send_offsets, out_tokens = self._combine_comms(
                out_tokens=out_tokens,
                weights=weights,
                expert_meta=expert_meta,
                expert_y=expert_y,
                expert_num_tokens=expert_num_tokens,
            )

            return out_tokens
        else:
            return self._combine_no_comms(
                out_tokens=out_tokens,
                weights=weights,
                expert_meta=expert_meta,
                expert_y=expert_y,
                expert_num_tokens=expert_num_tokens,
            )

    def _combine_no_comms(
        self,
        out_tokens: torch.Tensor,  # output, (max num tokens, token dim)
        weights: torch.Tensor,  # topk weight
        expert_meta: torch.Tensor,  # input
        expert_y: torch.Tensor,  # input, (num_local_experts, max_num_tokens * num_dp, token_dim)
        expert_num_tokens: torch.Tensor,
    ):  # input
        device = out_tokens.device
        cfg = self.cfg

        # 1. count send-back tokens in cur rank
        send_counts = [0] * self.world_size
        # 1.1 token that will send back
        y_map = [[] for _ in range(self.world_size)]
        # 1.2 meta info of each token that send back to its src rank
        meta_map = [[] for _ in range(self.world_size)]

        # 2. traverse each token of each local expert of each rank, fill into send_counts and y_map and meta_map
        for local_eid in range(self.num_local_experts):
            cnt = int(expert_num_tokens[local_eid].item())
            for j in range(cnt):
                # meta info token j of local eid
                meta = expert_meta[local_eid, j]
                dst_rank = int(meta[1].item())
                send_counts[dst_rank] += 1
                # token j and its meta that send back to dst rank/local eid
                y_map[dst_rank].append(expert_y[local_eid, j].unsqueeze(0))
                meta_map[dst_rank].extend(meta.tolist())
        # token nums that cur rank plan to send to other ranks
        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        # token nums that will recv from other ranks
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        # call all2all to send send counts and recv recv_counts_t at each rank by all2all
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        # 3.send buffers of each rank, that is, the tokens at its experts
        y_map_tensors = []
        for sub_list in y_map:
            if sub_list:
                y_map_tensors.append(torch.cat(sub_list, dim=0))
            else:
                y_map_tensors.append(
                    torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
                )
        send_buf = torch.cat(y_map_tensors, dim=0)
        # 4. flatten send meta by tokens
        send_meta = torch.tensor(
            [v for sub in meta_map for v in sub], dtype=torch.int32, device=device
        ).view(-1, self.META_DIM)
        # 5. total recv tokens of cur rank
        total_recv = int(recv_counts_t.sum().item())
        # 6. recv buffer of cur rank
        recv_buf = torch.empty(
            total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device
        )
        recv_meta = torch.empty(
            total_recv, self.META_DIM, dtype=torch.int32, device=device
        )
        # 7. call all2all to send and recv for each rank
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )
        # 8. call all2all to send meta and recv meta for each rank
        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
        )
        # 9. restore recv meta
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # 10. write back tokens from recv buf, per meta info, and do weighted sum
        for i in range(total_recv):
            src_token = int(recv_meta[i, 2].item())
            src_k = int(recv_meta[i, 3].item())
            src_rank = int(recv_meta[i, 1].item())
            w = weights[src_token, src_k].to(torch.float32)
            out_tokens[src_token] += recv_buf[i].to(torch.float32) * w

        return out_tokens

    def _dispatch_comms(self, dp_x: torch.Tensor, indices: torch.Tensor):
        # use custom kernel
        expert_num_tokens, expert_x, expert_meta = self.comms.dispatch(
            dp_x, indices, self.num_local_experts, self.max_recv
        )
        return expert_num_tokens, expert_x, expert_meta

    def compute(self, expert_num_tokens: torch.Tensor, expert_x: torch.Tensor):
        expert_y = self.comms.compute(expert_num_tokens, expert_x)
        return expert_y

    def _combine_comms(
        self,
        out_tokens: torch.Tensor,  # output, (max num tokens, token dim)
        weights: torch.Tensor,  # topk weight (num tokens, num_experts_per_token)
        expert_meta: torch.Tensor,  # input (num_local_experts, max_recv, META_DIM)
        expert_y: torch.Tensor,  # input, (num_local_experts, max_recv, hidden_dim)
        expert_num_tokens: torch.Tensor,
    ) -> torch.Tensor:

        output = self.comms.combine(
            weights=weights,
            expert_meta=expert_meta,
            expert_y=expert_y,
            expert_num_tokens=expert_num_tokens,
            out_tokens=out_tokens,
        )

        return output


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data

    ata = PyTorchAllToAll(cfg, rank, world_size)

    # Dispatch seems to work with kernels
    expert_num_tokens, expert_x, expert_meta = ata.dispatch(
        dp_x=rank_data.x, indices=rank_data.indices
    )

    if ata.comms is not None:
        # use the custom kernel
        # Compute seems to work with kernels
        expert_y = ata.compute(
            expert_num_tokens=expert_num_tokens,
            expert_x=expert_x,
        )
    else:
        expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    y = torch.zeros(
        cfg.max_num_tokens,
        cfg.hidden_dim,
        dtype=cfg.out_dtype,
        device=rank_data.x.device,
    )

    # Combine doesn't work with kernels
    y = ata.combine(
        out_tokens=y,
        weights=rank_data.weights,
        expert_meta=expert_meta,
        expert_y=expert_y,
        expert_num_tokens=expert_num_tokens,
    )

    return y[: rank_data.num_tokens]
