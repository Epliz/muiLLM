#include "comm_p2p.h"

#include "comm.h"
#include "comm_base.h"

#include <stdint.h>
#include <stdio.h>

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

#define MUILLM_COMM_INITIAL_BUFFER_CAPACITY (1024 * 1024) // 1MiB

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
    printf("(rank %d) Error while synchronizing device\n", local_rank);
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // make sure all CPUs have synchronized their GPUs
  if ((error =__local_socket_barrier(comm)) != MUILLM_COMM_SUCCESS) {
    return error;
  }

  // close all the previous mappings
  for (int d = 0; d < local_size; d++) {
    if (d == local_rank) continue;
    if (buffer_set->buffers[d] == nullptr) continue;

    if (hipIpcCloseMemHandle(buffer_set->buffers[d]) != hipSuccess) {
      // failed
      printf("(rank %d) Failed to close memory handle %d\n", local_rank, d);
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
      printf("(rank %d) Error while freeing recv_buffers\n", local_rank);
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }

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

  muillm_comm_error_t error;

  // we will import the memory mappings for that specific GPU
  if (hipSetDevice(local_rank) != hipSuccess) {
    printf("(rank %d) Failed to set device\n", local_rank);
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // free the memory mappings and so on
  if ((error =__free_buffer_set(comm, buffer_set)) != MUILLM_COMM_SUCCESS) {
    return error;
  }

  // allocate new buffers
  capacity = __next_power_of_2(capacity);

  void* ptr = nullptr;
  if (hipMalloc((void**)&ptr, capacity) != hipSuccess || ptr == nullptr) {
    std::cout<<"Allocation of buffer "<<local_rank<<" failed"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
  
  buffer_set->buffers[local_rank] = ptr;

  // get the memory pointers from other processes

  hipIpcMemHandle_t ipcHandle;
  if (hipIpcGetMemHandle(&ipcHandle, ptr) != hipSuccess) {
    printf("(rank %d) Failed to get mem handle\n", local_rank);
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
        printf("(rank %d) Failed to open memory handle %d\n", local_rank, d);
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
      if (recv_ptr == nullptr) {
        printf("(rank %d) Null pointer out of mem handle\n", local_rank);
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
      buffer_set->buffers[d] = recv_ptr;
    }
  }

  // we don't need this array anymore
  delete[] allMemHandles;

  // all buffer allocations suceeded
  buffer_set->capacity = capacity;

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t muillm_comm_p2p_get_buffer_set(
  muillm_comm_p2p_t* comm,
  size_t count,
  muillm_comm_datatype_t datatype,
  muillm_comm_p2p_buffer_set_t** buffer_set,
  hipStream_t stream
) {
  
  muillm_comm_error_t muillm_error;

  size_t capacity = __comm_size(datatype, count);
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

  if (hipSetDevice(local_rank) != hipSuccess) {
    printf("(rank %d) Failed to set device\n", local_rank);
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
    printf("(rank %d) Failed to set device\n", local_rank);
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  for (int d = 0; d < local_size; d++) {
    if (d == local_rank) continue;
    if (hipDeviceEnablePeerAccess(d, 0) != hipSuccess) {
      // TODO: return error
      printf("(rank %d) Failed to enable peer to peer with %d\n", local_rank, d);
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t check_p2p_is_working(
  muillm_comm_p2p_t* comm,
  hipStream_t stream) {

  int local_rank = comm->local_rank;

  // if an all-reduce works, then p2p should be working
  return muillm_comm_p2p_all_reduce_sum(
    comm,
    comm->first_buffers->buffers[local_rank],
    comm->first_buffers->buffers[local_rank],
    1,
    MUILLM_COMM_FP16,
    stream
  );
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

  printf("(rank %d local_rank %d) Initializing comm for world_size %d local_size %d ...\n", rank, local_rank, world_size, local_size);

  muillm_comm_error_t muillm_error;

  muillm_comm_method_t transfer_method = MUILLM_COMM_METHOD_P2P_TRANSFER;

  // create the comm object
  muillm_comm_p2p_t* comm = nullptr;
  comm = new muillm_comm_p2p_t;
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
    printf("(rank %d) Failed to set device\n", local_rank);
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // check that signal memory is supported
  int signals_supported;
  if (hipDeviceGetAttribute(&signals_supported, hipDeviceAttributeCanUseStreamWaitValue, 0) != hipSuccess) {
    std::cout<<"Error getting the the property"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  if (!signals_supported) {
    std::cout<<"Signal memory is not supported"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // setup p2p 
  __init_p2p_recv(comm);

  // allocate cache flush event
  if (hipEventCreateWithFlags(&comm->acquire_event, hipEventDisableTiming | hipEventReleaseToSystem) != hipSuccess) {
    std::cout<<"event creation failed\n"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // allocate signal memory
  __allocate_locked_shared_cpu_mem(
    comm,
    sizeof(uint64_t), // alloc 8 bytese even though we use only 4
    (void**) &comm->signal_host,
    (void**) &comm->signal
  );

  // initialize to 0
  if (hipMemset(comm->signal, 0, sizeof(uint64_t)) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  if (comm->signal_host == nullptr || comm->signal == nullptr) {
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
    printf("(rank %d) Failed to set device\n", local_rank);
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // check that the p2p transfers actually work
  if ((muillm_error = check_p2p_is_working(comm, stream)) != MUILLM_COMM_SUCCESS) {
    printf("(rank %d) p2p check failed\n", local_rank);
    // free the buffers
    __free_buffer_set(comm, comm->first_buffers);
    __free_buffer_set(comm, comm->second_buffers);

    // free the signal memory
    __deallocate_locked_shared_cpu_mem(comm, comm->signal_host);

    // set the device before exiting
    if (hipSetDevice(local_rank) != hipSuccess) {
      printf("(rank %d) Failed to set device\n", local_rank);
      return MUILLM_COMM_UNKNOWN_ERROR;
    }

    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  if (hipDeviceSynchronize() != hipSuccess) {
    printf("(rank %d) Error while synchronizing device\n", local_rank);
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // make sure every GPU has opened the memory before returning
  if ((muillm_error =__local_socket_barrier(comm)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  // return the comm object
  printf("(rank %d) Created p2p comm %p\n", local_rank, comm);
  
  *comm_ptr = comm;

  return MUILLM_COMM_SUCCESS;
}

__global__ void __muillm_inc_value_p2p_kernel(
  uint32_t* signal
) {
  if (threadIdx.x == 0) {
    atomicAdd_system(signal, 1);
    __threadfence_system();
  }
}

static muillm_comm_error_t __mui_stream_inc_value(hipStream_t stream, uint32_t* signal, int rank) {
  __muillm_inc_value_p2p_kernel<<<1, 1, 0, stream>>>(signal);
  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __mui_gpu_barrier(muillm_comm_p2p_t* comm, hipStream_t stream) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  if (comm->signal != nullptr) {
    comm->signal_seq_no += local_size;
    uint64_t seq_no = comm->signal_seq_no;

    // GPU barrier: all GPUs wait on each other
    // record an event to flush caches
    if (hipEventRecord(comm->acquire_event, stream) != hipSuccess) {
      std::cout<<"Failed to record event "<<local_rank<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }

    // write the values
    if ((muillm_error = __mui_stream_inc_value(stream, comm->signal, local_rank)) != MUILLM_COMM_SUCCESS) {
      std::cout<<"inc value failed"<<std::endl;
      return muillm_error;
    }

    // wait for the other ranks
    // we need the comparison to be >= as one GPU might already increment the value before all the other GPUs
    // have seen the previous one
    if ((hip_error = hipStreamWaitValue32(stream, comm->signal, seq_no, hipStreamWaitValueGte, -1)) != hipSuccess) {
      std::cout<<"Failed to wait for value"<<std::endl;
      std::cout<<"error: "<<hipGetErrorName(hip_error)<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}

#define THREADS_PER_BLOCK 256


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

static muillm_comm_error_t __muillm_gpu_copy(void* dst, const void* src, size_t count, hipStream_t stream) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(count, BYTES_PER_BLOCK);

  // a copy kernel is faster than a hipMemcpyAsync
  __muillm_copy_p2p_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint8_t*) src,
    (uint8_t*) dst,
    count
  );

  if (hipPeekAtLastError() != hipSuccess) {
    printf("copy p2p failed\n");
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}


// TP2 kernels

__global__ void __all_reduce_fp16_tp2_p2p_kernel(
  const half* x1,
  const half* x2,
  half* y,
  unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    half res = __hadd(x1[i], x2[i]);
    y[i] = res;
  }
}

__global__ void __all_reduce_fp32_tp2_p2p_kernel(
  const float* x1,
  const float* x2,
  float* y,
  unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float res = x1[i] + x2[i];
    y[i] = res;
  }
}

// TP4 kernels

__global__ void __all_reduce_fp16_tp4_p2p_kernel(
  const half* x1,
  const half* x2,
  const half* x3,
  const half* x4,
  half* y,
  unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    half res = __hadd(__hadd(x1[i], x2[i]), __hadd(x3[i], x4[i]));
    y[i] = res;
  }
}

__global__ void __all_reduce_fp32_tp4_p2p_kernel(
  const float* x1,
  const float* x2,
  const float* x3,
  const float* x4,
  float* y,
  unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float res = x1[i] + x2[i] + x3[i] + x4[i];
    y[i] = res;
  }
}

// TP8 kernels

__global__ void __all_reduce_fp16_tp8_p2p_kernel(
  const half* x1,
  const half* x2,
  const half* x3,
  const half* x4,
  const half* x5,
  const half* x6,
  const half* x7,
  const half* x8,
  half* y,
  unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    half x1x2 = __hadd(x1[i], x2[i]);
    half x3x4 = __hadd(x3[i], x4[i]);
    half x1x4 = __hadd(x1x2, x3x4);
    half x5x6 = __hadd(x5[i], x6[i]);
    half x7x8 = __hadd(x7[i], x8[i]);
    half x5x8 = __hadd(x5x6, x7x8);
    half res = __hadd(x1x4, x5x8);
    y[i] = res;
  }
}

__global__ void __all_reduce_fp32_tp8_p2p_kernel(
  const float* x1,
  const float* x2,
  const float* x3,
  const float* x4,
  const float* x5,
  const float* x6,
  const float* x7,
  const float* x8,
  float* y,
  unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float x1x2 = (x1[i] + x2[i]);
    float x3x4 = (x3[i] + x4[i]);
    float x1x4 = x1x2 + x3x4;
    float x5x6 = (x5[i] + x6[i]);
    float x7x8 = (x7[i] + x8[i]);
    float x5x8 = x5x6 + x7x8;
    float res = x1x4 + x5x8;
    y[i] = res;
  }
}


muillm_comm_error_t muillm_comm_p2p_placed_all_reduce_sum(
  muillm_comm_p2p_t* comm,
  const void** src_ptrs,
  void* dst_ptr,
  size_t count,
  muillm_comm_datatype_t datatype,
  hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  // ensure all GPUs have copied into the reduction buffers
  if ((muillm_error = __mui_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"p2p reduction barrier failed"<<std::endl;
    return muillm_error;
  }

  // do the reduction
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(count, THREADS_PER_BLOCK);

  if (datatype == MUILLM_COMM_FP16) {
    if (local_size == 8) {
      __all_reduce_fp16_tp8_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) src_ptrs[0],
        (const half*) src_ptrs[1],
        (const half*) src_ptrs[2],
        (const half*) src_ptrs[3],
        (const half*) src_ptrs[4],
        (const half*) src_ptrs[5],
        (const half*) src_ptrs[6],
        (const half*) src_ptrs[7],
        (half*) dst_ptr,
        count
      );
    } else if (local_size == 4) {
      __all_reduce_fp16_tp4_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) src_ptrs[0],
        (const half*) src_ptrs[1],
        (const half*) src_ptrs[2],
        (const half*) src_ptrs[3],
        (half*) dst_ptr,
        count
      );
    } else if (local_size == 2) {
      __all_reduce_fp16_tp2_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) src_ptrs[0],
        (const half*) src_ptrs[1],
        (half*) dst_ptr,
        count
      );
    } else {
      std::cout<<"reduction unsupported tp size"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else if (datatype == MUILLM_COMM_FP32) {
    if (local_size == 8) {
      __all_reduce_fp32_tp8_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) src_ptrs[0],
        (const float*) src_ptrs[1],
        (const float*) src_ptrs[2],
        (const float*) src_ptrs[3],
        (const float*) src_ptrs[4],
        (const float*) src_ptrs[5],
        (const float*) src_ptrs[6],
        (const float*) src_ptrs[7],
        (float*) dst_ptr,
        count
      );
    } else if (local_size == 4) {
      __all_reduce_fp32_tp4_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) src_ptrs[0],
        (const float*) src_ptrs[1],
        (const float*) src_ptrs[2],
        (const float*) src_ptrs[3],
        (float*) dst_ptr,
        count
      );
    } else if (local_size == 2) {
      __all_reduce_fp32_tp2_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) src_ptrs[0],
        (const float*) src_ptrs[1],
        (float*) dst_ptr,
        count
      );
    } else {
      std::cout<<"reduction unsupported tp size"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else {
    std::cout<<"reduction unsupported dtype"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  if (hipPeekAtLastError() != hipSuccess) {
    printf("p2p reduce failed\n");
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t muillm_comm_p2p_all_reduce_sum(
  muillm_comm_p2p_t* comm,
  const void* src_ptr,
  void* dst_ptr,
  size_t count,
  muillm_comm_datatype_t datatype,
  hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  // get reduction buffer set
  muillm_comm_p2p_buffer_set_t* buffer_set = nullptr;

  if ((muillm_error = muillm_comm_p2p_get_buffer_set(comm, count, datatype, &buffer_set, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"Reduction failed when ensuring capacity"<<std::endl;
    return muillm_error;
  }

  // TODO: avoid this copy if src_ptrs are the buffer set
  // (need to pay attention to the fact that muillm_comm_p2p_get_buffer_set flips the buffers,
  // so if we use it to place into buffers for parallel linear, we need to check somehow properly when calling here)

  // copy into reduction buffers
  size_t byte_count = __comm_size(datatype, count);
  if ((muillm_error = __muillm_gpu_copy(buffer_set->buffers[local_rank], src_ptr, byte_count, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"p2p copy failed"<<std::endl;
    return muillm_error;
  }

  return muillm_comm_p2p_placed_all_reduce_sum(
    comm,
    (const void**) buffer_set->buffers,
    dst_ptr,
    count,
    datatype,
    stream
  );
}

muillm_comm_error_t muillm_comm_p2p_broadcast(
  muillm_comm_p2p_t* comm,
  int src,
  const void* src_ptr,
  void* dst_ptr,
  size_t count,
  muillm_comm_datatype_t datatype,
  hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  // get reduction buffer set
  muillm_comm_p2p_buffer_set_t* buffer_set = nullptr;

  if ((muillm_error = muillm_comm_p2p_get_buffer_set(comm, count, datatype, &buffer_set, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"Reduction failed when ensuring capacity"<<std::endl;
    return muillm_error;
  }

  size_t byte_count = __comm_size(datatype, count);

  // copy into reduction buffer if needed
  if (local_rank == src) {
    if ((muillm_error = __muillm_gpu_copy(buffer_set->buffers[local_rank], src_ptr, byte_count, stream)) != MUILLM_COMM_SUCCESS) {
      std::cout<<"p2p forward copy failed"<<std::endl;
      return muillm_error;
    }
  }

  // ensure all GPUs have arrived
  if ((muillm_error = __mui_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"p2p broadcast barrier failed"<<std::endl;
    return muillm_error;
  }

  // do the broadcast
  if ((muillm_error = __muillm_gpu_copy(dst_ptr, buffer_set->buffers[src], byte_count, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"p2p back copy failed"<<std::endl;
    return muillm_error;
  }

  return MUILLM_COMM_SUCCESS;
}