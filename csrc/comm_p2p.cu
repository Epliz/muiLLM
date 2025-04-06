#include "comm_p2p.h"

#include "comm.h"
#include "comm_base.h"
#include "gpu_info.h"

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


struct __align__(8) half4 {
  half x;
  half y;
  half z;
  half w;
};

struct __align__(8) half8 {
  half x;
  half y;
  half z;
  half w;
  half a;
  half b;
  half c;
  half d;
};

struct __align__(8) float8 {
  float x;
  float y;
  float z;
  float w;
  float a;
  float b;
  float c;
  float d;
};

static inline __device__ half8 __hadd(half8 a, half8 b) {
  half8 res;
  res.x = __hadd(a.x, b.x);
  res.y = __hadd(a.y, b.y);
  res.z = __hadd(a.z, b.z);
  res.w = __hadd(a.w, b.w);
  res.a = __hadd(a.a, b.a);
  res.b = __hadd(a.b, b.b);
  res.c = __hadd(a.c, b.c);
  res.d = __hadd(a.d, b.d);
  return res;
}

static inline  __device__ half4 __hadd(half4 a, half4 b) {
  half4 res;
  res.x = __hadd(a.x, b.x);
  res.y = __hadd(a.y, b.y);
  res.z = __hadd(a.z, b.z);
  res.w = __hadd(a.w, b.w);
  return res;
}

static inline __device__ half2 __hadd(half2 a, half2 b) {
  half2 res;
  res.x = __hadd(a.x, b.x);
  res.y = __hadd(a.y, b.y);
  return res;
}

__device__ half2 load_nontemporal_half2(const half* p) {
  float _v = __builtin_nontemporal_load((const float*)p);
  return *((half2*)&_v);
}

__device__ half4 load_nontemporal_half4(const half* p) {
  float _v0 = __builtin_nontemporal_load(((const float*)p));
  float _v1 = __builtin_nontemporal_load(((const float*)p) + 1);

  half2 _hv0 = *((half2*)&_v0);
  half2 _hv1 = *((half2*)&_v1);

  half4 v;
  v.x = _hv0.x;
  v.y = _hv0.y;
  v.z = _hv1.x;
  v.w = _hv1.y;

  return v;
}

__device__ half8 load_nontemporal_half8(const half* p) {
  float _v0 = __builtin_nontemporal_load(((const float*)p));
  float _v1 = __builtin_nontemporal_load(((const float*)p) + 1);
  float _v2 = __builtin_nontemporal_load(((const float*)p) + 2);
  float _v3 = __builtin_nontemporal_load(((const float*)p) + 3);

  half2 _hv0 = *((half2*)&_v0);
  half2 _hv1 = *((half2*)&_v1);
  half2 _hv2 = *((half2*)&_v2);
  half2 _hv3 = *((half2*)&_v3);

  half8 v;
  v.x = _hv0.x;
  v.y = _hv0.y;
  v.z = _hv1.x;
  v.w = _hv1.y;
  v.a = _hv2.x;
  v.b = _hv2.y;
  v.c = _hv3.x;
  v.d = _hv3.y;

  return v;
}

__device__ float2 load_nontemporal_float2(const float* p) {
  float _v0 = __builtin_nontemporal_load(((const float*)p));
  float _v1 = __builtin_nontemporal_load(((const float*)p) + 1);

  float2 v;
  v.x = _v0;
  v.y = _v1;

  return v;
}

__device__ float4 load_nontemporal_float4(const float* p) {
  float _v0 = __builtin_nontemporal_load(((const float*)p));
  float _v1 = __builtin_nontemporal_load(((const float*)p) + 1);
  float _v2 = __builtin_nontemporal_load(((const float*)p) + 2);
  float _v3 = __builtin_nontemporal_load(((const float*)p) + 3);

  float4 v;
  v.x = _v0;
  v.y = _v1;
  v.z = _v2;
  v.w = _v3;

  return v;
}

static muillm_comm_error_t __mui_gpu_barrier(
  muillm_comm_p2p_t* comm,
  hipStream_t stream
);

#define MUILLM_COMM_INITIAL_BUFFER_CAPACITY (1024 * 1024) // 1MiB

static muillm_comm_error_t __allocate_shared_gpu_memory(
  muillm_comm_p2p_t* comm,
  size_t capacity,
  void** ptrs,
  bool zero_memory = false,
  bool uncached_memory = false
) {

  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  muillm_comm_error_t error;

  int allocation_flags = 0;
  if (uncached_memory) {
    std::cout<<"Allocating uncached memory"<<std::endl;
    allocation_flags = hipDeviceMallocUncached;
  }

  void* ptr = nullptr;
  if (hipExtMallocWithFlags((void**)&ptr, capacity, allocation_flags) != hipSuccess || ptr == nullptr) {
    std::cout<<"Allocation of buffer "<<local_rank<<" failed"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  if (zero_memory) {
    if (hipMemset(ptr, 0, capacity) != hipSuccess) {
      std::cout<<"Zeroing of buffer "<<local_rank<<" failed"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }
  
  if (hipDeviceSynchronize() != hipSuccess) {
    std::cerr << "Failed to synchronize device " << local_rank << std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  ptrs[local_rank] = ptr;

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
      void* ext_ptr = nullptr;
      // import the memory mapping on the current GPU
      if (hipIpcOpenMemHandle(&ext_ptr, allMemHandles[d], hipIpcMemLazyEnablePeerAccess) != hipSuccess) {
        // failed
        printf("(rank %d) Failed to open memory handle %d\n", local_rank, d);
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
      if (ext_ptr == nullptr) {
        printf("(rank %d) Null pointer out of mem handle\n", local_rank);
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
      ptrs[d] = ext_ptr;
    }
  }

  // we don't need this array anymore
  delete[] allMemHandles;

  return MUILLM_COMM_SUCCESS;
}


static muillm_comm_error_t __free_shared_gpu_memory(
  muillm_comm_p2p_t* comm,
  void** ptrs
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
    if (ptrs[d] == nullptr) continue;

    if (hipIpcCloseMemHandle(ptrs[d]) != hipSuccess) {
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
  if (ptrs[local_rank] != nullptr) {
    if (hipFree(ptrs[local_rank]) != hipSuccess) {
      printf("(rank %d) Error while freeing recv_buffers\n", local_rank);
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }

  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __free_buffer_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_buffer_set_t* buffer_set
) {
  return __free_shared_gpu_memory(comm, buffer_set->buffers);
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

  // use the next power of two to avoid frequent re-allocations
  capacity = __next_power_of_2(capacity);
  // also align with the number of GPUs
  // (so that reduction methods like two or three steps reductions are guaranteed to be able to hold
  // all packed chunks in one temp buffer set)
  capacity = ALIGN_UP(capacity, local_size);

  // allocate shared gpu memory
  if ((error = __allocate_shared_gpu_memory(comm, capacity, buffer_set->buffers, /*zero_memory*/ false)) != MUILLM_COMM_SUCCESS) {
    return error;
  }

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
  muillm_engine_t* engine,
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
  muillm_comm_p2p_t* comm = new muillm_comm_p2p_t;
  comm->transfer_method = transfer_method;

  comm->world_size = world_size;
  comm->local_size = local_size;
  comm->rank = rank;
  comm->local_rank = local_rank;

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

  // get the gpu info from the engine
  comm->gpu_info = engine->gpu_infos[local_rank];

  // setup p2p 
  __init_p2p_recv(comm);

  // by default, do not skip the cache flush
  // but MI300 and successors don't need it apparently
  comm->cant_skip_cache_flush_event = comm->gpu_info->arch < MUILLM_GPU_ARCH_MI300;

  // allocate cache flush event
  if (hipEventCreateWithFlags(&comm->cache_flush_event, hipEventDisableTiming | hipEventReleaseToSystem) != hipSuccess) {
    std::cout<<"event creation failed\n"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // allocate signal memory
  // needs to be uncached so that the spinning on local GPU memory works
  if ((muillm_error = __allocate_shared_gpu_memory(
    comm,
    sizeof(int),
    (void**)comm->signals,
    /*zero_memory*/ true,
    /*uncached_memory*/ true)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
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
    __free_shared_gpu_memory(comm, (void**)comm->signals);

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

__global__ void __muillm_inc_wait_value_p2p_tp2_kernel(
  int* remote_signal0,
  int* local_signal,
  int seq_no
) {
  if (threadIdx.x == 0) {
    // increment the value
    atomicAdd_system(remote_signal0, 1);

    __threadfence_system();

    // wait for all the other ranks to increment our value
    // we spin on it, it won't generate PCIe traffic as it is local memory
    // we need the comparison to be >= as one GPU might already increment the value before all the other GPUs
    // have seen the previous one
    while (atomicAdd_system(local_signal, 0) < seq_no) {
      __builtin_amdgcn_s_sleep(2);
      __threadfence_system();
    }
  }
}

__global__ void __muillm_inc_wait_value_p2p_tp4_kernel(
  int* remote_signal0,
  int* remote_signal1,
  int* remote_signal2,
  int* local_signal,
  int seq_no
) {
  if (threadIdx.x == 0) {
    // increment the value
    atomicAdd_system(remote_signal0, 1);
    atomicAdd_system(remote_signal1, 1);
    atomicAdd_system(remote_signal2, 1);

    __threadfence_system();

    // wait for all the other ranks to increment our value
    // we spin on it, it won't generate PCIe traffic as it is local memory
    // we need the comparison to be >= as one GPU might already increment the value before all the other GPUs
    // have seen the previous one
    while (atomicAdd_system(local_signal, 0) < seq_no) {
      __builtin_amdgcn_s_sleep(2);
      __threadfence_system();
    }
  }
}

__global__ void __muillm_inc_wait_value_p2p_tp8_kernel(
  int* remote_signal0,
  int* remote_signal1,
  int* remote_signal2,
  int* remote_signal3,
  int* remote_signal4,
  int* remote_signal5,
  int* remote_signal6,
  int* local_signal,
  int seq_no
) {
  if (threadIdx.x == 0) {
    // increment the value
    atomicAdd_system(remote_signal0, 1);
    atomicAdd_system(remote_signal1, 1);
    atomicAdd_system(remote_signal2, 1);
    atomicAdd_system(remote_signal3, 1);
    atomicAdd_system(remote_signal4, 1);
    atomicAdd_system(remote_signal5, 1);
    atomicAdd_system(remote_signal6, 1);

    __threadfence_system();

    // wait for all the other ranks to increment our value
    // we spin on it, it won't generate PCIe traffic as it is local memory
    // we need the comparison to be >= as one GPU might already increment the value before all the other GPUs
    // have seen the previous one
    while (atomicAdd_system(local_signal, 0) < seq_no) {
      __builtin_amdgcn_s_sleep(2);
      __threadfence_system();
    }
  }
}

static muillm_comm_error_t __mui_gpu_barrier(muillm_comm_p2p_t* comm, hipStream_t stream) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  // all other GPUs will increase the counter but ourselves
  // so we need to wait for an increment of K-1
  comm->signal_seq_no += (local_size - 1);
  uint64_t seq_no = comm->signal_seq_no;

  // GPU barrier: all GPUs wait on each other
  if (comm->cant_skip_cache_flush_event) {
    // on MI100, we get a crash if not putting this event here
    // record an event to flush caches
    if (hipEventRecord(comm->cache_flush_event, stream) != hipSuccess) {
      std::cout<<"Failed to record event "<<local_rank<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }

  // pack the remote counter pointers
  int* remote_signals[MUILLM_COMM_MAX_GPUS];
  int p = 0;
  for (int d = 0; d < local_size; d++) {
    if (d == local_rank) continue;
    remote_signals[p] = comm->signals[d];
    p++;
  }

  // write the values
  if (local_size == 8) {
    __muillm_inc_wait_value_p2p_tp8_kernel<<<1, 1, 0, stream>>>(
      remote_signals[0],
      remote_signals[1],
      remote_signals[2],
      remote_signals[3],
      remote_signals[4],
      remote_signals[5],
      remote_signals[6],
      comm->signals[local_rank],
      seq_no
    );
  } else if (local_size == 4) {
    __muillm_inc_wait_value_p2p_tp4_kernel<<<1, 1, 0, stream>>>(
      remote_signals[0],
      remote_signals[1],
      remote_signals[2],
      comm->signals[local_rank],
      seq_no
    );

  } else if (local_size == 2) {
    __muillm_inc_wait_value_p2p_tp2_kernel<<<1, 1, 0, stream>>>(
      remote_signals[0],
      comm->signals[local_rank],
      seq_no
    );
  } else {
    return MUILLM_COMM_UNSUPPORTED_SIZE;
  }

  return MUILLM_COMM_SUCCESS;
}

#define THREADS_PER_BLOCK 256


// each threads can copy 16 bytes
#define COPY_BYTES_PER_THREAD 16
#define COPY_BYTES_PER_BLOCK (THREADS_PER_BLOCK * COPY_BYTES_PER_THREAD)

#define COPY_FP16S_PER_BLOCK (COPY_BYTES_PER_BLOCK / 2)
#define COPY_FP32S_PER_BLOCK (COPY_BYTES_PER_BLOCK / 4)

typedef struct uint32x4{
uint32_t x, y, z, w;
} uint32x4_t;

__device__ uint32x4_t load_nontemporal_uint4(const uint32_t* p) {
  uint32_t _v0 = __builtin_nontemporal_load(((const uint32_t*)p));
  uint32_t _v1 = __builtin_nontemporal_load(((const uint32_t*)p) + 1);
  uint32_t _v2 = __builtin_nontemporal_load(((const uint32_t*)p) + 2);
  uint32_t _v3 = __builtin_nontemporal_load(((const uint32_t*)p) + 3);

  uint32x4_t v;
  v.x = _v0;
  v.y = _v1;
  v.z = _v2;
  v.w = _v3;

  return v;
}

// copy one chunk of data from src to dst
// (potentially the pointers have been adjusted by the caller)


// copy one chunk of data from src to dst
// (potentially the pointers have been adjusted by the caller)
__device__ void __muillm_do_copy_p2p(
  const uint8_t* __restrict__ src_ptr,
  uint8_t* __restrict__ dst_ptr,
  unsigned N,
  unsigned block_chunk_size = COPY_BYTES_PER_BLOCK
) {
  unsigned blockStart = blockIdx.x * block_chunk_size;
  unsigned blockEnd = std::min(blockStart + block_chunk_size, N);

  unsigned i = blockStart + (threadIdx.x * COPY_BYTES_PER_THREAD);

  while (i + (16 * COPY_BYTES_PER_BLOCK) <= blockEnd) {
    const uint32_t* src_u32_ptr0 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr0 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr1 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr1 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr2 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr2 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr3 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr3 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr4 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr4 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr5 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr5 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr6 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr6 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr7 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr7 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr8 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr8 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr9 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr9 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr10 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr10 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr11 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr11 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr12 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr12 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr13 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr13 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr14 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr14 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr15 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr15 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    *dst_x16_ptr0 = load_nontemporal_uint4(src_u32_ptr0);
    *dst_x16_ptr1 = load_nontemporal_uint4(src_u32_ptr1);
    *dst_x16_ptr2 = load_nontemporal_uint4(src_u32_ptr2);
    *dst_x16_ptr3 = load_nontemporal_uint4(src_u32_ptr3);
    *dst_x16_ptr4 = load_nontemporal_uint4(src_u32_ptr4);
    *dst_x16_ptr5 = load_nontemporal_uint4(src_u32_ptr5);
    *dst_x16_ptr6 = load_nontemporal_uint4(src_u32_ptr6);
    *dst_x16_ptr7 = load_nontemporal_uint4(src_u32_ptr7);
    *dst_x16_ptr8 = load_nontemporal_uint4(src_u32_ptr8);
    *dst_x16_ptr9 = load_nontemporal_uint4(src_u32_ptr9);
    *dst_x16_ptr10 = load_nontemporal_uint4(src_u32_ptr10);
    *dst_x16_ptr11 = load_nontemporal_uint4(src_u32_ptr11);
    *dst_x16_ptr12 = load_nontemporal_uint4(src_u32_ptr12);
    *dst_x16_ptr13 = load_nontemporal_uint4(src_u32_ptr13);
    *dst_x16_ptr14 = load_nontemporal_uint4(src_u32_ptr14);
    *dst_x16_ptr15 = load_nontemporal_uint4(src_u32_ptr15);
  }

  if (i + (8 * COPY_BYTES_PER_BLOCK) <= blockEnd) {
    const uint32_t* src_u32_ptr0 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr0 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr1 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr1 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr2 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr2 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr3 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr3 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr4 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr4 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr5 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr5 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr6 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr6 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr7 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr7 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    *dst_x16_ptr0 = load_nontemporal_uint4(src_u32_ptr0);
    *dst_x16_ptr1 = load_nontemporal_uint4(src_u32_ptr1);
    *dst_x16_ptr2 = load_nontemporal_uint4(src_u32_ptr2);
    *dst_x16_ptr3 = load_nontemporal_uint4(src_u32_ptr3);
    *dst_x16_ptr4 = load_nontemporal_uint4(src_u32_ptr4);
    *dst_x16_ptr5 = load_nontemporal_uint4(src_u32_ptr5);
    *dst_x16_ptr6 = load_nontemporal_uint4(src_u32_ptr6);
    *dst_x16_ptr7 = load_nontemporal_uint4(src_u32_ptr7);
  }

  if (i + (4 * COPY_BYTES_PER_BLOCK) <= blockEnd) {
    const uint32_t* src_u32_ptr0 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr0 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr1 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr1 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr2 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr2 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr3 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr3 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    *dst_x16_ptr0 = load_nontemporal_uint4(src_u32_ptr0);
    *dst_x16_ptr1 = load_nontemporal_uint4(src_u32_ptr1);
    *dst_x16_ptr2 = load_nontemporal_uint4(src_u32_ptr2);
    *dst_x16_ptr3 = load_nontemporal_uint4(src_u32_ptr3);
  }

  if (i + (2 * COPY_BYTES_PER_BLOCK) <= blockEnd) {
    const uint32_t* src_u32_ptr0 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr0 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr1 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr1 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    *dst_x16_ptr0 = load_nontemporal_uint4(src_u32_ptr0);
    *dst_x16_ptr1 = load_nontemporal_uint4(src_u32_ptr1);
  }

  if (i + COPY_BYTES_PER_THREAD <= blockEnd) {
    const uint32_t* src_u32_ptr0 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr0 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    *dst_x16_ptr0 = load_nontemporal_uint4(src_u32_ptr0);
  }

  // non vectorized copy
  unsigned iEnd = std::min(i + COPY_BYTES_PER_THREAD, blockEnd);
  for (; i < iEnd; i++) {
    dst_ptr[i] = src_ptr[i];
  }
}

__global__ void __muillm_copy_p2p_kernel(
  const uint8_t* src_ptr,
  uint8_t* dst_ptr,
  unsigned N
) {
  __muillm_do_copy_p2p(src_ptr, dst_ptr, N);
}

static muillm_comm_error_t __muillm_gpu_copy(void* dst, const void* src, size_t count, hipStream_t stream) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(count, COPY_BYTES_PER_BLOCK);

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

#define ELEMENTS_PER_THREAD_FP16 8
#define ELEMENTS_PER_BLOCK_FP16 (ELEMENTS_PER_THREAD_FP16 * THREADS_PER_BLOCK)

#define ELEMENTS_PER_THREAD_FP32 4
#define ELEMENTS_PER_BLOCK_FP32 (ELEMENTS_PER_THREAD_FP32 * THREADS_PER_BLOCK)

// TP2 kernels

__global__ void __all_reduce_fp16_tp2_p2p_kernel(
  const half* x1,
  const half* local_x,
  half* y,
  unsigned N
) {
  unsigned i = blockIdx.x * ELEMENTS_PER_BLOCK_FP16 + ELEMENTS_PER_THREAD_FP16 * threadIdx.x;

  if (i + 7 < N) {
    // full block
    // no remainder after that due to the block size being 8 * THREADS_PER_BLOCK

    half8 x1x8 = load_nontemporal_half8(&x1[i]);
    half8 x2x8 = *(const half8*)(&local_x[i]);

    half8 res = __hadd(x1x8, x2x8);

    *((half8*) &y[i]) = res;
  } else {
    // partial block
    if (i + 3 < N) {
      half4 x1x4 = load_nontemporal_half4(&x1[i]);
      half4 x2x4 = *(const half4*)(&local_x[i]);
  
      half4 res = __hadd(x1x4, x2x4);

      *((half4*) &y[i]) = res;

      i += 4;
    }
    if (i + 1 < N) {
      half2 x1x2 = load_nontemporal_half2(&x1[i]);
      half2 x2x2 = *(const half2*)(&local_x[i]);

      half2 res = __hadd(x1x2, x2x2);

      *((half2*) &y[i]) = res;

      i += 2;
    }
    if (i < N) {
      half res = __hadd(x1[i], local_x[i]);

      y[i] = res;
    }
  }
}

__global__ void __all_reduce_fp32_tp2_p2p_kernel(
  const float* x1,
  const float* local_x,
  float* y,
  unsigned N
) {
  unsigned i = blockIdx.x * ELEMENTS_PER_BLOCK_FP32 + ELEMENTS_PER_THREAD_FP32 * threadIdx.x;

  if (i + 3 < N) {
    // full block
    // no remainder after that due to the block size being 4 * THREADS_PER_BLOCK

    float4 x1x4 = load_nontemporal_float4(&x1[i]);
    float4 x2x4 = *(const float4*)(&local_x[i]);

    float4 res = x1x4 + x2x4;

    *((float4*) &y[i]) = res;

  } else {
    // partial block
    if (i + 1 < N) {
      float2 x1x2 = load_nontemporal_float2(&x1[i]);
      float2 x2x2 = *(const float2*)(&local_x[i]);

      float2 res = x1x2 + x2x2;

      *((float2*) &y[i]) = res;

      i += 2;
    }
    if (i < N) {
      float res = x1[i] + local_x[i];
      y[i] = res;
    }
  }
}

// TP4 kernels

__global__ void __all_reduce_fp16_tp4_p2p_kernel(
  const half* x1,
  const half* x2,
  const half* x3,
  const half* local_x,
  half* y,
  unsigned N
) {
  unsigned i = blockIdx.x * ELEMENTS_PER_BLOCK_FP16 + ELEMENTS_PER_THREAD_FP16 * threadIdx.x;


  if (i + 7 < N) {
    // full block
    // no remainder after that due to the block size being 8 * THREADS_PER_BLOCK

    half8 x1x8 = load_nontemporal_half8(&x1[i]);
    half8 x2x8 = load_nontemporal_half8(&x2[i]);
    half8 x3x8 = load_nontemporal_half8(&x3[i]);
    half8 x4x8 = *(const half8*)(&local_x[i]);

    half8 res = __hadd(__hadd(x1x8, x2x8), __hadd(x3x8, x4x8));

    *((half8*) &y[i]) = res;
  } else {
    // partial block
    if (i + 3 < N) {
      half4 x1x4 = load_nontemporal_half4(&x1[i]);
      half4 x2x4 = load_nontemporal_half4(&x2[i]);
      half4 x3x4 = load_nontemporal_half4(&x3[i]);
      half4 x4x4 = *(const half4*)(&local_x[i]);

      half4 res = __hadd(__hadd(x1x4, x2x4), __hadd(x3x4, x4x4));

      *((half4*) &y[i]) = res;

      i += 4;
    }
    if (i + 1 < N) {
      half2 x1x2 = load_nontemporal_half2(&x1[i]);
      half2 x2x2 = load_nontemporal_half2(&x2[i]);
      half2 x3x2 = load_nontemporal_half2(&x3[i]);
      half2 x4x2 = *(const half2*)(&local_x[i]);

      half2 res = __hadd(__hadd(x1x2, x2x2), __hadd(x3x2, x4x2));

      *((half2*) &y[i]) = res;

      i += 2;
    }
    if (i < N) {
      half res = __hadd(__hadd(x1[i], x2[i]), __hadd(x3[i], local_x[i]));
      y[i] = res;
    }
  }
}

__global__ void __all_reduce_fp32_tp4_p2p_kernel(
  const float* x1,
  const float* x2,
  const float* x3,
  const float* local_x,
  float* y,
  unsigned N
) {
  unsigned i = blockIdx.x * ELEMENTS_PER_BLOCK_FP32 + ELEMENTS_PER_THREAD_FP32 * threadIdx.x;

  if (i + 3 < N) {
    // full block
    // no remainder after that due to the block size being 4 * THREADS_PER_BLOCK

    float4 x1x4 = load_nontemporal_float4(&x1[i]);
    float4 x2x4 = load_nontemporal_float4(&x2[i]);
    float4 x3x4 = load_nontemporal_float4(&x3[i]);
    float4 x4x4 = *(const float4*)(&local_x[i]);

    float4 res = (x1x4 + x2x4) + (x3x4 + x4x4);
    *((float4*) &y[i]) = res;
  } else {
    // partial block
    if (i + 1 < N) {
      float2 x1x2 = load_nontemporal_float2(&x1[i]);
      float2 x2x2 = load_nontemporal_float2(&x2[i]);
      float2 x3x2 = load_nontemporal_float2(&x3[i]);
      float2 x4x2 = *(const float2*)(&local_x[i]);

      float2 res = (x1x2 + x2x2) + (x3x2 + x4x2);
      *((float2*) &y[i]) = res;

      i += 2;
    }
    if (i < N) {
      float res = x1[i] + x2[i] + x3[i] + local_x[i];
      y[i] = res;
    }
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
  const half* local_x,
  half* y,
  unsigned N
) {
  unsigned i = blockIdx.x * ELEMENTS_PER_BLOCK_FP16 + ELEMENTS_PER_THREAD_FP16 * threadIdx.x;

  if (i + 7 < N) {
    // full block
    // no remainder after that due to the block size being 8 * THREADS_PER_BLOCK
    half8 x1x8 = load_nontemporal_half8(&x1[i]);
    half8 x2x8 = load_nontemporal_half8(&x2[i]);
    half8 x3x8 = load_nontemporal_half8(&x3[i]);
    half8 x4x8 = load_nontemporal_half8(&x4[i]);
    half8 x5x8 = load_nontemporal_half8(&x5[i]);
    half8 x6x8 = load_nontemporal_half8(&x6[i]);
    half8 x7x8 = load_nontemporal_half8(&x7[i]);
    half8 x8x8 = *(const half8*)(&local_x[i]);

    half8 x1x2x8 = __hadd(x1x8, x2x8);
    half8 x3x4x8 = __hadd(x3x8, x4x8);
    half8 x1x4x8 = __hadd(x1x2x8, x3x4x8);

    half8 x5x6x8 = __hadd(x5x8, x6x8);
    half8 x7x8x8 = __hadd(x7x8, x8x8);
    half8 x5x8x8 = __hadd(x5x6x8, x7x8x8);

    half8 res = __hadd(x1x4x8, x5x8x8);

    *((half8*) &y[i]) = res;

  } else {
    // partial block
    if (i + 3 < N) {
      half4 x1x4 = load_nontemporal_half4(&x1[i]);
      half4 x2x4 = load_nontemporal_half4(&x2[i]);
      half4 x3x4 = load_nontemporal_half4(&x3[i]);
      half4 x4x4 = load_nontemporal_half4(&x4[i]);
      half4 x5x4 = load_nontemporal_half4(&x5[i]);
      half4 x6x4 = load_nontemporal_half4(&x6[i]);
      half4 x7x4 = load_nontemporal_half4(&x7[i]);
      half4 x8x4 = *(const half4*)(&local_x[i]);

      half4 x1x2x4 = __hadd(x1x4, x2x4);
      half4 x3x4x4 = __hadd(x3x4, x4x4);
      half4 x1x4x4 = __hadd(x1x2x4, x3x4x4);

      half4 x5x6x4 = __hadd(x5x4, x6x4);
      half4 x7x8x4 = __hadd(x7x4, x8x4);
      half4 x5x8x4 = __hadd(x5x6x4, x7x8x4);

      half4 res = __hadd(x1x4x4, x5x8x4);

      *((half4*) &y[i]) = res;

      i += 4;
    }
    if (i + 1 < N) {
      half2 x1x2 = load_nontemporal_half2(&x1[i]);
      half2 x2x2 = load_nontemporal_half2(&x2[i]);
      half2 x3x2 = load_nontemporal_half2(&x3[i]);
      half2 x4x2 = load_nontemporal_half2(&x4[i]);
      half2 x5x2 = load_nontemporal_half2(&x5[i]);
      half2 x6x2 = load_nontemporal_half2(&x6[i]);
      half2 x7x2 = load_nontemporal_half2(&x7[i]);
      half2 x8x2 = *(const half2*)(&local_x[i]);

      half2 x1x2x2 = __hadd(x1x2, x2x2);
      half2 x3x4x2 = __hadd(x3x2, x4x2);
      half2 x1x4x2 = __hadd(x1x2x2, x3x4x2);

      half2 x5x6x2 = __hadd(x5x2, x6x2);
      half2 x7x8x2 = __hadd(x7x2, x8x2);
      half2 x5x8x2 = __hadd(x5x6x2, x7x8x2);

      half2 res = __hadd(x1x4x2, x5x8x2);
      *((half2*) &y[i]) = res;

      i += 2;
    }
    if (i < N) {
      half x1x2 = __hadd(x1[i], x2[i]);
      half x3x4 = __hadd(x3[i], x4[i]);
      half x1x4 = __hadd(x1x2, x3x4);
      half x5x6 = __hadd(x5[i], x6[i]);
      half x7x8 = __hadd(x7[i], local_x[i]);
      half x5x8 = __hadd(x5x6, x7x8);
      half res = __hadd(x1x4, x5x8);
      y[i] = res;
    }
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
  const float* local_x,
  float* y,
  unsigned N
) {
  unsigned i = blockIdx.x * ELEMENTS_PER_BLOCK_FP32 + ELEMENTS_PER_THREAD_FP32 * threadIdx.x;

  if (i + 3 < N) {
    float4 x1x4 = load_nontemporal_float4(&x1[i]);
    float4 x2x4 = load_nontemporal_float4(&x2[i]);
    float4 x3x4 = load_nontemporal_float4(&x3[i]);
    float4 x4x4 = load_nontemporal_float4(&x4[i]);
    float4 x5x4 = load_nontemporal_float4(&x5[i]);
    float4 x6x4 = load_nontemporal_float4(&x6[i]);
    float4 x7x4 = load_nontemporal_float4(&x7[i]);
    float4 x8x4 = *(const float4*)(&local_x[i]);

    float4 x1x2x4 = x1x4 + x2x4;
    float4 x3x4x4 = x3x4 + x4x4;
    float4 x1x4x4 = x1x2x4 + x3x4x4;

    float4 x5x6x4 = x5x4 + x6x4;
    float4 x7x8x4 = x7x4 + x8x4;
    float4 x5x8x4 = x5x6x4 + x7x8x4;

    float4 res = x1x4x4 + x5x8x4;
    *((float4*) &y[i]) = res;
  } else {
    // partial block
    if (i + 1 < N) {
      float2 x1x2 = load_nontemporal_float2(&x1[i]);
      float2 x2x2 = load_nontemporal_float2(&x2[i]);
      float2 x3x2 = load_nontemporal_float2(&x3[i]);
      float2 x4x2 = load_nontemporal_float2(&x4[i]);
      float2 x5x2 = load_nontemporal_float2(&x5[i]);
      float2 x6x2 = load_nontemporal_float2(&x6[i]);
      float2 x7x2 = load_nontemporal_float2(&x7[i]);
      float2 x8x2 = *(const float2*)(&local_x[i]);

      float2 x1x2x2 = x1x2 + x2x2;
      float2 x3x4x2 = x3x2 + x4x2;
      float2 x1x4x2 = x1x2x2 + x3x4x2;

      float2 x5x6x2 = x5x2 + x6x2;
      float2 x7x8x2 = x7x2 + x8x2;
      float2 x5x8x2 = x5x6x2 + x7x8x2;

      float2 res = x1x4x2 + x5x8x2;
      *((float2*) &y[i]) = res;
      i += 2;
    }

    if (i < N) {
      float x1x2 = (x1[i] + x2[i]);
      float x3x4 = (x3[i] + x4[i]);
      float x1x4 = x1x2 + x3x4;
      float x5x6 = (x5[i] + x6[i]);
      float x7x8 = (x7[i] + local_x[i]);
      float x5x8 = x5x6 + x7x8;
      float res = x1x4 + x5x8;
      y[i] = res;
    }
  }
}

static inline muillm_comm_error_t __muillm_reduce_chunk(
  muillm_comm_p2p_t* comm,
  const void** src_ptrs,
  void* dst_ptr,
  size_t offset,
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

  if (count == 0) {
    // might happen, e.g. during the p2p check or when we have a very small array
    // and doing two step reduce
    // in that case we don't need to do a reduction
    return MUILLM_COMM_SUCCESS;
  }

  // do the reduction
  const int threads_per_blocks = THREADS_PER_BLOCK;

  const void* packed_src_ptrs[MUILLM_COMM_MAX_GPUS];
  int p = 0;
  for (int d = 0; d < local_size; d++) {
    if (d == local_rank) continue;
    packed_src_ptrs[p] = src_ptrs[d];
    p++;
  }

  if (datatype == MUILLM_COMM_FP16) {
    const int num_blocks = DIV_ROUND_UP(count, ELEMENTS_PER_BLOCK_FP16);
    if (local_size == 8) {
      __all_reduce_fp16_tp8_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        ((const half*) packed_src_ptrs[0]) + offset,
        ((const half*) packed_src_ptrs[1]) + offset,
        ((const half*) packed_src_ptrs[2]) + offset,
        ((const half*) packed_src_ptrs[3]) + offset,
        ((const half*) packed_src_ptrs[4]) + offset,
        ((const half*) packed_src_ptrs[5]) + offset,
        ((const half*) packed_src_ptrs[6]) + offset,
        ((const half*) src_ptrs[local_rank]) + offset,
        ((half*) dst_ptr) + offset,
        count
      );
    } else if (local_size == 4) {
      __all_reduce_fp16_tp4_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        ((const half*) packed_src_ptrs[0]) + offset,
        ((const half*) packed_src_ptrs[1]) + offset,
        ((const half*) packed_src_ptrs[2]) + offset,
        ((const half*) src_ptrs[local_rank]) + offset,
        ((half*) dst_ptr) + offset,
        count
      );
    } else if (local_size == 2) {
      __all_reduce_fp16_tp2_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        ((const half*) packed_src_ptrs[0]) + offset,
        ((const half*) src_ptrs[local_rank]) + offset,
        ((half*) dst_ptr) + offset,
        count
      );
    } else {
      std::cout<<"reduction unsupported tp size"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else if (datatype == MUILLM_COMM_FP32) {
    const int num_blocks = DIV_ROUND_UP(count, ELEMENTS_PER_BLOCK_FP32);
    if (local_size == 8) {
      __all_reduce_fp32_tp8_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        ((const float*) packed_src_ptrs[0]) + offset,
        ((const float*) packed_src_ptrs[1]) + offset,
        ((const float*) packed_src_ptrs[2]) + offset,
        ((const float*) packed_src_ptrs[3]) + offset,
        ((const float*) packed_src_ptrs[4]) + offset,
        ((const float*) packed_src_ptrs[5]) + offset,
        ((const float*) packed_src_ptrs[6]) + offset,
        ((const float*) src_ptrs[local_rank]) + offset,
        ((float*) dst_ptr) + offset,
        count
      );
    } else if (local_size == 4) {
      __all_reduce_fp32_tp4_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        ((const float*) packed_src_ptrs[0]) + offset,
        ((const float*) packed_src_ptrs[1]) + offset,
        ((const float*) packed_src_ptrs[2]) + offset,
        ((const float*) src_ptrs[local_rank]) + offset,
        ((float*) dst_ptr) + offset,
        count
      );
    } else if (local_size == 2) {
      __all_reduce_fp32_tp2_p2p_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        ((const float*) packed_src_ptrs[0]) + offset,
        ((const float*) src_ptrs[local_rank]) + offset,
        ((float*) dst_ptr) + offset,
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

  if ((hip_error = hipPeekAtLastError()) != hipSuccess) {
    printf("p2p reduce chunk failed: %s\n", hipGetErrorString(hip_error));
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}

// one step algorithm where each GPU reads from all the other
// GPUs and reduces in their own buffer
// lower latency than the two steps algorithm for small sizes
// (not communication efficient for large sizes as all GPUs read the entire
// array from al the other GPUs)
muillm_comm_error_t muillm_comm_p2p_one_step_placed_all_reduce_sum(
  muillm_comm_p2p_t* comm,
  const void** src_ptrs,
  void* dst_ptr,
  size_t count,
  muillm_comm_datatype_t datatype,
  hipStream_t stream
) {
  // reduce as a single chunk
  return __muillm_reduce_chunk(
    comm,
    src_ptrs,
    dst_ptr,
    /* offset */ 0,
    count,
    datatype,
    stream
  );
}

typedef enum x_copy {
  X1_COPY,
  X2_COPY,
  X4_COPY
} x_copy_t;

// TODO: for tp2 we could precompute the offsets and use the normal copy kernel
__global__ void __muillm_chunk_copy_fp16_tp2_p2p_kernel(
  const half* src_ptr0,
  const half* src_ptr1,
  half* dst_ptr,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  // and skip the block for the local rank
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  const half* src_ptrs[2] = {src_ptr0, src_ptr1};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptrs[chunk_idx] + chunk_start),
    (uint8_t*) (dst_ptr + chunk_start),
    actual_chunk_size * 2,
    block_chunk_size
  );
}

__global__ void __muillm_chunk_copy_fp32_tp2_p2p_kernel(
  const float* src_ptr0,
  const float* src_ptr1,
  float* dst_ptr,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  // and skip the block for the local rank
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  const float* src_ptrs[2] = {src_ptr0, src_ptr1};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptrs[chunk_idx] + chunk_start),
    (uint8_t*) (dst_ptr + chunk_start),
    actual_chunk_size * 4,
    block_chunk_size
  );
}

__global__ void __muillm_chunk_copy_fp16_tp4_p2p_kernel(
  const half* src_ptr0,
  const half* src_ptr1,
  const half* src_ptr2,
  const half* src_ptr3,
  half* dst_ptr,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  // and skip the block for the local rank
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  const half* src_ptrs[4] = {src_ptr0, src_ptr1, src_ptr2, src_ptr3};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptrs[chunk_idx] + chunk_start),
    (uint8_t*) (dst_ptr + chunk_start),
    actual_chunk_size * 2,
    block_chunk_size
  );
}

__global__ void __muillm_chunk_copy_fp32_tp4_p2p_kernel(
  const float* src_ptr0,
  const float* src_ptr1,
  const float* src_ptr2,
  const float* src_ptr3,
  float* dst_ptr,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  // and skip the block for the local rank
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  const float* src_ptrs[4] = {src_ptr0, src_ptr1, src_ptr2, src_ptr3};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptrs[chunk_idx] + chunk_start),
    (uint8_t*) (dst_ptr + chunk_start),
    actual_chunk_size * 4,
    block_chunk_size
  );
}

__global__ void __muillm_chunk_copy_fp16_tp8_p2p_kernel(
  const half* src_ptr0,
  const half* src_ptr1,
  const half* src_ptr2,
  const half* src_ptr3,
  const half* src_ptr4,
  const half* src_ptr5,
  const half* src_ptr6,
  const half* src_ptr7,
  half* dst_ptr,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  // and skip the block for the local rank
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  const half* src_ptrs[8] = {src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_ptr4, src_ptr5, src_ptr6, src_ptr7};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptrs[chunk_idx] + chunk_start),
    (uint8_t*) (dst_ptr + chunk_start),
    actual_chunk_size * 2,
    block_chunk_size
  );
}

__global__ void __muillm_chunk_copy_fp32_tp8_p2p_kernel(
  const float* src_ptr0,
  const float* src_ptr1,
  const float* src_ptr2,
  const float* src_ptr3,
  const float* src_ptr4,
  const float* src_ptr5,
  const float* src_ptr6,
  const float* src_ptr7,
  float* dst_ptr,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  // and skip the block for the local rank
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  const float* src_ptrs[8] = {src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_ptr4, src_ptr5, src_ptr6, src_ptr7};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptrs[chunk_idx] + chunk_start),
    (uint8_t*) (dst_ptr + chunk_start),
    actual_chunk_size * 4,
    block_chunk_size
  );
}

// two step algorithm where
// 1) each GPU reduces one chunk by reading from all the other
// GPUs and reduces that chunk in their own buffer
// 2) each GPU gathers all the other chunks from the other GPUs
// (communication efficient)
muillm_comm_error_t muillm_comm_p2p_two_steps_placed_all_reduce_sum(
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

  if (local_size > 8) {
    std::cout<<"reduction unsupported tp size"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  //
  // 1) reduce the chunk we are responsible for
  //

  // we do a straight division instead of rounding up
  // so that reads are aligned on cache lines
  size_t chunk_size = count / local_size;
  size_t max_chunk_size = chunk_size + (count % local_size);

  size_t chunk_start = local_rank * chunk_size;
  // the chunk end is the end of the array for the last rank, otherwise of the size of a chunk
  size_t chunk_end = (local_rank == (local_size - 1)) ? count : (chunk_start + chunk_size);

  size_t actual_chunk_size = chunk_end - chunk_start;

  if ((muillm_error = __muillm_reduce_chunk(
    comm,
    src_ptrs,
    (void*) src_ptrs[local_rank], // dest for this reduce is the src buffer
    chunk_start,
    actual_chunk_size,
    datatype,
    stream
  )) != MUILLM_COMM_SUCCESS) {
    std::cout<<"p2p reduce chunk failed"<<std::endl;
    return muillm_error;
  }

  //
  // 2) gather all the chunks
  //

  if ((muillm_error = __mui_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"p2p reduction barrier failed"<<std::endl;
    return muillm_error;
  }

  const int threads_per_blocks = THREADS_PER_BLOCK;
  int num_other_chunks = local_size - 1;

  int num_simd_lanes = comm->gpu_info->simd_lanes;

  if (datatype == MUILLM_COMM_FP16) {
    int num_blocks = DIV_ROUND_UP(max_chunk_size, COPY_FP16S_PER_BLOCK);

    int num_total_blocks = num_blocks * local_size;
    int num_total_threads = num_total_blocks * threads_per_blocks;
    unsigned block_chunk_size = COPY_BYTES_PER_BLOCK;

    while (num_total_threads > 2 * num_simd_lanes && num_blocks > 1) {
      num_total_threads /= 2;
      num_blocks /= 2;
      num_total_blocks /= 2;
      block_chunk_size *= 2;
    }

    if (local_size == 8) {
      __muillm_chunk_copy_fp16_tp8_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) src_ptrs[0],
        (const half*) src_ptrs[1],
        (const half*) src_ptrs[2],
        (const half*) src_ptrs[3],
        (const half*) src_ptrs[4],
        (const half*) src_ptrs[5],
        (const half*) src_ptrs[6],
        (const half*) src_ptrs[7],
        (half*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 4) {
      __muillm_chunk_copy_fp16_tp4_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) src_ptrs[0],
        (const half*) src_ptrs[1],
        (const half*) src_ptrs[2],
        (const half*) src_ptrs[3],
        (half*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 2) {
      __muillm_chunk_copy_fp16_tp2_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) src_ptrs[0],
        (const half*) src_ptrs[1],
        (half*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else {
      std::cout<<"reduction unsupported tp size"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else if (datatype == MUILLM_COMM_FP32) {
    int num_blocks = DIV_ROUND_UP(max_chunk_size, COPY_FP32S_PER_BLOCK);

    int num_total_blocks = num_blocks * local_size;
    int num_total_threads = num_total_blocks * threads_per_blocks;
    unsigned block_chunk_size = COPY_BYTES_PER_BLOCK;

    while (num_total_threads > 2 * num_simd_lanes && num_blocks > 1) {
      num_total_threads /= 2;
      num_blocks /= 2;
      num_total_blocks /= 2;
      block_chunk_size *= 2;
    }

    if (local_size == 8) {
      __muillm_chunk_copy_fp32_tp8_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) src_ptrs[0],
        (const float*) src_ptrs[1],
        (const float*) src_ptrs[2],
        (const float*) src_ptrs[3],
        (const float*) src_ptrs[4],
        (const float*) src_ptrs[5],
        (const float*) src_ptrs[6],
        (const float*) src_ptrs[7],
        (float*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 4) {
      __muillm_chunk_copy_fp32_tp4_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) src_ptrs[0],
        (const float*) src_ptrs[1],
        (const float*) src_ptrs[2],
        (const float*) src_ptrs[3],
        (float*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 2) {
      __muillm_chunk_copy_fp32_tp2_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) src_ptrs[0],
        (const float*) src_ptrs[1],
        (float*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
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
    printf("p2p chunk copy failed\n");
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}


// TODO: for tp2 we could precompute the offsets and use the normal copy kernel
__global__ void __muillm_chunk_broadcast_fp16_tp2_p2p_kernel(
  const half* src_ptr,
  half* dst_ptr0,
  half* dst_ptr1,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  half* dst_ptrs[2] = {dst_ptr0, dst_ptr1};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptr + chunk_start),
    (uint8_t*) (dst_ptrs[chunk_idx] + local_rank * actual_chunk_size),
    actual_chunk_size * 2,
    block_chunk_size
  );
}

__global__ void __muillm_chunk_broadcast_fp32_tp2_p2p_kernel(
  const float* src_ptr,
  float* dst_ptr0,
  float* dst_ptr1,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  float* dst_ptrs[2] = {dst_ptr0, dst_ptr1};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptr + chunk_start),
    (uint8_t*) (dst_ptrs[chunk_idx] + local_rank * actual_chunk_size),
    actual_chunk_size * 4,
    block_chunk_size
  );
}

__global__ void __muillm_chunk_broadcast_fp16_tp4_p2p_kernel(
  const half* src_ptr,
  half* dst_ptr0,
  half* dst_ptr1,
  half* dst_ptr2,
  half* dst_ptr3,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  half* dst_ptrs[4] = {dst_ptr0, dst_ptr1, dst_ptr2, dst_ptr3};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptr + chunk_start),
    (uint8_t*) (dst_ptrs[chunk_idx] + local_rank * actual_chunk_size),
    actual_chunk_size * 2,
    block_chunk_size
  );
}

__global__ void __muillm_chunk_broadcast_fp32_tp4_p2p_kernel(
  const float* src_ptr,
  float* dst_ptr0,
  float* dst_ptr1,
  float* dst_ptr2,
  float* dst_ptr3,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  float* dst_ptrs[4] = {dst_ptr0, dst_ptr1, dst_ptr2, dst_ptr3};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptr + chunk_start),
    (uint8_t*) (dst_ptrs[chunk_idx] + local_rank * actual_chunk_size),
    actual_chunk_size * 4,
    block_chunk_size
  );
}

__global__ void __muillm_chunk_broadcast_fp16_tp8_p2p_kernel(
  const half* src_ptr,
  half* dst_ptr0,
  half* dst_ptr1,
  half* dst_ptr2,
  half* dst_ptr3,
  half* dst_ptr4,
  half* dst_ptr5,
  half* dst_ptr6,
  half* dst_ptr7,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  half* dst_ptrs[8] = {dst_ptr0, dst_ptr1, dst_ptr2, dst_ptr3, dst_ptr4, dst_ptr5, dst_ptr6, dst_ptr7};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptr + chunk_start),
    (uint8_t*) (dst_ptrs[chunk_idx] + local_rank * actual_chunk_size),
    actual_chunk_size * 2,
    block_chunk_size
  );
}

__global__ void __muillm_chunk_broadcast_fp32_tp8_p2p_kernel(
  const float* src_ptr,
  float* dst_ptr0,
  float* dst_ptr1,
  float* dst_ptr2,
  float* dst_ptr3,
  float* dst_ptr4,
  float* dst_ptr5,
  float* dst_ptr6,
  float* dst_ptr7,
  unsigned N,
  unsigned chunk_size,
  unsigned block_chunk_size,
  unsigned local_rank,
  unsigned local_size
) {
  // we launch one series of blocks per other rank
  // so we need to adjust the block index to get the right chunk
  unsigned chunk_idx = blockIdx.y;

  unsigned chunk_start = chunk_idx * chunk_size;
  unsigned chunk_end = (chunk_idx == (local_size - 1)) ? N : (chunk_start + chunk_size);
  unsigned actual_chunk_size = chunk_end - chunk_start;

  float* dst_ptrs[8] = {dst_ptr0, dst_ptr1, dst_ptr2, dst_ptr3, dst_ptr4, dst_ptr5, dst_ptr6, dst_ptr7};

  // copy the chunk
  __muillm_do_copy_p2p(
    (const uint8_t*) (src_ptr + chunk_start),
    (uint8_t*) (dst_ptrs[chunk_idx] + local_rank * actual_chunk_size),
    actual_chunk_size * 4,
    block_chunk_size
  );
}

// three step algorithm where
// 1) each GPU broadcast their chunks to the other GPUs
// 2) GPUs reduce their chunk in their own buffer
// 3) GPUs broadcast their reduced chunk to the other GPUs
// (communication efficient)
muillm_comm_error_t muillm_comm_p2p_three_steps_placed_all_reduce_sum(
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

  if (local_size > 8) {
    std::cout<<"reduction unsupported tp size"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // get the reduction buffer set in which we will store the broadcasted chunks
  // from the first phase
  muillm_comm_p2p_buffer_set_t* buffer_set = nullptr;

  void** tmp_broadcast_buffers = (void**)buffer_set->buffers;

  if ((muillm_error = muillm_comm_p2p_get_buffer_set(comm, count, datatype, &buffer_set, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"Reduction failed when ensuring capacity"<<std::endl;
    return muillm_error;
  }

  //
  // 1) each GPU broadcast their chunks to the other GPUs
  //

  // we do a straight division instead of rounding up
  // so that reads are aligned on cache lines
  size_t chunk_size = count / local_size;
  size_t max_chunk_size = chunk_size + (count % local_size);

  size_t chunk_start = local_rank * chunk_size;
  // the chunk end is the end of the array for the last rank, otherwise of the size of a chunk
  size_t chunk_end = (local_rank == (local_size - 1)) ? count : (chunk_start + chunk_size);

  size_t actual_chunk_size = chunk_end - chunk_start;


  const int threads_per_blocks = THREADS_PER_BLOCK;
  int num_other_chunks = local_size - 1;

  int num_simd_lanes = comm->gpu_info->simd_lanes;

  if (datatype == MUILLM_COMM_FP16) {
    int num_blocks = DIV_ROUND_UP(max_chunk_size, COPY_FP16S_PER_BLOCK);

    int num_total_blocks = num_blocks * local_size;
    int num_total_threads = num_total_blocks * threads_per_blocks;
    unsigned block_chunk_size = COPY_BYTES_PER_BLOCK;

    while (num_total_threads > 2 * num_simd_lanes && num_blocks > 1) {
      num_total_threads /= 2;
      num_blocks /= 2;
      num_total_blocks /= 2;
      block_chunk_size *= 2;
    }

    if (local_size == 8) {
      __muillm_chunk_broadcast_fp16_tp8_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) src_ptrs[local_rank],
        (half*) tmp_broadcast_buffers[0],
        (half*) tmp_broadcast_buffers[1],
        (half*) tmp_broadcast_buffers[2],
        (half*) tmp_broadcast_buffers[3],
        (half*) tmp_broadcast_buffers[4],
        (half*) tmp_broadcast_buffers[5],
        (half*) tmp_broadcast_buffers[6],
        (half*) tmp_broadcast_buffers[7],
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 4) {
      __muillm_chunk_broadcast_fp16_tp4_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) src_ptrs[local_rank],
        (half*) tmp_broadcast_buffers[0],
        (half*) tmp_broadcast_buffers[1],
        (half*) tmp_broadcast_buffers[2],
        (half*) tmp_broadcast_buffers[3],
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 2) {
      __muillm_chunk_broadcast_fp16_tp2_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) src_ptrs[local_rank],
        (half*) tmp_broadcast_buffers[0],
        (half*) tmp_broadcast_buffers[1],
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else {
      std::cout<<"reduction unsupported tp size"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else if (datatype == MUILLM_COMM_FP32) {
    int num_blocks = DIV_ROUND_UP(max_chunk_size, COPY_FP32S_PER_BLOCK);

    int num_total_blocks = num_blocks * local_size;
    int num_total_threads = num_total_blocks * threads_per_blocks;
    unsigned block_chunk_size = COPY_BYTES_PER_BLOCK;

    while (num_total_threads > 2 * num_simd_lanes && num_blocks > 1) {
      num_total_threads /= 2;
      num_blocks /= 2;
      num_total_blocks /= 2;
      block_chunk_size *= 2;
    }

    if (local_size == 8) {
      __muillm_chunk_broadcast_fp32_tp8_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) src_ptrs[local_rank],
        (float*) tmp_broadcast_buffers[0],
        (float*) tmp_broadcast_buffers[1],
        (float*) tmp_broadcast_buffers[2],
        (float*) tmp_broadcast_buffers[3],
        (float*) tmp_broadcast_buffers[4],
        (float*) tmp_broadcast_buffers[5],
        (float*) tmp_broadcast_buffers[6],
        (float*) tmp_broadcast_buffers[7],
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 4) {
      __muillm_chunk_broadcast_fp32_tp4_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) src_ptrs[local_rank],
        (float*) tmp_broadcast_buffers[0],
        (float*) tmp_broadcast_buffers[1],
        (float*) tmp_broadcast_buffers[2],
        (float*) tmp_broadcast_buffers[3],
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 2) {
      __muillm_chunk_broadcast_fp32_tp2_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) src_ptrs[local_rank],
        (float*) tmp_broadcast_buffers[0],
        (float*) tmp_broadcast_buffers[1],
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
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
    printf("p2p chunk broadcast failed\n");
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // make sure all GPUs have arrived
  if ((muillm_error = __mui_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"p2p chunk broadcast barrier failed"<<std::endl;
    return muillm_error;
  }
    
  //
  // 2) GPUs reduce their chunk in their own buffer
  //
  void* tmp_ptrs[MUILLM_COMM_MAX_GPUS];
  if (datatype == MUILLM_COMM_FP16) {
    for (int i = 0; i < local_size; i++) {
      tmp_ptrs[i] = ((half*)tmp_broadcast_buffers[local_rank]) + i * actual_chunk_size;
    }
  } else if (datatype == MUILLM_COMM_FP32) {
    for (int i = 0; i < local_size; i++) {
      tmp_ptrs[i] = ((float*)tmp_broadcast_buffers[local_rank]) + i * actual_chunk_size;
    }
  } else {
    std::cout<<"reduction unsupported dtype"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // reduce the chunk we are responsible for
  if ((muillm_error = __muillm_reduce_chunk(
    comm,
    (const void**) tmp_ptrs,
    (void*) tmp_ptrs[local_rank], // dest for this reduced chunk is the temp buffer
    0,
    actual_chunk_size,
    datatype,
    stream
  )) != MUILLM_COMM_SUCCESS) {
    std::cout<<"p2p reduce chunk failed"<<std::endl;
    return muillm_error;
  }

  // make sure all GPUs have arrived
  if ((muillm_error = __mui_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"p2p reduce chunk barrier failed"<<std::endl;
    return muillm_error;
  }

  //
  // 3) GPUs read the reduced chunks from the other GPUs
  //

  if (datatype == MUILLM_COMM_FP16) {
    int num_blocks = DIV_ROUND_UP(max_chunk_size, COPY_FP16S_PER_BLOCK);

    int num_total_blocks = num_blocks * local_size;
    int num_total_threads = num_total_blocks * threads_per_blocks;
    unsigned block_chunk_size = COPY_BYTES_PER_BLOCK;

    while (num_total_threads > 2 * num_simd_lanes && num_blocks > 1) {
      num_total_threads /= 2;
      num_blocks /= 2;
      num_total_blocks /= 2;
      block_chunk_size *= 2;
    }

    if (local_size == 8) {
      __muillm_chunk_copy_fp16_tp8_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) tmp_broadcast_buffers[0],
        (const half*) tmp_broadcast_buffers[1],
        (const half*) tmp_broadcast_buffers[2],
        (const half*) tmp_broadcast_buffers[3],
        (const half*) tmp_broadcast_buffers[4],
        (const half*) tmp_broadcast_buffers[5],
        (const half*) tmp_broadcast_buffers[6],
        (const half*) tmp_broadcast_buffers[7],
        (half*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 4) {
      __muillm_chunk_copy_fp16_tp4_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) tmp_broadcast_buffers[0],
        (const half*) tmp_broadcast_buffers[1],
        (const half*) tmp_broadcast_buffers[2],
        (const half*) tmp_broadcast_buffers[3],
        (half*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 2) {
      __muillm_chunk_copy_fp16_tp2_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const half*) tmp_broadcast_buffers[0],
        (const half*) tmp_broadcast_buffers[1],
        (half*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else {
      std::cout<<"reduction unsupported tp size"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else if (datatype == MUILLM_COMM_FP32) {
    int num_blocks = DIV_ROUND_UP(max_chunk_size, COPY_FP32S_PER_BLOCK);

    int num_total_blocks = num_blocks * local_size;
    int num_total_threads = num_total_blocks * threads_per_blocks;
    unsigned block_chunk_size = COPY_BYTES_PER_BLOCK;

    while (num_total_threads > 2 * num_simd_lanes && num_blocks > 1) {
      num_total_threads /= 2;
      num_blocks /= 2;
      num_total_blocks /= 2;
      block_chunk_size *= 2;
    }

    if (local_size == 8) {
      __muillm_chunk_copy_fp32_tp8_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) tmp_broadcast_buffers[0],
        (const float*) tmp_broadcast_buffers[1],
        (const float*) tmp_broadcast_buffers[2],
        (const float*) tmp_broadcast_buffers[3],
        (const float*) tmp_broadcast_buffers[4],
        (const float*) tmp_broadcast_buffers[5],
        (const float*) tmp_broadcast_buffers[6],
        (const float*) tmp_broadcast_buffers[7],
        (float*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 4) {
      __muillm_chunk_copy_fp32_tp4_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) tmp_broadcast_buffers[0],
        (const float*) tmp_broadcast_buffers[1],
        (const float*) tmp_broadcast_buffers[2],
        (const float*) tmp_broadcast_buffers[3],
        (float*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
      );
    } else if (local_size == 2) {
      __muillm_chunk_copy_fp32_tp2_p2p_kernel<<<dim3(num_blocks, local_size), THREADS_PER_BLOCK, 0, stream>>>(
        (const float*) tmp_broadcast_buffers[0],
        (const float*) tmp_broadcast_buffers[1],
        (float*) dst_ptr,
        count,
        chunk_size,
        block_chunk_size,
        local_rank,
        local_size
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
    printf("p2p chunk copy failed\n");
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // TODO: maybe unneeded?
  // make sure all GPUs have arrived
  if ((muillm_error = __mui_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"p2p chunk copy barrier failed"<<std::endl;
    return muillm_error;
  }
  
  return MUILLM_COMM_SUCCESS;
}

#define LOW_LATENCY_THRESHOLD (1024 * 256)

muillm_comm_error_t muillm_comm_p2p_placed_all_reduce_sum(
  muillm_comm_p2p_t* comm,
  const void** src_ptrs,
  void* dst_ptr,
  size_t count,
  muillm_comm_datatype_t datatype,
  hipStream_t stream
) {
  int local_size = comm->local_size;
  if (false) {
  //if (local_size <= 2 || count < LOW_LATENCY_THRESHOLD) {
    // the one step algorithm is good for small sizes
    // of for tp2
    return muillm_comm_p2p_one_step_placed_all_reduce_sum(
      comm,
      src_ptrs,
      dst_ptr,
      count,
      datatype,
      stream
    );
  } else if (false) {
    // the two step algorithm is better for large sizes
    return muillm_comm_p2p_two_steps_placed_all_reduce_sum(
      comm,
      src_ptrs,
      dst_ptr,
      count,
      datatype,
      stream
    );
  } else {
    return muillm_comm_p2p_three_steps_placed_all_reduce_sum(
      comm,
      src_ptrs,
      dst_ptr,
      count,
      datatype,
      stream
    );
  }
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

  // TODO: this copy is not required for the three step reduce

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