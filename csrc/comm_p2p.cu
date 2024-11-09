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

static void __allocate_wait_events(
  muillm_comm_p2p_t* comm
);

static muillm_comm_error_t __init_p2p_recv(
    muillm_comm_p2p_t* comm
);

static void __reallocate_p2p_recv_buffer(
    muillm_comm_p2p_t* comm,
    size_t required_recv_buffer_size,
    hipStream_t stream
);

static void __ensure_p2p_buffer_capacity(
    muillm_comm_p2p_t* comm,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  size_t required_recv_buffer_size = ALIGN_UP(__comm_size(datatype, count), GPU_CACHELINE_SIZE);

  if (comm->recv_buffer_size >= required_recv_buffer_size) {
    // we have sufficient buffer space
    return;
  }

  printf("(rank %d) Reallocating recv_buffers\n", local_rank);

  // TODO error handling
  __reallocate_p2p_recv_buffer(
    (muillm_comm_p2p_t*) comm,
    required_recv_buffer_size,
    stream
  );

  printf("(rank %d) Allocated receive buffers\n", local_rank);
}

muillm_comm_error_t __init_p2p_comm(
    int world_size,
    int local_size,
    int rank,
    int local_rank,
    muillm_comm_p2p_t** comm_ptr
) {
  if (world_size != local_size) {
    // we currently ony support single machine, so
    // we should fail
    return MUILLM_COMM_UNSUPPORTED_SIZE;
  }

  printf("(rank %d local_rank %d) Initializing comm for world_size %d local_size %d ...\n", rank, local_rank, world_size, local_size);

  muillm_comm_error_t error;

  muillm_comm_method_t transfer_method = MUILLM_COMM_METHOD_P2P_TRANSFER;

  // create the comm object
  muillm_comm_p2p_t* comm = nullptr;
  comm = new muillm_comm_p2p_t;
  comm->transfer_method = transfer_method;

  comm->world_size = world_size;
  comm->local_size = local_size;
  comm->rank = rank;
  comm->local_rank = local_rank;

  comm->all_reduce_no = 0;

  // establish the local socket connection
  printf("(rank %d) Opening local socket...\n", local_rank);
  error = __open_local_socket(comm, local_size, local_rank);
  if (error != MUILLM_COMM_SUCCESS) {
    return error;
  }
  printf("(rank %d) Opened local socket\n", local_rank);

  
  if (hipSetDevice(local_rank) != hipSuccess) {
    printf("(rank %d) Failed to set device\n", local_rank);
    return MUILLM_UNKNOWN_ERROR;
  }

  // allocate the wait buffer
  // TODO: check for errors
  __allocate_wait_events(comm);
  // TODO: check for errors
  __init_p2p_recv(comm);

  // allocate an initial buffer
  // TODO: check for errors
  __ensure_p2p_buffer_capacity(comm, 1024*1024, MUILLM_COMM_FP32, 0);

  // return the comm object
  printf("(rank %d) Created comm %p\n", local_rank, comm);
  
  *comm_ptr = comm;

  return MUILLM_COMM_SUCCESS;
}

static inline void __swap_p2p_recv_buffer_sets(muillm_comm_p2p_t* comm) {
  void* p2p_recv_buffer_temp = comm->p2p_recv_buffer_set.p2p_recv_buffer;
  void** all_p2p_recv_buffers_temp = comm->p2p_recv_buffer_set.all_p2p_recv_buffers;
  void** all_p2p_recv_buffers_device_temp = comm->p2p_recv_buffer_set.all_p2p_recv_buffers_device;
  hipMemPool_t* memPools_temp = comm->p2p_recv_buffer_set.memPools;

  comm->p2p_recv_buffer_set.p2p_recv_buffer = comm->second_p2p_recv_buffer_set.p2p_recv_buffer;
  comm->p2p_recv_buffer_set.all_p2p_recv_buffers = comm->second_p2p_recv_buffer_set.all_p2p_recv_buffers;
  comm->p2p_recv_buffer_set.all_p2p_recv_buffers_device = comm->second_p2p_recv_buffer_set.all_p2p_recv_buffers_device;
  comm->p2p_recv_buffer_set.memPools = comm->second_p2p_recv_buffer_set.memPools;

  comm->second_p2p_recv_buffer_set.p2p_recv_buffer = p2p_recv_buffer_temp;
  comm->second_p2p_recv_buffer_set.all_p2p_recv_buffers = all_p2p_recv_buffers_temp;
  comm->second_p2p_recv_buffer_set.all_p2p_recv_buffers_device = all_p2p_recv_buffers_device_temp;
  comm->second_p2p_recv_buffer_set.memPools = memPools_temp;
}

// returns the number of receive buffers needed to do any of the operations
static inline size_t __comm_buffer_set_size(
    muillm_comm_p2p_t* comm
) {
  // we need a single buffer
  return 1;
}

static void __allocate_wait_event_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_event_set_t* event_set
) {
  int local_rank = comm->local_rank;
  int local_size = comm->local_size;

  event_set->all_events = new hipEvent_t[local_size];

  // allocate local event
  if (hipEventCreateWithFlags(&event_set->event, hipEventDisableTiming | hipEventReleaseToSystem | hipEventInterprocess) != hipSuccess) {
    printf("(rank %d) event creation failed\n", local_rank);
    return;
  } 

  // get event handle
  hipIpcEventHandle_t event_handle;
  if (hipIpcGetEventHandle(&event_handle, event_set->event) != hipSuccess) {
    printf("(rank %d) could not get event handle\n", local_rank);
    return;
  }

#if 1
  // gather all event handles
  hipIpcEventHandle_t* allHandles = new hipIpcEventHandle_t[local_size];

  // gather all memory handles
  __local_socket_all_gather(comm, &event_handle, sizeof(hipIpcEventHandle_t), allHandles);

  // open all even handles (except ours)
  for (int r = 0; r < local_size; r++) {
    if (r == local_rank) {
      event_set->all_events[r] = event_set->event;
    } else {
      if (hipIpcOpenEventHandle(&event_set->all_events[r], allHandles[r]) != hipSuccess) {
        printf("(rank %d) could open event handle\n", local_rank);
        return;
      }
    }
  }

  delete[] allHandles;
#else
  // gather all memory handles
  __local_socket_all_gather(comm, &event_set->event, sizeof(hipEvent_t), event_set->all_events);
#endif
}

static void __allocate_wait_events(
  muillm_comm_p2p_t* comm
) {
  int local_rank = comm->local_rank;
  __allocate_wait_event_set(comm, &comm->wait_event_set);
  __allocate_wait_event_set(comm, &comm->second_wait_event_set);
  printf("(rank %d) Allocated wait events\n", local_rank);
}

static muillm_comm_error_t __init_p2p_recv_buffer_set(
    muillm_comm_p2p_t* comm,
    muillm_comm_p2p_recv_buffer_set_t* buffer_set
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  if (hipSetDevice(local_rank) != hipSuccess) {
    printf("(rank %d) Failed to set device\n", local_rank);
    return MUILLM_UNKNOWN_ERROR;
  }

  buffer_set->p2p_recv_buffer = nullptr;
  buffer_set->all_p2p_recv_buffers = new void*[local_size];

  for (int d = 0; d < local_size; d++) {
    buffer_set->all_p2p_recv_buffers[d] = nullptr;
  }

  buffer_set->all_p2p_recv_buffers_device = nullptr;
  if (hipMalloc((void**) &buffer_set->all_p2p_recv_buffers_device, local_size * sizeof(void*)) != hipSuccess) {
    printf("failed to allocated all buffers device");
    return MUILLM_UNKNOWN_ERROR;
  }

  if (hipMemset(buffer_set->all_p2p_recv_buffers_device, 0, local_size * sizeof(void*)) != hipSuccess) {
    printf("failed to set all buffers device");
    return MUILLM_UNKNOWN_ERROR;
  }

  //create mem pools
  buffer_set->memPools = new hipMemPool_t[local_size];
  for (int d = 0; d < local_size; d++) {
    if (d == local_rank) continue;

    if (hipSetDevice(d) != hipSuccess) {
      printf("(rank %d) Failed to set device\n", local_rank);
      return MUILLM_UNKNOWN_ERROR;
    }

    // Create a memory pool with default properties.
    hipMemPoolProps poolProps = {};
    poolProps.allocType = hipMemAllocationTypePinned;
    poolProps.handleTypes = hipMemHandleTypePosixFileDescriptor;
    poolProps.location.type = hipMemLocationTypeDevice;
    poolProps.location.id = d; // Assuming this remote device.

    hipMemPool_t memPool;
    if (hipMemPoolCreate(&memPool, &poolProps) != hipSuccess) {
      printf("(rank %d) Mem pool creation failed for %d\n", local_rank, d);
      return MUILLM_UNKNOWN_ERROR;
    }
    buffer_set->memPools[d] = memPool;
  }

  if (hipSetDevice(local_rank) != hipSuccess) {
    printf("(rank %d) Failed to set device\n", local_rank);
    return MUILLM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __init_p2p_recv(
    muillm_comm_p2p_t* comm
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  printf("(rank %d) Initializing peer to peer transfers\n", local_rank);

  comm->recv_buffer_size = 0;

  if (hipSetDevice(local_rank) != hipSuccess) {
    printf("(rank %d) Failed to set device\n", local_rank);
    return MUILLM_UNKNOWN_ERROR;
  }

  // enable peer to peer
  for (int d = 0; d < local_size; d++) {
    if (d == local_rank) continue;
    if (hipDeviceEnablePeerAccess(d, 0) != hipSuccess) {
      // TODO: return error
      printf("(rank %d) Failed to enable peer to peer with %d\n", local_rank, d);
      return MUILLM_UNKNOWN_ERROR;
    }
  }
  printf("(rank %d) Enabled peer to peer\n", local_rank);

  if (__init_p2p_recv_buffer_set(comm, &comm->p2p_recv_buffer_set) != MUILLM_COMM_SUCCESS) {
    return MUILLM_UNKNOWN_ERROR;
  }
  if (__init_p2p_recv_buffer_set(comm, &comm->second_p2p_recv_buffer_set) != MUILLM_COMM_SUCCESS) {
    return MUILLM_UNKNOWN_ERROR;
  }
  return MUILLM_COMM_SUCCESS;
}


muillm_comm_error_t __local_p2p_gpu_barrier(
    muillm_comm_p2p_t* comm,
    hipStream_t stream) {
  int local_rank = comm->local_rank;
  int local_size = comm->local_size;


  muillm_comm_error_t error;

  // record our event
  if (hipEventRecord(comm->wait_event_set.event, stream) != hipSuccess) {
    printf("(rank %d) Recording event failed\n", local_rank);
    return MUILLM_UNKNOWN_ERROR;
  }

  if (hipStreamSynchronize(stream) != hipSuccess) {
    printf("(rank %d) gpu barrier sync error\n", local_rank);
    return MUILLM_UNKNOWN_ERROR;
  }

  if (hipDeviceSynchronize() != hipSuccess) {
    printf("(rank %d) gpu barrier device sync error\n", local_rank);
    return MUILLM_UNKNOWN_ERROR;
  }

  // use the socket to do a CPU barrier
  if ((error = __local_socket_barrier(comm)) != MUILLM_COMM_SUCCESS) {
    return error;
  }

  // wait on the other events
  for (int r = 0; r < local_size; r++) {
    if (r == local_rank) {
      continue;
    }

    hipError_t error;
    if ((error = hipStreamWaitEvent(stream, comm->wait_event_set.all_events[r], 0)) != hipSuccess) {
      printf("(rank %d) Waiting event %d failed error %s\n", local_rank, r, hipGetErrorName(error));
      return MUILLM_UNKNOWN_ERROR;
    }
  }

  if (hipStreamSynchronize(stream) != hipSuccess) {
    printf("(rank %d) gpu barrier sync error\n", local_rank);
    return MUILLM_UNKNOWN_ERROR;
  }

  if (hipDeviceSynchronize() != hipSuccess) {
    printf("(rank %d) gpu barrier device sync error\n", local_rank);
    return MUILLM_UNKNOWN_ERROR;
  }

  // swap the sets of events to avoid livelocks
  // when a GPU already writes the next seq_no while another GPU is waiting
  muillm_comm_event_set_t temp_event_set;
  temp_event_set.event = comm->wait_event_set.event;
  temp_event_set.all_events = comm->wait_event_set.all_events;

  comm->wait_event_set.event = comm->second_wait_event_set.event;
  comm->wait_event_set.all_events = comm->second_wait_event_set.all_events;

  comm->second_wait_event_set.event = temp_event_set.event;
  comm->second_wait_event_set.all_events = temp_event_set.all_events;

  //__local_socket_barrier(comm);

  return MUILLM_COMM_SUCCESS;
}

static void __reallocate_p2p_recv_buffer_set(
    muillm_comm_p2p_t* comm,
    size_t required_recv_buffer_size,
    hipStream_t stream,
    muillm_comm_p2p_recv_buffer_set_t* buffer_set
  ) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  // unsufficient buffer space: we have to allocate some bigger space
  if (buffer_set->p2p_recv_buffer != nullptr) {
    // we need to synchronize the ranks and block the  CPU so that we can deallocate
    // the previous receive buffers
    __local_p2p_gpu_barrier(comm, stream);

    // synchronize to make sure no GPU is going to reference the previous memory
    if (hipDeviceSynchronize() != hipSuccess) {
      printf("(rank %d) Error while synchronizing device\n", local_rank);
      return;
    }

    if (hipFree(buffer_set->p2p_recv_buffer) != hipSuccess) {
      printf("(rank %d) Error while freeing recv_buffers\n", local_rank);
      return;
    }
  }

  // then allocate new receive buffers
  buffer_set->p2p_recv_buffer = nullptr;

  // each GPU will write in a different buffer
  size_t num_recv_buffers = __comm_buffer_set_size(comm);
  // align on a 2MiB as it is the shareable page size
  required_recv_buffer_size = ALIGN_UP(required_recv_buffer_size, GPU_SHAREABLE_PAGE_SIZE);

  size_t all_required_recv_buffer_size = num_recv_buffers * required_recv_buffer_size;

  if (hipSetDevice(local_rank) != hipSuccess) {
    printf("(rank %d) Failed to set device\n", local_rank);
    return;
  }

  // allocate the new memory
  if (hipMalloc(&buffer_set->p2p_recv_buffer, all_required_recv_buffer_size) != 0) {
    // failed
    printf("(rank %d) Failed to allocate new receive buffer of size %zuB\n", local_rank, all_required_recv_buffer_size);
    return;
  }

  // TODO: try device pointer sharing from stream ordered memory allocator
  // https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/stream_ordered_allocator.html


  // exchange pointers to recv buffers
#if 1
  //*
  hipMemPoolPtrExportData memHandle;
  if (hipMemPoolExportPointer(&memHandle, buffer_set->p2p_recv_buffer) != hipSuccess) {
    // failed
    printf("(rank %d) Failed to get memory handle\n", local_rank);
    return;
  }
  
  hipMemPoolPtrExportData* allMemHandles = new hipMemPoolPtrExportData[local_size];

  // gather all memory handles
  __local_socket_all_gather(comm, &memHandle, sizeof(hipMemPoolPtrExportData), allMemHandles);

  // get the remote pointers
  for (int d = 0; d < local_size; d++) {
    if (hipSetDevice(d) != hipSuccess) {
      printf("(rank %d) Failed to set device\n", local_rank);
      return;
    }

    if (d == local_rank) {
      buffer_set->all_p2p_recv_buffers[d] = buffer_set->p2p_recv_buffer;
    } else {
      // need to open the memory handle
      int* recv_buffer = nullptr;
      if (hipMemPoolImportPointer((void**)&buffer_set->all_p2p_recv_buffers[d], buffer_set->memPools[d], &allMemHandles[d]) != hipSuccess) {
        // failed
        printf("(rank %d) Failed to open memory handle %d\n", local_rank, d);
        return;
      }
    }
  }

  if (hipSetDevice(local_rank) != hipSuccess) {
    printf("(rank %d) Failed to set device\n", local_rank);
    return;
  }

  // we don't need this array anymore
  delete[] allMemHandles;
#else
  __local_socket_all_gather(comm, &buffer_set->p2p_recv_buffer, sizeof(void*), buffer_set->all_p2p_recv_buffers);
#endif

  // copy to the device memory all the pointers
  if (hipMemcpyAsync(buffer_set->all_p2p_recv_buffers_device, buffer_set->all_p2p_recv_buffers, local_size * sizeof(void*), hipMemcpyHostToDevice, stream) != hipSuccess) {
    printf("failed to set all buffers device");
    return;
  }

  comm->recv_buffer_size = all_required_recv_buffer_size / local_size;
}


static void __reallocate_p2p_recv_buffer(
    muillm_comm_p2p_t* comm,
    size_t required_recv_buffer_size,
    hipStream_t stream
  ) {
  __reallocate_p2p_recv_buffer_set(comm, required_recv_buffer_size, stream, &comm->p2p_recv_buffer_set);
  __reallocate_p2p_recv_buffer_set(comm, required_recv_buffer_size, stream, &comm->second_p2p_recv_buffer_set);
}


// each threads can copy 16 bytes
#define THREADS_PER_BLOCK 256
#define BYTES_PER_THREAD 16
#define BYTES_PER_BLOCK (THREADS_PER_BLOCK * BYTES_PER_THREAD)

typedef struct uint32x4{
  uint32_t x, y, z, w;
} uint32x4_t;

__global__ void __copy_kernel(
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

static void __gpu_copy(void* dst, const void* src, size_t count, hipStream_t stream) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(count, BYTES_PER_BLOCK);

  // a copy kernel is faster than a hipMemcpyAsync
  __copy_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint8_t*) src,
    (uint8_t*) dst,
    count
  );
}


__global__ void muillm_reduce_sum_p2p_fp32_kernel(
    const float* local_buff,
    const float** remote_buffs,
    float* dest_buff,
    unsigned local_size,
    unsigned local_rank,
    unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float r = local_buff[i];

    // data to reduce is packed in the receive buffer
    // TODO: vectorize, unroll
    for (unsigned b = 0; b < local_size; b++) {
      if (b == local_rank) continue;
      const float* remote_buff = (const float*)remote_buffs[b];

      r += remote_buff[i];
    }

    dest_buff[i] = r;
  }
}

__global__ void muillm_reduce_sum_p2p_fp16_kernel(
    const half* local_buff,
    const half** remote_buffs,
    half* dest_buff,
    unsigned local_size,
    unsigned local_rank,
    unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float r = __half2float(local_buff[i]);

    // data to reduce is packed in the receive buffer
    // TODO: vectorize, unroll
    for (unsigned b = 0; b < local_size; b++) {
      if (b == local_rank) continue;
      const half* remote_buff = (const half*)remote_buffs[b];

      r += __half2float(remote_buff[i]);
    }

    dest_buff[i] = __float2half(r);
  }
}


__global__ void muillm_reduce_sum_p2p_fp16_tp4_kernel(
    const half* local_buff,
    const half* remote_buff0,
    const half* remote_buff1,
    const half* remote_buff2,
    const half* remote_buff3,
    half* dest_buff,
    unsigned local_size,
    unsigned local_rank,
    unsigned N
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float r = __half2float(local_buff[i]);

    // data to reduce is packed in the receive buffer
    // TODO: vectorize, unroll
    if (0 != local_rank) {
      r += __half2float(remote_buff0[i]);
    }
    if (1 != local_rank) {
      r += __half2float(remote_buff1[i]);
    }
    if (2 != local_rank) {
      r += __half2float(remote_buff2[i]);
    }
    if (3 != local_rank) {
      r += __half2float(remote_buff3[i]);
    }
    dest_buff[i] = __float2half(r);
  }
}

muillm_comm_error_t __all_reduce_sum_p2p(
    muillm_comm_p2p_t* comm,
    void* src_ptr,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  muillm_comm_error_t error;

  // printf("(rank %d) all reduce %d (count %zu datatype %d)\n", local_rank, comm->all_reduce_no, count, datatype);
  // fflush(stdout);

  // first, make sure we have enough buffer space
  __ensure_p2p_buffer_capacity(comm, count, datatype, stream);

  size_t recv_buffer_size = __comm_size(datatype, count);

  // copy the src in our buffer
  //if (comm->all_reduce_no != 98) {
  {
  __gpu_copy(comm->p2p_recv_buffer_set.p2p_recv_buffer, src_ptr, recv_buffer_size, stream);
  }
  // if (true) {//if (comm->all_reduce_no % 1 == 0) {
  //   if (hipStreamSynchronize(stream) != hipSuccess) {
  //     printf("(rank %d) all reduce drain sync error\n", local_rank);
  //     return MUILLM_UNKNOWN_ERROR;
  //   }
  // }

  // sync the GPUs
  if ((error = __local_p2p_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    printf("(rank %d) all reduce sync error\n", local_rank);
    return error;
  }

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(count, threads_per_blocks);

  // do the reductions
  //if (comm->all_reduce_no != 98) {
  {
    if (datatype == MUILLM_COMM_FP16) {
      if (local_size == 4) {
        muillm_reduce_sum_p2p_fp16_tp4_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
          (const half*)src_ptr,
          (const half*)comm->p2p_recv_buffer_set.all_p2p_recv_buffers[0],
          (const half*)comm->p2p_recv_buffer_set.all_p2p_recv_buffers[1],
          (const half*)comm->p2p_recv_buffer_set.all_p2p_recv_buffers[2],
          (const half*)comm->p2p_recv_buffer_set.all_p2p_recv_buffers[3],
          (half*)dst_ptr,
          local_size,
          local_rank,
          count
        );
      } else {
        muillm_reduce_sum_p2p_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
          (const half*)src_ptr,
          (const half**)comm->p2p_recv_buffer_set.all_p2p_recv_buffers_device,
          (half*)dst_ptr,
          local_size,
          local_rank,
          count
        );
      }
    } else if (datatype == MUILLM_COMM_FP32) {
      muillm_reduce_sum_p2p_fp32_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
        (const float*)src_ptr,
        (const float**)comm->p2p_recv_buffer_set.all_p2p_recv_buffers_device,
        (float*)dst_ptr,
        local_size,
        local_rank,
        count
      );
    } else {
      // TODO: error
      printf("unsupported type\n");
    }
  }

  // if (true) {//(comm->all_reduce_no % 1 == 0) {
  //   if (hipStreamSynchronize(stream) != hipSuccess) {
  //     printf("(rank %d) all reduce drain sync error\n", local_rank);
  //     return MUILLM_UNKNOWN_ERROR;
  //   }
  // }

  // sync the GPUs
  // sync the GPUs
  // if ((error = __local_p2p_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
  //   printf("(rank %d) all reduce sync 2 error\n", local_rank);
  //   return MUILLM_UNKNOWN_ERROR;
  // }

  // swap buffer sets to avoid overwrites
  __swap_p2p_recv_buffer_sets(comm);


  // printf("(rank %d) finished all reduce %d\n", local_rank, comm->all_reduce_no);
  // fflush(stdout);


  comm->all_reduce_no++;

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t __broadcast_p2p(
    muillm_comm_p2p_t* comm,
    int src_rank,
    void* ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  muillm_comm_error_t error;

  //printf("(rank %d) broadcast\n", local_rank);

  // first, make sure we have enough buffer space
  __ensure_p2p_buffer_capacity(comm, count, datatype, stream);

  size_t recv_buffer_size = __comm_size(datatype, count);

  if (src_rank == local_rank) {
    // send if we are the broadcaster the pieces

    // copy the src in our buffer
    __gpu_copy(comm->p2p_recv_buffer_set.p2p_recv_buffer, ptr, recv_buffer_size, stream);
  }

  // if (true) {//if (comm->all_reduce_no % 1 == 0) {
  //   if (hipStreamSynchronize(stream) != hipSuccess) {
  //     printf("(rank %d) broadcast drain sync error\n", local_rank);
  //     return MUILLM_UNKNOWN_ERROR;
  //   }
  // }

  // sync the GPUs
  // sync the GPUs
  if ((error = __local_p2p_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
    printf("(rank %d) broadcast sync error\n", local_rank);
    return error;
  }

  // copy back
  if (src_rank != local_rank) {
    __gpu_copy(ptr, comm->p2p_recv_buffer_set.all_p2p_recv_buffers[src_rank], recv_buffer_size, stream);
  }

  // if (true) {//if (comm->all_reduce_no % 1 == 0) {
  //   if (hipStreamSynchronize(stream) != hipSuccess) {
  //     printf("(rank %d) broadcast drain sync 2 error\n", local_rank);
  //     return MUILLM_UNKNOWN_ERROR;
  //   }
  // }

  // if ((error = __local_p2p_gpu_barrier(comm, stream)) != MUILLM_COMM_SUCCESS) {
  //   printf("(rank %d) broadcast sync 2 error\n", local_rank);
  //   return error;
  // }

  // swap buffer sets to avoid overwrites
  __swap_p2p_recv_buffer_sets(comm);

  return MUILLM_COMM_SUCCESS;
}