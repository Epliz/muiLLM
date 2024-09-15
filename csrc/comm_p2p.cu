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
  hipMemPool_t* memPools_temp = comm->p2p_recv_buffer_set.memPools;

  comm->p2p_recv_buffer_set.p2p_recv_buffer = comm->second_p2p_recv_buffer_set.p2p_recv_buffer;
  comm->p2p_recv_buffer_set.all_p2p_recv_buffers = comm->second_p2p_recv_buffer_set.all_p2p_recv_buffers;
  comm->p2p_recv_buffer_set.memPools = comm->second_p2p_recv_buffer_set.memPools;

  comm->second_p2p_recv_buffer_set.p2p_recv_buffer = p2p_recv_buffer_temp;
  comm->second_p2p_recv_buffer_set.all_p2p_recv_buffers = all_p2p_recv_buffers_temp;
  comm->second_p2p_recv_buffer_set.memPools = memPools_temp;
}

// returns the number of receive buffers needed to do any of the operations
static inline size_t __comm_buffer_set_size(
    muillm_comm_p2p_t* comm
) {
  // we need as many buffers as local ranks
  return comm->local_size;
}

static void __allocate_wait_event_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_event_set_t* event_set
) {
  int local_rank = comm->local_rank;
  int local_size = comm->local_size;

  event_set->all_events = new hipEvent_t[local_size];

  // allocate local event
  if (hipEventCreateWithFlags(&event_set->event, hipEventDisableTiming | hipEventInterprocess) != hipSuccess) {
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

  buffer_set->p2p_recv_buffer = nullptr;
  buffer_set->all_p2p_recv_buffers = new void*[local_size];

  for (int d = 0; d < local_size; d++) {
    buffer_set->all_p2p_recv_buffers[d] = nullptr;
  }

  //create mem pools
  buffer_set->memPools = new hipMemPool_t[local_size];
  for (int d = 0; d < local_size; d++) {
    if (d == local_rank) continue;

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

  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __init_p2p_recv(
    muillm_comm_p2p_t* comm
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  printf("(rank %d) Initializing peer to peer transfers\n", local_rank);

  comm->recv_buffer_size = 0;

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


void __local_p2p_gpu_barrier(
    muillm_comm_p2p_t* comm,
    hipStream_t stream) {
  int local_rank = comm->local_rank;
  int local_size = comm->local_size;

  // record our event
  if (hipEventRecord(comm->wait_event_set.event, stream) != hipSuccess) {
    printf("(rank %d) Recording event failed\n", local_rank);
    return;
  }

  // use the socket to do a CPU barrier
  __local_socket_barrier(comm);

  // wait on the other events
  for (int r = 0; r < local_size; r++) {
    if (r == local_rank) {
      continue;
    }

    hipError_t error;
    if ((error = hipStreamWaitEvent(stream, comm->wait_event_set.all_events[r], 0)) != hipSuccess) {
      printf("(rank %d) Waiting event %d failed error %s\n", local_rank, r, hipGetErrorName(error));
      return;
    }
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

  // we don't need this array anymore
  delete[] allMemHandles;

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


#define THREADS_PER_BLOCK 256

__global__ void muillm_reduce_sum_p2p_fp32_kernel(
    const float* local_buff,
    const float* remote_buffs,
    float* dest_buff,
    unsigned local_size,
    unsigned N,
    unsigned padded_recv_buffer_size
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float r = local_buff[i];

    // data to reduce is packed in the receive buffer
    // TODO: vectorize, unroll
    for (unsigned b = 0; b < local_size - 1; b++) {
      size_t offset = padded_recv_buffer_size * b;
      const float* remote_buff = (const float*)((const uint8_t*)remote_buffs + offset);

      r += remote_buff[i];
    }

    dest_buff[i] = r;
  }
}

__global__ void muillm_reduce_sum_p2p_fp16_kernel(
    const half* local_buff,
    const half* remote_buffs,
    half* dest_buff,
    unsigned local_size,
    unsigned N,
    unsigned padded_recv_buffer_size
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float r = __half2float(local_buff[i]);

    // data to reduce is packed in the receive buffer
    // TODO: vectorize, unroll
    for (unsigned b = 0; b < local_size - 1; b++) {
      size_t offset = padded_recv_buffer_size * b;
      const half* remote_buff = (const half*)((const uint8_t*)remote_buffs + offset);

      r += __half2float(remote_buff[i]);
    }

    dest_buff[i] = __float2half(r);
  }
}

static inline void __local_gpu_send_all_async_p2p_copy(
    muillm_comm_p2p_t* comm,
    void* src_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  // copy the memories around
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  size_t recv_buffer_size = __comm_size(datatype, count);
  // align to avoid cache line crossing
  // (might avoid correctness issues due to crossing PCIe writes)
  size_t padded_recv_buffer_size = ALIGN_UP(recv_buffer_size, GPU_CACHELINE_SIZE);

  // copy our memory to all remote GPUs
  // (we pack all rank data almost contiguously to simplify the reduce kernel)
  for (int r = 0; r < local_rank; r++) {
    // we have to write at this place:
    size_t dest_offset = padded_recv_buffer_size * (local_rank - 1);
    uint8_t* dst_ptr = ((uint8_t*)comm->p2p_recv_buffer_set.all_p2p_recv_buffers[r]) + dest_offset;

    if (hipMemcpyPeerAsync(dst_ptr, r, src_ptr, local_rank, recv_buffer_size, stream) != hipSuccess) {
      printf("(rank %d) Memcpy to %d failed\n", local_rank, r);
    }
  }
  for (int r = local_rank + 1; r < local_size; r++) {
    // we have to write at this place:
    size_t dest_offset = padded_recv_buffer_size * local_rank;
    uint8_t* dst_ptr = ((uint8_t*)comm->p2p_recv_buffer_set.all_p2p_recv_buffers[r]) + dest_offset;

    if (hipMemcpyPeerAsync(dst_ptr, r, src_ptr, local_rank, recv_buffer_size, stream) != hipSuccess) {
      printf("(rank %d) Memcpy to %d failed\n", local_rank, r);
    }
  }
}

void __all_reduce_sum_p2p(
    muillm_comm_p2p_t* comm,
    void* src_ptr,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  // first, make sure we have enough buffer space
  __ensure_p2p_buffer_capacity(comm, count, datatype, stream);

  // gather all the pieces
  __local_gpu_send_all_async_p2p_copy(
    comm,
    src_ptr,
    count,
    datatype,
    stream
  );

  // sync the GPUs
  __local_p2p_gpu_barrier(comm, stream);

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(count, threads_per_blocks);

  size_t recv_buffer_size = __comm_size(datatype, count);
  // align to avoid cache line crossing
  // (might avoid correctness issues due to crossing PCIe writes)
  size_t padded_recv_buffer_size = ALIGN_UP(recv_buffer_size, GPU_CACHELINE_SIZE);

  // do the reductions
  if (datatype == MUILLM_COMM_FP16) {
    muillm_reduce_sum_p2p_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      (const half*)src_ptr,
      (const half*)comm->p2p_recv_buffer_set.p2p_recv_buffer,
      (half*)dst_ptr,
      local_size,
      count,
      padded_recv_buffer_size
    );
  } else if (datatype == MUILLM_COMM_FP32) {
    muillm_reduce_sum_p2p_fp32_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      (const float*)src_ptr,
      (const float*)comm->p2p_recv_buffer_set.p2p_recv_buffer,
      (float*)dst_ptr,
      local_size,
      count,
      padded_recv_buffer_size
    );
  } else {
    // TODO: error
    printf("unsupported type\n");
  }

  // swap buffer sets to avoid overwrites
  __swap_p2p_recv_buffer_sets(comm);
}

static inline void __local_gpu_receive_async_p2p_copy(
    muillm_comm_p2p_t* comm,
    int src_rank,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  // copy the memories around
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  size_t recv_buffer_size = __comm_size(datatype, count);
  // align to avoid cache line crossing
  // (might avoid correctness issues due to crossing PCIe writes)
  size_t padded_recv_buffer_size = ALIGN_UP(recv_buffer_size, GPU_CACHELINE_SIZE);

  // (we pack all rank data almost contiguously to simplify the reduce kernel)
  size_t buff_idx = (src_rank < local_rank) ? src_rank : (src_rank - 1);

  size_t src_offset = padded_recv_buffer_size * buff_idx;
  
  // we have to read at this place:
  uint8_t* src_ptr = ((uint8_t*)comm->p2p_recv_buffer_set.p2p_recv_buffer) + src_offset;

  // dst and src are on the local rank
  if (hipMemcpyAsync(dst_ptr, src_ptr, recv_buffer_size, hipMemcpyDeviceToDevice, stream) != hipSuccess) {
    printf("(rank %d) Memcpy from %d recv buffer failed\n", local_rank, src_rank);
  }
}

void __broadcast_p2p(
    muillm_comm_p2p_t* comm,
    int src_rank,
    void* ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  // first, make sure we have enough buffer space
  __ensure_p2p_buffer_capacity(comm, count, datatype, stream);

  //if (src_rank == local_rank) {
    // send if we are the broadcaster the pieces
    __local_gpu_send_all_async_p2p_copy(
      comm,
      ptr,
      count,
      datatype,
      stream
    );
  //}

  // if (hipStreamSynchronize(stream) != hipSuccess) {
  //   printf("(rank %d) after send async \n");
  //   return;
  // }

  // sync the GPUs
  __local_p2p_gpu_barrier(comm, stream);

  // if (hipStreamSynchronize(stream) != hipSuccess) {
  //   printf("(rank %d) after barrier \n");
  //   return;
  // }

  if (src_rank != local_rank) {
    // receive if we are not the broadcaster
    __local_gpu_receive_async_p2p_copy(
      comm,
      src_rank,
      ptr,
      count,
      datatype,
      stream
    );
  }

  // if (hipStreamSynchronize(stream) != hipSuccess) {
  //   printf("(rank %d) after receive \n");
  //   return;
  // }

  // swap buffer sets to avoid overwrites
  __swap_p2p_recv_buffer_sets(comm);
}