#include "comm.h"

#include <iostream>

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define MUILLM_COMM_INITIAL_BUFFER_CAPACITY (1024 * 1024) // 1MiB

size_t __next_power_of_2(size_t n) {
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

muillm_comm_error_t __ensure_buffer_set_capacity(muillm_comm_buffer_set_t* buffer_set, size_t capacity, int local_size) {
  if (capacity <= buffer_set->capacity) {
    // the buffers are big enough
    return MUILLM_COMM_SUCCESS;
  }

  int default_device_id;
  if (hipGetDevice(&default_device_id) != hipSuccess) {
    std::cout<<"Error getting the default device"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // if needed deallocate previous memory
  for (int i = 0; i < MUILLM_COMM_MAX_GPUS; i++) {
    void* ptr = buffer_set->buffers[i];
    if (ptr != nullptr) {
      if (hipFree(ptr) != hipSuccess) {
        std::cout<<"Error free buffer "<<i<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }

      buffer_set->buffers[i] = nullptr;
    }
  }

  // allocate new buffers
  capacity = __next_power_of_2(capacity);

  for (int i = 0; i < local_size; i++) {
    if (hipSetDevice(i) != hipSuccess) {
      std::cout<<"Error setting the device before allocation"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }

    void* ptr = nullptr;
    if (hipMalloc((void**)&ptr, capacity) != hipSuccess || ptr == nullptr) {
      std::cout<<"Allocation of buffer "<<i<<" failed"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
    
    buffer_set->buffers[i] = ptr;
  }

  // all buffer allocations suceeded
  buffer_set->capacity = capacity;

  // set back the correct default device
  if (hipSetDevice(default_device_id) != hipSuccess) {
    std::cout<<"Error setting back the default device"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}


muillm_comm_error_t muillm_comm_get_buffer_set(muillm_comm_t* comm, size_t count, muillm_comm_datatype_t datatype, muillm_comm_buffer_set_t** buffer_set) {
  
  muillm_comm_error_t muillm_error;

  size_t capacity = __comm_size(datatype, count);
  if ((muillm_error = __ensure_buffer_set_capacity(comm->first_buffers, capacity, comm->local_size)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  // always return the current first buffer set
  *buffer_set = comm->first_buffers;

  // swap buffer sets for next time
  muillm_comm_buffer_set_t* tmp = comm->first_buffers;
  comm->first_buffers = comm->second_buffers;
  comm->second_buffers = tmp;

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t __init_buffer_set(int local_size, muillm_comm_buffer_set_t** buffer_set_ptr) {
  muillm_comm_buffer_set_t* buffer_set = new muillm_comm_buffer_set_t;
  buffer_set->capacity = 0;

  if (buffer_set == nullptr) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  for (int i = 0; i < MUILLM_COMM_MAX_GPUS; i++) {
    buffer_set->buffers[i] = nullptr;
  }

  // ensure a certain good initial size
  muillm_comm_error_t muillm_error;
  if ((muillm_error = __ensure_buffer_set_capacity(buffer_set, MUILLM_COMM_INITIAL_BUFFER_CAPACITY, local_size)) != MUILLM_COMM_SUCCESS) {
    *buffer_set_ptr = nullptr;
    return muillm_error;
  }

  *buffer_set_ptr = buffer_set;
  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t muillm_comm_init(
  int local_size,
  bool allocate_streams,
  muillm_comm_t** comm_ptr
) {

  hipError_t error;
  muillm_comm_error_t muillm_error;

  int device_count;
  if (hipGetDeviceCount(&device_count) != hipSuccess) {
    std::cout<<"Failed to get the numnber of GPUs"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  if (device_count < local_size) {
    // unsifficient devices
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  muillm_comm_t* comm = new muillm_comm_t;
  comm->local_size = local_size;
  comm->streams = new hipStream_t[local_size];
  comm->acquire_events = new hipEvent_t[local_size];
  comm->release_events = new hipEvent_t[local_size];
  comm->signals = new uint64_t*[local_size];
  comm->signal_seq_no = 0;

  int default_device_id;
  if (hipGetDevice(&default_device_id) != hipSuccess) {
    std::cout<<"Error getting the default device"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // enable peer to peer
    for (int r = 0; r < local_size; r++) {
    // create a stream
    if (hipSetDevice(r) != hipSuccess) {
      std::cout<<"(rank "<<r<<") Error setting the device"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }

    // enable peer to peer
    for (int other_rank = 0; other_rank < device_count; other_rank++) {
      if (other_rank == r) continue;
  
      if (hipDeviceEnablePeerAccess(other_rank, 0) != hipSuccess) {
        // TODO: return error
        std::cout<<"(rank "<<r<<") Failed to enable peer to peer with "<<other_rank<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }
  }

  // create streams and events
  for (int r = 0; r < local_size; r++) {
    // create a stream
    if (hipSetDevice(r) != hipSuccess) {
      std::cout<<"(rank "<<r<<") Error setting the device"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }

    if (allocate_streams) {
      if ((error = hipStreamCreateWithFlags(&comm->streams[r], hipStreamNonBlocking)) != hipSuccess) {
        std::cout<<"(rank "<<r<<") Error creating the stream "<<hipGetErrorName(error)<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }

    if (hipEventCreateWithFlags(&comm->acquire_events[r], hipEventDisableTiming | hipEventReleaseToSystem) != hipSuccess) {
      std::cout<<"event creation failed\n"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
    // is it correct to disable the system fence?
    if (hipEventCreateWithFlags(&comm->release_events[r], hipEventDisableTiming | hipEventReleaseToSystem) != hipSuccess) {
      std::cout<<"event creation failed\n"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }

  int signals_supported;
  if (hipDeviceGetAttribute(&signals_supported, hipDeviceAttributeCanUseStreamWaitValue, 0) != hipSuccess) {
    std::cout<<"Error getting the the property"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  if (signals_supported) {
    std::cout<<"wait stream value supported"<<std::endl;

    // need to allocate 8 bytes
    for (int r = 0; r < local_size; r++) {
      if (hipSetDevice(r) != hipSuccess) {
        std::cout<<"(rank "<<r<<") Error setting the device"<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
      if ((error = hipExtMallocWithFlags((void**) &comm->signals[r], sizeof(uint64_t), hipMallocSignalMemory)) == hipSuccess) {
        *comm->signals[r] = 0;
      } else {
        delete[] comm->signals;
        comm->signals = nullptr;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }
  } else {
    delete[] comm->signals;
    comm->signals = nullptr;
  }

  // initialize buffer sets
  if ((muillm_error = __init_buffer_set(local_size, &comm->first_buffers)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }
  if ((muillm_error = __init_buffer_set(local_size, &comm->second_buffers)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  // set back the correct default device
  if (hipSetDevice(default_device_id) != hipSuccess) {
    std::cout<<"Error setting back the default device"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  *comm_ptr = comm;
  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __mui_gpu_barrier(muillm_comm_t* comm) {
  int local_size = comm->local_size;
  hipError_t hip_error;
  if (comm->signals != nullptr) {
    comm->signal_seq_no++;
    uint64_t seq_no = comm->signal_seq_no;

    // GPU barrier: all GPUs wait on each other
    for (int r = 0; r < local_size; r++) {
      if ((hip_error = hipStreamWriteValue32(comm->streams[r], (uint32_t*)comm->signals[r], seq_no, 0)) != hipSuccess) {
        std::cout<<"Failed to write value "<<r<<std::endl;
        std::cout<<"error: "<<hipGetErrorName(hip_error)<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }

      // wait for the other ranks
      for (int other_rank = 0; other_rank < local_size; other_rank++) {
        if (other_rank == r) continue;
        if ((hip_error = hipStreamWaitValue32(comm->streams[r], (uint32_t*)comm->signals[other_rank], seq_no, hipStreamWaitValueEq, -1)) != hipSuccess) {
          std::cout<<"Failed to wait for value"<<std::endl;
          std::cout<<"error: "<<hipGetErrorName(hip_error)<<std::endl;
          return MUILLM_COMM_UNKNOWN_ERROR;
        }
      }
    }
  } else {
    // GPU barrier: all GPUs wait on each other
    for (int r = 0; r < local_size; r++) {
      if (hipEventRecord(comm->acquire_events[r], comm->streams[r]) != hipSuccess) {
        std::cout<<"Failed to record event "<<r<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }

    // wait for the other ranks
    for (int r = 0; r < local_size; r++) {
      for (int other_rank = 0; other_rank < local_size; other_rank++) {
        if (other_rank == r) continue;

        hipError_t error;
        if ((error = hipStreamWaitEvent(comm->streams[r], comm->acquire_events[other_rank], 0)) != hipSuccess) {
          std::cout<<"Failed to wait for event"<<std::endl;
          std::cout<<"error: "<<hipGetErrorName(hip_error)<<std::endl;
          return MUILLM_COMM_UNKNOWN_ERROR;
        }
      }
    }
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

__global__ void __muillm_copy_kernel(
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

static void __muillm_gpu_copy(void* dst, const void* src, size_t count, hipStream_t stream) {
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(count, BYTES_PER_BLOCK);

  // a copy kernel is faster than a hipMemcpyAsync
  __muillm_copy_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint8_t*) src,
    (uint8_t*) dst,
    count
  );
}

// TP2 kernels

__global__ void __all_reduce_fp16_tp2_multi_write_out_kernel(
    const half* x1,
    const half* x2,
    half* y1,
    half* y2,
    unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    half res = __hadd(x1[i], x2[i]);
    y1[i] = res;
    y2[i] = res;
  }
}

__global__ void __all_reduce_fp32_tp2_multi_write_out_kernel(
    const float* x1,
    const float* x2,
    float* y1,
    float* y2,
    unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float res = x1[i] + x2[i];
    y1[i] = res;
    y2[i] = res;
  }
}

// TP4 kernels

__global__ void __all_reduce_fp16_tp4_multi_write_out_kernel(
    const half* x1,
    const half* x2,
    const half* x3,
    const half* x4,
    half* y1,
    half* y2,
    half* y3,
    half* y4,
    unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    half res = __hadd(__hadd(x1[i], x2[i]), __hadd(x3[i], x4[i]));
    y1[i] = res;
    y2[i] = res;
    y3[i] = res;
    y4[i] = res;
  }
}

__global__ void __all_reduce_fp32_tp4_multi_write_out_kernel(
    const float* x1,
    const float* x2,
    const float* x3,
    const float* x4,
    float* y1,
    float* y2,
    float* y3,
    float* y4,
    unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float res = x1[i] + x2[i] + x3[i] + x4[i];
    y1[i] = res;
    y2[i] = res;
    y3[i] = res;
    y4[i] = res;
  }
}


muillm_comm_error_t muillm_comm_all_reduce_sum_single_gpu(
  muillm_comm_t* comm,
  const void** src_ptrs,
  void** dst_ptrs,
  size_t count,
  muillm_comm_datatype_t datatype
) {
  hipError_t hip_error;

  // an approach doing
  // 1) barrier
  // 2) reduce on GPU0
  // 3) barrier
  int local_size = comm->local_size;

  if (comm->signals != nullptr) {
    comm->signal_seq_no++;
    uint64_t seq_no = comm->signal_seq_no;

    // GPU barrier: GPU0 will wait on the others
    for (int r = 1; r < local_size; r++) {
      if ((hip_error = hipStreamWriteValue32(comm->streams[r], (uint32_t*)comm->signals[r], seq_no, 0)) != hipSuccess) {
        std::cout<<"Failed to write value "<<r<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }

    // wait for the other ranks
    for (int other_rank = 1; other_rank < local_size; other_rank++) {
      if ((hip_error = hipStreamWaitValue32(comm->streams[0], (uint32_t*)comm->signals[other_rank], seq_no, hipStreamWaitValueEq, -1)) != hipSuccess) {
        std::cout<<"Failed to wait for value"<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }
  } else {
    // GPU barrier: GPU0 will wait on the others
    for (int r = 1; r < local_size; r++) {
      if (hipEventRecord(comm->acquire_events[r], comm->streams[r]) != hipSuccess) {
        std::cout<<"Failed to record event "<<r<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }

    // wait for the other ranks
    for (int other_rank = 1; other_rank < local_size; other_rank++) {
      if ((hip_error = hipStreamWaitEvent(comm->streams[0], comm->acquire_events[other_rank], 0)) != hipSuccess) {
        std::cout<<"Failed to wait for event"<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }
  }

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(count, THREADS_PER_BLOCK);

/*
On MI300x :
  4 GPUs:

  2 GPUs:

On MI100:
  2 GPUs:
  426843us -> 26us/reduce
*/

  // reduce on one GPU
  if (datatype == MUILLM_COMM_FP16) {
    if (local_size == 4) {
      __all_reduce_fp16_tp4_multi_write_out_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[0]>>>(
        (const half*) src_ptrs[0],
        (const half*) src_ptrs[1],
        (const half*) src_ptrs[2],
        (const half*) src_ptrs[3],
        (half*) dst_ptrs[0],
        (half*) dst_ptrs[1],
        (half*) dst_ptrs[2],
        (half*) dst_ptrs[3],
        count
      );
    } else if (local_size == 2) {
      __all_reduce_fp16_tp2_multi_write_out_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[0]>>>(
        (const half*) src_ptrs[0],
        (const half*) src_ptrs[1],
        (half*) dst_ptrs[0],
        (half*) dst_ptrs[1],
        count
      );
    } else {
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else if (datatype == MUILLM_COMM_FP32) {
    if (local_size == 4) {
      __all_reduce_fp32_tp4_multi_write_out_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[0]>>>(
        (const float*) src_ptrs[0],
        (const float*) src_ptrs[1],
        (const float*) src_ptrs[2],
        (const float*) src_ptrs[3],
        (float*) dst_ptrs[0],
        (float*) dst_ptrs[1],
        (float*) dst_ptrs[2],
        (float*) dst_ptrs[3],
        count
      );
    } else if (local_size == 2) {
      __all_reduce_fp32_tp2_multi_write_out_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[0]>>>(
        (const float*) src_ptrs[0],
        (const float*) src_ptrs[1],
        (float*) dst_ptrs[0],
        (float*) dst_ptrs[1],
        count
      );
    } else {
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // make other GPUs wait for the reduction to be done on GPU0
  if (comm->signals != nullptr) {
    comm->signal_seq_no++;
    uint64_t seq_no = comm->signal_seq_no;

    if (hipStreamWriteValue32(comm->streams[0], (uint32_t*)comm->signals[0], seq_no, 0) != hipSuccess) {
      std::cout<<"Failed to write value 0"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }

    // wait for the other ranks
    for (int other_rank = 1; other_rank < local_size; other_rank++) {
      if ((hip_error = hipStreamWaitValue32(comm->streams[other_rank], (uint32_t*)comm->signals[0], seq_no, hipStreamWaitValueEq, -1)) != hipSuccess) {
        std::cout<<"Failed to wait for value"<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }
  } else {
    if (hipEventRecord(comm->release_events[0], comm->streams[0]) != hipSuccess) {
      std::cout<<"Failed to record event for 0"<<std::endl;
      return MUILLM_COMM_UNKNOWN_ERROR;
    }

    for (int other_rank = 1; other_rank < local_size; other_rank++) {
      if ((hip_error = hipStreamWaitEvent(comm->streams[other_rank], comm->release_events[0], 0)) != hipSuccess) {
        std::cout<<"Failed to wait for event"<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }
  }

  return MUILLM_COMM_SUCCESS;
}


// TP2 kernels

__global__ void __all_reduce_fp16_tp2_kernel(
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

__global__ void __all_reduce_fp32_tp2_kernel(
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

__global__ void __all_reduce_fp16_tp4_kernel(
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

__global__ void __all_reduce_fp32_tp4_kernel(
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


muillm_comm_error_t muillm_comm_all_reduce_sum(
  muillm_comm_t* comm,
  const void** src_ptrs,
  void** dst_ptrs,
  size_t count,
  muillm_comm_datatype_t datatype
) {
  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  int local_size = comm->local_size;

  // we only do the copy in the reduction buffers only if there is aliasing
  bool aliasing = false;

  for (int r = 0; r < local_size; r++) {
    if (src_ptrs[r] == dst_ptrs[r]) {
      aliasing = true;
      break;
    }
  }

  // get reduction buffer set
  muillm_comm_buffer_set_t* buffer_set = nullptr;

  if ((muillm_error = muillm_comm_get_buffer_set(comm, count, datatype, &buffer_set)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"Reduction failed when ensuring capacity"<<std::endl;
    return muillm_error;
  }

  // copy into reduction buffers if needed
  if (aliasing) {
    size_t byte_count = __comm_size(datatype, count);
    for (int r = 0; r < local_size; r++) {

      __muillm_gpu_copy(buffer_set->buffers[r], src_ptrs[r], byte_count, comm->streams[r]);
    }
  }

  // ensure all GPUs have copied into the reduction buffers
  if ((muillm_error = __mui_gpu_barrier(comm)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  // do the reduction

/*
On MI300x :
  4 GPUs:

  2 GPUs:

On MI100:
  2 GPUs:
  426843us -> 26us/reduce
*/

  // reduce on one GPU
  if (datatype == MUILLM_COMM_FP16) {
    if (local_size == 4) {
      const int threads_per_blocks = THREADS_PER_BLOCK;
      const int num_blocks = DIV_ROUND_UP(count, THREADS_PER_BLOCK);
      for (int r = 0; r < local_size; r++) {
        __all_reduce_fp16_tp4_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[r]>>>(
          (const half*) (aliasing ? buffer_set->buffers[0] : src_ptrs[0]),
          (const half*) (aliasing ? buffer_set->buffers[1] : src_ptrs[1]),
          (const half*) (aliasing ? buffer_set->buffers[2] : src_ptrs[2]),
          (const half*) (aliasing ? buffer_set->buffers[3] : src_ptrs[3]),
          (half*) dst_ptrs[r],
          count
        );
      }
    } else if (local_size == 2) {
      const int threads_per_blocks = THREADS_PER_BLOCK;
      const int num_blocks = DIV_ROUND_UP(count, THREADS_PER_BLOCK);
      for (int r = 0; r < local_size; r++) {
        __all_reduce_fp16_tp2_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[r]>>>(
          (const half*) (aliasing ? buffer_set->buffers[0] : src_ptrs[0]),
          (const half*) (aliasing ? buffer_set->buffers[1] : src_ptrs[1]),
          (half*) dst_ptrs[r],
          count
        );
      }
    } else {
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else if (datatype == MUILLM_COMM_FP32) {
    if (local_size == 4) {
      const int threads_per_blocks = THREADS_PER_BLOCK;
      const int num_blocks = DIV_ROUND_UP(count, THREADS_PER_BLOCK);
      for (int r = 0; r < local_size; r++) {
        __all_reduce_fp32_tp4_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[r]>>>(
          (const float*) (aliasing ? buffer_set->buffers[0] : src_ptrs[0]),
          (const float*) (aliasing ? buffer_set->buffers[1] : src_ptrs[1]),
          (const float*) (aliasing ? buffer_set->buffers[2] : src_ptrs[2]),
          (const float*) (aliasing ? buffer_set->buffers[3] : src_ptrs[3]),
          (float*) dst_ptrs[r],
          count
        );
      }
    } else if (local_size == 2) {
      const int threads_per_blocks = THREADS_PER_BLOCK;
      const int num_blocks = DIV_ROUND_UP(count, THREADS_PER_BLOCK);

      for (int r = 0; r < local_size; r++) {
        __all_reduce_fp32_tp2_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[r]>>>(
          (const float*) (aliasing ? buffer_set->buffers[0] : src_ptrs[0]),
          (const float*) (aliasing ? buffer_set->buffers[1] : src_ptrs[1]),
          (float*) dst_ptrs[r],
          count
        );
      }
    } else {
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}



// TP2 kernels

__global__ void __all_gather_fp16_tp2_kernel(
    const half* x1,
    const half* x2,
    half* y,
    unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    half* y0 = y;
    half* y1 = y0 + N;

    y0[i] = x1[i];
    y1[i] = x2[i];
  }
}

__global__ void __all_gather_fp32_tp2_kernel(
    const float* x1,
    const float* x2,
    float* y,
    unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float* y0 = y;
    float* y1 = y0 + N;

    y0[i] = x1[i];
    y1[i] = x2[i];
  }
}

// TP4 kernels

__global__ void __all_gather_fp16_tp4_kernel(
    const half* x1,
    const half* x2,
    const half* x3,
    const half* x4,
    half* y,
    unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    half* y0 = y;
    half* y1 = y0 + N;
    half* y2 = y1 + N;
    half* y3 = y2 + N;

    y0[i] = x1[i];
    y1[i] = x2[i];
    y2[i] = x3[i];
    y3[i] = x4[i];
  }
}

__global__ void __all_gather_fp32_tp4_kernel(
    const float* x1,
    const float* x2,
    const float* x3,
    const float* x4,
    float* y,
    unsigned N
) {
  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float* y0 = y;
    float* y1 = y0 + N;
    float* y2 = y1 + N;
    float* y3 = y2 + N;

    y0[i] = x1[i];
    y1[i] = x2[i];
    y2[i] = x3[i];
    y3[i] = x4[i];
  }
}

muillm_comm_error_t muillm_comm_all_gather(
  muillm_comm_t* comm,
  const void** src_ptrs,
  size_t in_count,
  void** dst_ptrs,
  size_t dst_count,
  muillm_comm_datatype_t datatype
) {
  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  int local_size = comm->local_size;

  if (dst_count != in_count * local_size) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // we only do the copy in the reduction buffers only if there is aliasing
  bool aliasing = false;

  for (int r = 0; r < local_size; r++) {
    if (src_ptrs[r] == dst_ptrs[r]) {
      aliasing = true;
      break;
    }
  }

  // get reduction buffer set
  muillm_comm_buffer_set_t* buffer_set = nullptr;

  if ((muillm_error = muillm_comm_get_buffer_set(comm, in_count, datatype, &buffer_set)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"Reduction failed when ensuring capacity"<<std::endl;
    return muillm_error;
  }

  // copy into reduction buffers if needed
  if (aliasing) {
    size_t byte_count = __comm_size(datatype, in_count);
    for (int r = 0; r < local_size; r++) {

      __muillm_gpu_copy(buffer_set->buffers[r], src_ptrs[r], byte_count, comm->streams[r]);
    }
  }

  // ensure all GPUs have copied into the reduction buffers
  if ((muillm_error = __mui_gpu_barrier(comm)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  // do the all-gather
  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(in_count, THREADS_PER_BLOCK);

  // gather
  if (datatype == MUILLM_COMM_FP16) {
    if (local_size == 4) {
      for (int r = 0; r < local_size; r++) {
        __all_gather_fp16_tp4_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[r]>>>(
          (const half*) (aliasing ? buffer_set->buffers[0] : src_ptrs[0]),
          (const half*) (aliasing ? buffer_set->buffers[1] : src_ptrs[1]),
          (const half*) (aliasing ? buffer_set->buffers[2] : src_ptrs[2]),
          (const half*) (aliasing ? buffer_set->buffers[3] : src_ptrs[3]),
          (half*) dst_ptrs[r],
          in_count
        );
      }
    } else if (local_size == 2) {
      for (int r = 0; r < local_size; r++) {
        __all_gather_fp16_tp2_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[r]>>>(
          (const half*) (aliasing ? buffer_set->buffers[0] : src_ptrs[0]),
          (const half*) (aliasing ? buffer_set->buffers[1] : src_ptrs[1]),
          (half*) dst_ptrs[r],
          in_count
        );
      }
    } else {
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else if (datatype == MUILLM_COMM_FP32) {
    if (local_size == 4) {
      for (int r = 0; r < local_size; r++) {
        __all_gather_fp32_tp4_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[r]>>>(
          (const float*) (aliasing ? buffer_set->buffers[0] : src_ptrs[0]),
          (const float*) (aliasing ? buffer_set->buffers[1] : src_ptrs[1]),
          (const float*) (aliasing ? buffer_set->buffers[2] : src_ptrs[2]),
          (const float*) (aliasing ? buffer_set->buffers[3] : src_ptrs[3]),
          (float*) dst_ptrs[r],
          in_count
        );
      }
    } else if (local_size == 2) {

      for (int r = 0; r < local_size; r++) {
        __all_gather_fp32_tp2_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, comm->streams[r]>>>(
          (const float*) (aliasing ? buffer_set->buffers[0] : src_ptrs[0]),
          (const float*) (aliasing ? buffer_set->buffers[1] : src_ptrs[1]),
          (float*) dst_ptrs[r],
          in_count
        );
      }
    } else {
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  } else {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}
