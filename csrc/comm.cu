#include "comm.h"

#include <iostream>

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))


muillm_comm_error_t muillm_comm_init(
  int local_size,
  bool allocate_streams,
  muillm_comm_t** comm_ptr
) {

  hipError_t error;

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
      if ((error = hipExtMallocWithFlags((void**) &comm->signals[r], sizeof(uint64_t), hipMallocSignalMemory)) == hipSuccess) {
        std::cout<<"Succeeded allocating signal memory"<<std::endl;
        *comm->signals[r] = 0;
      } else {
        delete[] comm->signals;
        comm->signals = nullptr;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
      std::cout<<"signal "<<r<<" ptr: "<<comm->signals[r]<<std::endl;
    }
  } else {
    delete[] comm->signals;
    comm->signals = nullptr;
  }

  if (hipSetDevice(default_device_id) != hipSuccess) {
    std::cout<<"Error setting back the default device"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  *comm_ptr = comm;
  return MUILLM_COMM_SUCCESS;
}

#define THREADS_PER_BLOCK 256

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

muillm_comm_error_t muillm_comm_all_reduce_sum(
  muillm_comm_t* comm,
  const void** src_ptrs,
  void** dst_ptrs,
  size_t count,
  muillm_comm_datatype_t datatype
) {

  // TODO:
  //  try approach where
  // 1) place vectors in remote GPU memories (double buffed)
  // 2) do a rendez-vous (record event, wait on other gpus)
  // 3) finalize reduction
  // it should have lower latency than rendez-vous, reduce-broadcast, rendez-vous
  // TODO next:
  // fuse all reduce vector copy in gemv
  int local_size = comm->local_size;

  if (comm->signals != nullptr) {
    comm->signal_seq_no++;
    uint64_t seq_no = comm->signal_seq_no;

    // GPU barrier: GPU0 will wait on the others
    for (int r = 1; r < local_size; r++) {
      if (hipStreamWriteValue32(comm->streams[r], (uint32_t*)comm->signals[r], seq_no, 0) != hipSuccess) {
        std::cout<<"Failed to write value "<<r<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }

    // wait for the other ranks
    for (int other_rank = 1; other_rank < local_size; other_rank++) {

      hipError_t error;
      if ((error = hipStreamWaitValue32(comm->streams[0], (uint32_t*)comm->signals[other_rank], seq_no, hipStreamWaitValueEq, -1)) != hipSuccess) {
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

      hipError_t error;
      if ((error = hipStreamWaitEvent(comm->streams[0], comm->acquire_events[other_rank], 0)) != hipSuccess) {
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
      hipError_t error;
      if ((error = hipStreamWaitValue32(comm->streams[other_rank], (uint32_t*)comm->signals[0], seq_no, hipStreamWaitValueEq, -1)) != hipSuccess) {
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

      hipError_t error;
      if ((error = hipStreamWaitEvent(comm->streams[other_rank], comm->release_events[0], 0)) != hipSuccess) {
        std::cout<<"Failed to wait for event"<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }
  }

  return MUILLM_COMM_SUCCESS;
}
