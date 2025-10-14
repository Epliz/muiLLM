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

#define HIP_CHECK(rank, call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "Rank " << rank << " HIP error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

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

#include <distributed/c10d/ProcessGroup.hpp>

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
  std::shared_ptr<c10d::ProcessGroup> process_group;
} muillm_comm_local_socket_t;

// base structure
typedef struct muillm_comm {
  muillm_comm_method_t transfer_method;

  int world_size;
  int local_size;
  int rank;
  int local_rank;

  std::shared_ptr<c10d::ProcessGroup> process_group;
} muillm_comm_t;

muillm_comm_error_t __open_local_socket(
    int local_size,
    int local_rank,
    std::shared_ptr<c10d::ProcessGroup>& process_group,
    muillm_comm_local_socket_t* local_socket
);

muillm_comm_error_t __close_local_socket(
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
#include <sys/un.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <errno.h>
#include <poll.h>

#include <torch/torch.h>
#include <distributed/c10d/ProcessGroup.hpp>

muillm_comm_error_t __open_local_socket(
    int local_size,
    int local_rank,
    std::shared_ptr<c10d::ProcessGroup>& process_group,
    muillm_comm_local_socket_t* local_socket
) {
  local_socket->process_group = process_group;
  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t __close_local_socket(
    muillm_comm_local_socket_t* local_socket
) {
  return MUILLM_COMM_SUCCESS;
}
// do a barrier using the local socket
muillm_comm_error_t __local_socket_barrier(
    muillm_comm_t* comm
) {
  comm->process_group->barrier()->wait();
  
  return MUILLM_COMM_SUCCESS;
}

// do a broadcast using the local socker
muillm_comm_error_t __local_socket_broadcast(
    muillm_comm_t* comm,
    int src_local_rank,
    void* ptr,
    size_t byte_count
) {
  // allocate a tensor of the right size on CPU
  auto tensor_options = at::TensorOptions()
                            .dtype(torch::kInt8) // int8
                            .layout(at::kStrided)
                            .device(torch::kCPU)
                            .requires_grad(false);
  torch::Tensor cpu_tensor = torch::empty({(int) byte_count}, tensor_options);

  // copy the data into the tensor
  if (comm->local_rank == src_local_rank) {
    memcpy(cpu_tensor.data_ptr(), ptr, byte_count);
  }

  // make a tensor on the device required for comms
  auto device = comm->process_group->getDeviceTypes()[0];
  torch::Tensor device_tensor = cpu_tensor.to(device);

  // do a broadcast using the process group
  auto broadcast_options = c10d::BroadcastOptions();
  broadcast_options.rootRank = src_local_rank;
  broadcast_options.rootTensor = 0;

  // create a std::vector with device_tensor inside
  auto tensor_vector = std::vector<torch::Tensor>{device_tensor};
  comm->process_group->broadcast(tensor_vector, broadcast_options)->wait();

  // copy back to the cpu tensor
  auto back_tensor = device_tensor.to(torch::kCPU);

  // copy the data back
  memcpy(ptr, back_tensor.data_ptr(), byte_count);

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
  int local_size = comm->local_size;
  // allocate a tensor of the right size on CPU
  auto tensor_options = at::TensorOptions()
                            .dtype(torch::kInt8) // int8
                            .layout(at::kStrided)
                            .device(torch::kCPU)
                            .requires_grad(false);
  torch::Tensor cpu_tensor = torch::empty({(int) byte_count}, tensor_options);

  // copy the data into the tensor
  memcpy(cpu_tensor.data_ptr(), in_ptr, byte_count);

  // make a tensor on the device required for comms
  auto device = comm->process_group->getDeviceTypes()[0];
  torch::Tensor device_tensor = cpu_tensor.to(device);

  // create the output tensors as well
  auto out_tensor_options = at::TensorOptions()
                            .dtype(torch::kInt8) // int8
                            .layout(at::kStrided)
                            .device(device) // on the device for communications
                            .requires_grad(false);

  std::vector<torch::Tensor> out_tensor_vector;
  for (int r = 0; r < local_size; r++) {
    auto out_tensor = torch::empty({(int) byte_count}, out_tensor_options);
    out_tensor_vector.push_back(out_tensor);
  }

  // create the vectors of tensors for inputs/outputs
  auto in_tensor_vector = std::vector<torch::Tensor>{device_tensor};
  auto out_tensor_vectors = std::vector<std::vector<torch::Tensor>>{out_tensor_vector};

  // do the all gather
  comm->process_group->allgather(out_tensor_vectors, in_tensor_vector)->wait();

  // copy back to the CPU memory
  for (int r = 0; r < local_size; r++) {
    auto out_tensor_cpu = out_tensor_vector[r].to(torch::kCPU);
    memcpy(static_cast<char*>(out_ptr) + r * byte_count, out_tensor_cpu.data_ptr(), byte_count);
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
  size_t capacity;
} muillm_comm_p2p_buffer_set_t;


typedef struct muillm_comm_p2p_counter_set {
  uint32_t* counters_host;
  uint32_t* counters;
  uint32_t* local_count_cache;
} muillm_comm_p2p_counter_set_t;

typedef struct muillm_comm_p2p_stream_context {
  hipStream_t stream;

  // reduction buffer sets
  muillm_comm_p2p_buffer_set_t* first_buffers;
  muillm_comm_p2p_buffer_set_t* second_buffers;

  // counters
  muillm_comm_p2p_counter_set_t* first_counters;
  muillm_comm_p2p_counter_set_t* second_counters;
  muillm_comm_p2p_counter_set_t* third_counters;
  muillm_comm_p2p_counter_set_t* fourth_counters;

  // shared signal memory to synchronize GPUs
  uint32_t* signal_host;
  uint32_t* signal;

  uint32_t signal_seq_no;

  // event for synchronizing the stream
  hipEvent_t stream_sync_event;

  // event to flush the caches
  hipEvent_t cache_flush_event;

  // indicator whether we can skip the cache flush event
  bool cant_skip_cache_flush_event;
} muillm_comm_p2p_stream_context_t;

#define MUILLM_COMM_P2P_NUM_STREAM_CONTEXTS 2

typedef struct muillm_comm_p2p: muillm_comm {
  muillm_comm_p2p_stream_context_t* stream_contexts[MUILLM_COMM_P2P_NUM_STREAM_CONTEXTS];

  muillm_gpu_info_t* gpu_info;
} muillm_comm_p2p_t;

muillm_comm_error_t muillm_comm_p2p_init_stream_context(
    muillm_comm_p2p_t* comm,
    bool cant_skip_cache_flush_event,
    muillm_comm_p2p_stream_context_t** ctx_ptr,
    hipStream_t stream
);

muillm_comm_error_t muillm_comm_p2p_init_comm(
    int world_size,
    int local_size,
    int rank,
    int local_rank,
    const muillm_comm_local_socket_t* local_socket,
    muillm_comm_p2p_t** comm_ptr,
    hipStream_t stream
);

muillm_comm_error_t muillm_comm_p2p_destroy_comm(
    muillm_comm_p2p_t* comm
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
  muillm_comm_p2p_stream_context_t* ctx,
  hipStream_t stream
);

#define MUILLM_COMM_INITIAL_BUFFER_CAPACITY (128 * 1024 * 1024) // 128MiB

static muillm_comm_error_t __free_buffer_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_buffer_set_t* buffer_set,
  bool sync = true
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
  if (sync) {
    if ((error =__local_socket_barrier(comm)) != MUILLM_COMM_SUCCESS) {
      TORCH_CHECK(false, "an error happened when doing barrier");
      return error;
    }
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
  if (sync) {
    if ((error =__local_socket_barrier(comm)) != MUILLM_COMM_SUCCESS) {
      return error;
    }
}

  // deallocate the previous memory
  if (buffer_set->buffers[local_rank] != nullptr) {
    if (hipFree(buffer_set->buffers[local_rank]) != hipSuccess) {
      TORCH_CHECK(false, "an error happened when freeing GPU memory");
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }

  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __allocate_counter_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_counter_set_t** counter_set_
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  muillm_comm_error_t error;

  muillm_comm_p2p_counter_set_t* counter_set = new muillm_comm_p2p_counter_set_t;
  if (counter_set == nullptr) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
  counter_set->counters_host = nullptr;
  counter_set->counters = nullptr;
  *counter_set_ = counter_set;


  // we will import the memory mappings for that specific GPU
  if (hipSetDevice(local_rank) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // allocate counters memory
  // we need it to be on the CPU side so that there is no coherency issues between GPUs
  // (need correct fine-grained atomic operations)
  __allocate_locked_shared_cpu_mem(
    comm,
    sizeof(uint64_t) * MUILLM_MAX_GPUS, // alloc 8 bytes even though we use only 4
    (void**) &counter_set->counters_host,
    (void**) &counter_set->counters
  );

  // initialize the counters to 0
  if (local_rank == 0) {
    // the counters are shared, so only one rank needs to initialize them
    // __allocate_shared_gpu_mem after will guarantee all ranks see the updated value
    if (hipMemset(counter_set->counters, 0, sizeof(uint64_t) * MUILLM_MAX_GPUS) != hipSuccess) {
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
  }

  // allocate local count cache on GPU
  if (hipMalloc(&counter_set->local_count_cache, sizeof(uint32_t)) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // synchronize the device
  if (hipDeviceSynchronize() != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __free_counter_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_counter_set_t* counter_set,
  bool sync = true
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
  if (sync) {
    if ((error =__local_socket_barrier(comm)) != MUILLM_COMM_SUCCESS) {
      TORCH_CHECK(false, "an error happened when doing barrier");
      return error;
    }
  }

  // free the counters memory as well
  if (counter_set->counters_host != nullptr) {
    __deallocate_locked_shared_cpu_mem(
      comm,
      counter_set->counters_host
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

  //std::cout<<"rank "<<local_rank<<" reallocating buffers for capacity "<<capacity<<"..."<<std::endl;

  muillm_comm_error_t error;

  // we will import the memory mappings for that specific GPU
  if (hipSetDevice(local_rank) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // free the memory mappings and so on
  if ((error =__free_buffer_set(comm, buffer_set)) != MUILLM_COMM_SUCCESS) {
    return error;
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

muillm_comm_error_t muillm_comm_p2p_get_stream_context(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_stream_context_t** ctx_ptr
) {
  muillm_comm_p2p_stream_context_t* first_context = comm->stream_contexts[0];
  *ctx_ptr = first_context;

  // rotate the contexts
  for (int i = 0; i < MUILLM_COMM_P2P_NUM_STREAM_CONTEXTS - 1; i++) {
    comm->stream_contexts[i] = comm->stream_contexts[i + 1];
  }
  comm->stream_contexts[MUILLM_COMM_P2P_NUM_STREAM_CONTEXTS - 1] = first_context;

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t muillm_comm_p2p_get_buffer_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_stream_context_t* ctx,
  size_t capacity,
  muillm_comm_p2p_buffer_set_t** buffer_set,
  hipStream_t stream
) {
  
  muillm_comm_error_t muillm_error;

  if ((muillm_error = __ensure_buffer_set_capacity(comm, ctx->first_buffers, capacity, stream)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  // always return the current first buffer set
  *buffer_set = ctx->first_buffers;

  // swap buffer sets for next time
  muillm_comm_p2p_buffer_set_t* tmp = ctx->first_buffers;
  ctx->first_buffers = ctx->second_buffers;
  ctx->second_buffers = tmp;

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t muillm_comm_p2p_get_buffer_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_stream_context_t* ctx,
  size_t count,
  muillm_comm_datatype_t datatype,
  muillm_comm_p2p_buffer_set_t** buffer_set,
  hipStream_t stream
) {
  size_t size = __comm_size(datatype, count);
  return muillm_comm_p2p_get_buffer_set(comm, ctx, size, buffer_set, stream);
}

muillm_comm_error_t muillm_comm_p2p_get_counter_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_stream_context_t* ctx,
  muillm_comm_p2p_counter_set_t** counter_set
) {
  
  muillm_comm_error_t muillm_error;


  // always return the current first buffer set
  *counter_set = ctx->first_counters;

  // swap buffer sets for next time
  muillm_comm_p2p_counter_set_t* tmp = ctx->first_counters;
  ctx->first_counters = ctx->second_counters;
  ctx->second_counters = ctx->third_counters;
  ctx->third_counters = ctx->fourth_counters;
  ctx->fourth_counters = tmp;

  return MUILLM_COMM_SUCCESS;
}


muillm_comm_error_t muillm_comm_p2p_get_next_counter_set(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_stream_context_t* ctx,
  muillm_comm_p2p_counter_set_t** counter_set
) {
  
  muillm_comm_error_t muillm_error;

  // always return the current first buffer set
  *counter_set = ctx->second_counters;

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
    int can_access = 0;
    if (hipDeviceCanAccessPeer(&can_access, local_rank, d) != hipSuccess) {
      // TODO: return error
      return MUILLM_COMM_UNKNOWN_ERROR;
    }
    if (!can_access) {
      if (hipDeviceEnablePeerAccess(d, 0) != hipSuccess) {
        // TODO: return error
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }
  }

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t muillm_comm_p2p_init_stream_context(
    muillm_comm_p2p_t* comm,
    bool cant_skip_cache_flush_event,
    muillm_comm_p2p_stream_context_t** ctx_ptr,
    hipStream_t stream
) {
  muillm_comm_error_t muillm_error;

  // create the ctx object
  muillm_comm_p2p_stream_context_t* ctx = new muillm_comm_p2p_stream_context_t;

  // create an additional non blocking stream
  if (hipStreamCreateWithFlags(&ctx->stream, hipStreamNonBlocking) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  ctx->signal_host = nullptr;
  ctx->signal = nullptr;
  ctx->signal_seq_no = 0;

  if (hipEventCreateWithFlags(&ctx->stream_sync_event, hipEventDisableTiming | hipEventDisableSystemFence) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // by default, do not skip the cache flush
  // but MI300 and successors don't need it apparently
  ctx->cant_skip_cache_flush_event = cant_skip_cache_flush_event;

  // allocate cache flush event
  if (hipEventCreateWithFlags(&ctx->cache_flush_event, hipEventDisableTiming | hipEventReleaseToSystem) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // allocate signal memory
  __allocate_locked_shared_cpu_mem(
    comm,
    sizeof(uint64_t), // alloc 8 bytese even though we use only 4
    (void**) &ctx->signal_host,
    (void**) &ctx->signal
  );

  if (ctx->signal_host == nullptr || ctx->signal == nullptr) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
  // initialize to 0
  if (hipMemset(ctx->signal, 0, sizeof(uint64_t)) != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // initialize buffer sets
  if ((muillm_error = __init_buffer_set(comm, &ctx->first_buffers, stream)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  if ((muillm_error = __init_buffer_set(comm, &ctx->second_buffers, stream)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  // initialize counter sets
  if ((muillm_error = __allocate_counter_set(comm, &ctx->first_counters)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }
  if ((muillm_error = __allocate_counter_set(comm, &ctx->second_counters)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }
  if ((muillm_error = __allocate_counter_set(comm, &ctx->third_counters)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }
  if ((muillm_error = __allocate_counter_set(comm, &ctx->fourth_counters)) != MUILLM_COMM_SUCCESS) {
    return muillm_error;
  }

  // return the comm object
  *ctx_ptr = ctx;

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

  comm->process_group = local_socket->process_group;

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
  bool cant_skip_cache_flush_event = comm->gpu_info->arch < MUILLM_GPU_ARCH_MI300;

  // allocate the stream contexts
  for (int i = 0; i < MUILLM_COMM_P2P_NUM_STREAM_CONTEXTS; i++) {
    comm->stream_contexts[i] = nullptr;
    if ((muillm_error = muillm_comm_p2p_init_stream_context(
          comm,
          cant_skip_cache_flush_event,
          &comm->stream_contexts[i],
          stream
        )) != MUILLM_COMM_SUCCESS) {
      return muillm_error;
    }
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

muillm_comm_error_t muillm_comm_p2p_destroy_stream_context(
    muillm_comm_p2p_t* comm,
    muillm_comm_p2p_stream_context_t* ctx
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  muillm_comm_error_t error;

  // destroy the stream afters synchronizing
  if (hipStreamSynchronize(ctx->stream) != hipSuccess) {
    std::cout<<"rank "<<local_rank<<" failed to synchronize stream"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
  if (hipStreamDestroy(ctx->stream) != hipSuccess) {
    std::cout<<"rank "<<local_rank<<" failed to destroy stream"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // free buffer sets
  if ((error = __free_buffer_set(comm, ctx->first_buffers, /*sync*/ false)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to free first buffer set"<<std::endl;
    return error;
  }
  delete ctx->first_buffers;

  if ((error = __free_buffer_set(comm, ctx->second_buffers, /*sync*/ false)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to free second buffer set"<<std::endl;
    return error;
  }
  delete ctx->second_buffers;

  // free counter sets
  if ((error = __free_counter_set(comm, ctx->first_counters, /*sync*/ false)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to free first counter set"<<std::endl;
    return error;
  }
  delete ctx->first_counters;
  if ((error = __free_counter_set(comm, ctx->second_counters, /*sync*/ false)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to free second counter set"<<std::endl;
    return error;
  }
  delete ctx->second_counters;
  if ((error = __free_counter_set(comm, ctx->third_counters, /*sync*/ false)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to free third counter set"<<std::endl;
    return error;
  }
  delete ctx->third_counters;
  if ((error = __free_counter_set(comm, ctx->fourth_counters, /*sync*/ false)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to free fourth counter set"<<std::endl;
    return error;
  }
  delete ctx->fourth_counters;

  // free signal memory
  if (ctx->signal_host != nullptr) {
    __deallocate_locked_shared_cpu_mem(
      comm,
      ctx->signal_host
    );
    ctx->signal_host = nullptr;
    ctx->signal = nullptr;
  }

  // destroy stream sync event
  if (hipEventDestroy(ctx->stream_sync_event) != hipSuccess) {
    std::cout<<"rank "<<local_rank<<" failed to destroy stream sync event"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // destroy cache flush event
  if (hipEventDestroy(ctx->cache_flush_event) != hipSuccess) {
    std::cout<<"rank "<<local_rank<<" failed to destroy cache flush event"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // delete the ctx object
  delete ctx;

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t muillm_comm_p2p_destroy_comm(
    muillm_comm_p2p_t* comm
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  muillm_comm_error_t error;

  muillm_comm_p2p_stream_context_t* ctx;
  if ((error = muillm_comm_p2p_get_stream_context(comm, &ctx)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to get stream context"<<std::endl;
    return error;
  }

  // we need to synchronize the ranks and block the  CPU so that we can deallocate
  // the previous receive buffers
  // synchronize to make sure no GPU is going to reference the previous memory
  if (hipDeviceSynchronize() != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // gpu barrier
  if ((error = __mui_gpu_barrier(comm, ctx, /*stream*/ 0)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to do gpu barrier"<<std::endl;
    return error;
  }

  // synchronize to make sure no GPU is going to reference the previous memory
  if (hipDeviceSynchronize() != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // make sure all CPUs have synchronized their GPUs
  if ((error =__local_socket_barrier(comm)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when doing barrier");
    return error;
  }

  // destroy stream contexts
  for (int i = 0; i < MUILLM_COMM_P2P_NUM_STREAM_CONTEXTS; i++) {
    if (comm->stream_contexts[i] != nullptr) {
      if ((error = muillm_comm_p2p_destroy_stream_context(comm, comm->stream_contexts[i])) != MUILLM_COMM_SUCCESS) {
        std::cout<<"rank "<<local_rank<<" failed to destroy stream context "<<i<<std::endl;
        return error;
      }
      comm->stream_contexts[i] = nullptr;
    }
  }

  // close local socket
  if (__close_local_socket((muillm_comm_local_socket_t*) comm) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to close local socket"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // delete gpu info
  if (comm->gpu_info != nullptr) {
    delete comm->gpu_info;
    comm->gpu_info = nullptr;
  }

  // delete the comm object
  delete comm;

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t __mui_stream_inc_value(hipStream_t stream, uint32_t* signal);

muillm_comm_error_t __mui_stream_inc_wait_value(hipStream_t stream, uint32_t* signal, uint32_t seq_no);

static muillm_comm_error_t __mui_gpu_barrier(muillm_comm_p2p_t* comm, muillm_comm_p2p_stream_context_t* ctx, hipStream_t stream) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  if (ctx->signal != nullptr) {
    ctx->signal_seq_no += local_size;
    uint64_t seq_no = ctx->signal_seq_no;

    // GPU barrier: all GPUs wait on each other
    if (ctx->cant_skip_cache_flush_event) {
      // on MI100, we get a crash if not putting this event here
      // record an event to flush caches
      if (hipEventRecord(ctx->cache_flush_event, stream) != hipSuccess) {
        std::cout<<"rank "<<local_rank<<" gpu barrier failed because hipEventRecord failed"<<std::endl;
        hipError_t err = hipGetLastError();
        const char* errStr = hipGetErrorString(err);
        std::cout<<"Last HIP error: "<<errStr<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }

    //std::cout<<"rank "<<local_rank<<" ctx "<<ctx<<" doing gpu barrier on signal "<<ctx->signal<<" with seq_no "<<seq_no<<std::endl;

    // write the values
    if ((muillm_error = __mui_stream_inc_wait_value(stream, ctx->signal, seq_no)) != MUILLM_COMM_SUCCESS) {
      std::cout<<"rank "<<local_rank<<" gpu barrier failed because __mui_stream_inc_wait_value failed"<<std::endl;
      return muillm_error;
    }
    //HIP_CHECK(local_rank, hipDeviceSynchronize()); // synchronize to see better interleaving
  } else {
    std::cout<<"rank "<<local_rank<<" gpu barrier failed because there is no signal memory"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t __mui_stream_inc_wait_value_cache_val(
  hipStream_t stream,
  uint32_t* signal,
  uint32_t seq_no,
  const uint32_t* __restrict__ uncached_val,
  uint32_t* __restrict__ cached_val
);

static muillm_comm_error_t __mui_gpu_barrier_cache_val(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_stream_context_t* ctx,
  hipStream_t stream,
  const uint32_t* __restrict__ uncached_val,
  uint32_t* __restrict__ cached_val
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  if (ctx->signal != nullptr) {
    ctx->signal_seq_no += local_size;
    uint64_t seq_no = ctx->signal_seq_no;

    // GPU barrier: all GPUs wait on each other
    if (ctx->cant_skip_cache_flush_event) {
      // on MI100, we get a crash if not putting this event here
      // record an event to flush caches
      if (hipEventRecord(ctx->cache_flush_event, stream) != hipSuccess) {
        std::cout<<"rank "<<local_rank<<" caching gpu barrier failed because hipEventRecord failed"<<std::endl;
        hipError_t err = hipGetLastError();
        const char* errStr = hipGetErrorString(err);
        std::cout<<"Last HIP error: "<<errStr<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }

    //std::cout<<"rank "<<local_rank<<" ctx "<<ctx<<" doing caching gpu barrier on signal "<<ctx->signal<<" with seq_no "<<seq_no<<std::endl;

    // write the values
    if ((muillm_error = __mui_stream_inc_wait_value_cache_val(stream, ctx->signal, seq_no, uncached_val, cached_val)) != MUILLM_COMM_SUCCESS) {
      std::cout<<"rank "<<local_rank<<" caching gpu barrier failed because __mui_stream_inc_wait_value failed"<<std::endl;
      return muillm_error;
    }

    //HIP_CHECK(local_rank, hipDeviceSynchronize()); // synchronize to see better interleaving
  } else {
    std::cout<<"rank "<<local_rank<<" caching gpu barrier failed because there is no signal memory"<<std::endl;
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

void* all2all_comm_init(
  int world_size,
  int rank,
  std::shared_ptr<c10d::ProcessGroup>& process_group
) {
  // assumme local_size == world_size for now
  int local_size = world_size;
  int local_rank = rank;

  muillm_comm_error_t muillm_error;

  // establish the local socket connection
  muillm_comm_local_socket_t local_socket;
  if ((muillm_error = __open_local_socket(local_size, local_rank, process_group, &local_socket)) != MUILLM_COMM_SUCCESS) {
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

void all2all_comm_destroy(void* comms) {
  muillm_comm_p2p_t* comm = (muillm_comm_p2p_t*) comms;

  muillm_comm_error_t muillm_error = muillm_comm_p2p_destroy_comm(comm);

  TORCH_CHECK(muillm_error == MUILLM_COMM_SUCCESS, "an error happened when destroying mui comm");
}

void all2all_dispatch_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ indices,
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
    uint32_t* __restrict__ expert_num_tokens,
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
    uint32_t* __restrict__ expert_num_tokens,
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
    uint32_t* __restrict__ local_count_cache,
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
    uint32_t* __restrict__ local_count_cache,
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> all2all_comm_dispatch_on_stream(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_stream_context_t* ctx,
  at::cuda::CUDAStream stream,
  torch::Tensor& x, // shape [num_tokens, hidden_dim]
  torch::Tensor& indices, // shape [num_tokens, experts_per_token]
  int num_local_experts,
  int max_recv
) {
  CHECK_INPUT(x);
  CHECK_INPUT(indices);

  auto device = x.device();

  int local_size = comm->local_size;
  int local_rank = comm->local_rank;


  int num_tokens = x.size(0);
  int hidden_dim = x.size(1);
  int num_experts_per_token = indices.size(1);

  // int max_recv = max_num_tokens * local_size;
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

  // set the stream
  at::cuda::setCurrentCUDAStream(stream);

  // TODO: an approach where we place the data in the buffers, sync GPUs, then read from the buffers
  // would probably be better due to less GPU syncs

  //
  // First we compute the send offsets for each rank
  //

  auto send_offsets_options = at::TensorOptions()
                            .dtype(torch::kInt32)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto send_offsets = torch::empty({local_size}, send_offsets_options);

  // we align the metadata pointer to 4k for better performance
  // so we align up the data size as such as the metadata pointer will be right after the data pointer
  size_t size_data = max_total_recv * hidden_dim * dtype_size;
  size_t aligned_size_data = ALIGN_UP(size_data, 4096);
  size_t size_metadata = max_total_recv * META_DIM * sizeof(int32_t);
  size_t capacity = aligned_size_data + size_metadata;

  muillm_comm_error_t muillm_error;

  // get the next counters to clear
  muillm_comm_p2p_counter_set_t* next_counter_set = nullptr;

  // we have to do this call before flipping the buffer sets with muillm_comm_p2p_get_buffer_set
  if (muillm_comm_p2p_get_next_counter_set(comm, ctx, &next_counter_set) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting next counter set");
  }

  // get the current counter set
  muillm_comm_p2p_counter_set_t* current_counter_set = nullptr;
  if (muillm_comm_p2p_get_counter_set(comm, ctx, &current_counter_set) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting current counter set");
  }

  // get reduction buffer set
  muillm_comm_p2p_buffer_set_t* buffer_set = nullptr;

  // this call flips the buffer sets
  if ((muillm_error = muillm_comm_p2p_get_buffer_set(comm, ctx, capacity, &buffer_set, stream)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting buffer set");
  }

  uint32_t* counters = (uint32_t*)current_counter_set->counters;
  uint32_t* next_counters = (uint32_t*) next_counter_set->counters;

  if (num_tokens > 0) {
    all2all_dispatch_compute_send_counts(
      stream,
      (const int32_t*)indices.data_ptr(),
      (uint32_t*)send_offsets.data_ptr(),
      counters,
      next_counters,
      num_local_experts,
      num_tokens,
      num_experts_per_token,
      local_size,
      local_rank
    );
  } else {
    // zero tokens to process
    if (local_rank == 0) {
      // we still need to zero out the next local counter
      // but we don't need to zero out the send_offsets as they are not used
      if (hipMemsetAsync(
            next_counters, // local counter to clear
            0,
            sizeof(uint32_t) * local_size,
            stream
          ) != hipSuccess) {
        TORCH_CHECK(false, "an error happened when doing hipMemsetAsync to zero out next local counter");
      }
    }
  }


  // we will zero expert_num_tokens in the dispatch pack send kernels
  auto expert_num_tokens_options = at::TensorOptions()
                            .dtype(torch::kInt32)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  torch::Tensor expert_num_tokens = torch::empty({num_local_experts}, expert_num_tokens_options);

  //
  // Second we pack and send the data to the experts
  //
  int buff_meta_offset = aligned_size_data;

  // // print the pointers for debugging
  // std::cout<<"rank "<<local_rank<<" num_local_experts "<<num_local_experts<<" num_tokens "<<num_tokens<<" num_experts_per_token "<<num_experts_per_token<<" hidden_dim "<<hidden_dim<<" max_recv "<<max_recv<<" max_total_recv "<<max_total_recv<<" total_send "<<total_send<<std::endl;
  // std::cout<<"rank "<<local_rank<<" buffer pointers: ";
  // for (int i = 0; i < 8; i++) {
  //   std::cout<<" "<<buffer_set->buffers[i];
  // }
  // std::cout<<std::endl;

  if (num_tokens > 0) {
    if (dtype == torch::kFloat32) {
      // buffers will be nullptr if local_size < 8, but it's ok to pass nullptr to the kernel
      all2all_dispatch_pack_send_buffers_fp32(
        stream,
        (const float*)x.data_ptr(),
        (const int32_t*)indices.data_ptr(),
        (uint32_t*)send_offsets.data_ptr(),
        (uint32_t*)expert_num_tokens.data_ptr(),
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
        (uint32_t*)expert_num_tokens.data_ptr(),
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
  } else {
    // zero tokens to process
    // we still need to zero out expert_num_tokens
    expert_num_tokens.zero_();
  }

  // HIP_CHECK(local_rank, hipGetLastError());
  // HIP_CHECK(local_rank, hipDeviceSynchronize());

  const uint32_t* uncached_val = &counters[local_rank]; // total_recv
  uint32_t* cached_val = current_counter_set->local_count_cache;
  //
  // We wait for all the GPUs to be done with sending data
  //
  if ((muillm_error = __mui_gpu_barrier_cache_val(comm, ctx, stream, uncached_val, cached_val)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when doing dispatch barrier 2");
  }

  //
  // Third, we unpack into the output tensors
  //

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
      cached_val,
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
      cached_val,
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
  muillm_comm_p2p_stream_context_t* ctx = nullptr;

  if (muillm_comm_p2p_get_stream_context(comm, &ctx) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting stream context");
  }

  auto device = x.device();
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device.index());

  return all2all_comm_dispatch_on_stream(
    comm,
    ctx,
    stream,
    x,
    indices,
    num_local_experts,
    max_recv
  );
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
at::Tensor all2all_compute_on_stream(
  at::cuda::CUDAStream stream,
  torch::Tensor& expert_num_tokens, // shape [num_local_experts]
  torch::Tensor& expert_x, // shape [num_local_experts, max_recv, hidden_dim]
  int rank
) {
  CHECK_INPUT(expert_x);

  auto device = expert_x.device();

  int num_local_experts = expert_x.size(0);
  int max_recv = expert_x.size(1);
  int hidden_dim = expert_x.size(2);

  auto dtype = expert_x.dtype();

  // set the stream
  at::cuda::setCurrentCUDAStream(stream);

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

at::Tensor all2all_compute(
  torch::Tensor& expert_num_tokens, // shape [num_local_experts]
  torch::Tensor& expert_x, // shape [num_local_experts, max_recv, hidden_dim]
  int rank
) {
  auto device = expert_x.device();
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device.index());

  return all2all_compute_on_stream(
    stream,
    expert_num_tokens,
    expert_x,
    rank
  );
}

void all2all_combine_pack_send_buffers_fp16(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const half* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    half* __restrict__ send_buf0, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf1, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf2, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf3, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf4, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf5, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf6, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf7, // shape [max_num_tokens, experts_per_token, hidden_dim]
    int num_local_experts,
    int max_recv,
    int experts_per_token,
    int hidden_dim,
    float s
);

void all2all_combine_pack_send_buffers_fp32(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const float* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    float* __restrict__ send_buf0, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf1, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf2, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf3, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf4, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf5, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf6, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf7, // shape [max_num_tokens, experts_per_token, hidden_dim]
    int num_local_experts,
    int max_recv,
    int experts_per_token,
    int hidden_dim,
    float s
);

void all2all_combine_unpack_fp32(
    hipStream_t stream,
    const float* __restrict__ recv_buf, // shape [num_tokens, experts_per_token, hidden_dim]
    const float* __restrict__ weights, // shape [num_tokens, experts_per_token]
    float* __restrict__ output, // shape [max_num_tokens, hidden_dim]
    int hidden_dim,
    int num_tokens,
    int experts_per_token
);

void all2all_combine_unpack_fp16(
    hipStream_t stream,
    const half* __restrict__ recv_buf, // shape [total_recv, hidden_sim]
    const float* __restrict__ weights, // shape [num_tokens, experts_per_token]
    half* __restrict__ output, // shape [max_num_tokens, hidden_dim]
    int hidden_dim,
    int num_tokens,
    int experts_per_token
);

// combine
torch::Tensor all2all_comm_combine_on_stream(
  muillm_comm_p2p_t* comm,
  muillm_comm_p2p_stream_context_t* ctx,
  at::cuda::CUDAStream stream,
  torch::Tensor& weights, // shape [num_tokens, experts_per_token]
  torch::Tensor& expert_meta, // shape [num_local_experts, max_recv, meta_dim] (expert_id, src_rank, src_token_id, topk_offset)
  torch::Tensor& expert_y, // shape [num_local_experts, max_recv, hidden_dim]
  torch::Tensor& expert_num_tokens, // shape [num_local_experts]
  float s = 1.0f
) {

  CHECK_INPUT(weights);
  CHECK_INPUT(expert_meta);
  CHECK_INPUT(expert_y);
  CHECK_INPUT(expert_num_tokens);

  auto device = expert_meta.device();

  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  int num_tokens = weights.size(0);
  int num_experts_per_token = weights.size(1);
  int num_local_experts = expert_num_tokens.size(0);
  int max_recv = expert_meta.size(1);
  int meta_dim = expert_meta.size(2);
  int hidden_dim = expert_y.size(2);

  // total number of tokens a rank has to combine is at most this much.
  // we use this to allocate the send buffer with the same size on all ranks
  int max_total_send = max_recv * num_experts_per_token; // TODO: inaccurate, could max_num_tokens * num_experts_per_token
  // but for this rank, the actual number of tokens to combine is:
  // int total_recv = num_tokens * num_experts_per_token;

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

  // set the stream
  at::cuda::setCurrentCUDAStream(stream);

  // TODO: an approach where we place the data in the buffers, sync GPUs, then read from the buffers
  // would probably be better due to less GPU syncs

  //
  // First we compute the send offsets for each rank
  //

  auto send_offsets_options = at::TensorOptions()
                            .dtype(torch::kInt32)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  auto send_offsets = torch::empty({local_size}, send_offsets_options);

  // we align the metadata pointer to 4k for better performance
  // so we align up the data size as such as the metadata pointer will be right after the data pointer
  size_t size_data = max_total_send * hidden_dim * dtype_size;
  size_t aligned_size_data = ALIGN_UP(size_data, 4096);
  size_t size_metadata = max_total_send * META_DIM * sizeof(int32_t);
  size_t capacity = aligned_size_data + size_metadata;

  muillm_comm_error_t muillm_error;

  // get reduction buffer set
  muillm_comm_p2p_buffer_set_t* buffer_set = nullptr;

  // this call flips the buffer sets
  if ((muillm_error = muillm_comm_p2p_get_buffer_set(comm, ctx, capacity, &buffer_set, stream)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting buffer set");
  }

  //
  // then we send/receive the data
  //

  if (dtype == torch::kFloat16) {
    // buffers will be nullptr if local_size < 8, but it's ok to pass nullptr to the kernel
    all2all_combine_pack_send_buffers_fp16(
      stream,
      (const int32_t*) expert_num_tokens.data_ptr(),
      (const int32_t*) expert_meta.data_ptr(),
      (const half*) expert_y.data_ptr(),
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
      num_experts_per_token,
      hidden_dim,
      s
    );
  } else if (dtype == torch::kFloat32) {
    // buffers will be nullptr if local_size < 8, but it's ok to pass nullptr to the kernel
    all2all_combine_pack_send_buffers_fp32(
      stream,
      (const int32_t*) expert_num_tokens.data_ptr(),
      (const int32_t*) expert_meta.data_ptr(),
      (const float*) expert_y.data_ptr(),
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
      num_experts_per_token,
      hidden_dim,
      s
    );
  } else {
    TORCH_CHECK(false, "datatype must be float16 for now");
  }

  //
  // We wait for all the GPUs to be done with sending data
  //
  if ((muillm_error = __mui_gpu_barrier(comm, ctx, stream)) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when doing combine barrier 2");
  }

  //
  // Finally, we need to combine the received data
  //
  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  torch::Tensor out_tokens = torch::empty({num_tokens, hidden_dim}, output_options);

  void* recv_buf = buffer_set->buffers[local_rank];

  if (dtype == at::kFloat) {
    all2all_combine_unpack_fp32(
      stream,
      (const float*) recv_buf,
      (const float*) weights.data_ptr(),
      (float*) out_tokens.data_ptr(),
      hidden_dim,
      num_tokens,
      num_experts_per_token
    );
  } else if (dtype == at::kHalf) {
    all2all_combine_unpack_fp16(
      stream,
      (const half*) recv_buf,
      (const float*) weights.data_ptr(),
      (half*) out_tokens.data_ptr(),
      hidden_dim,
      num_tokens,
      num_experts_per_token
    );
  } else {
    TORCH_CHECK(false, "Unsupported data type");
  }

  return out_tokens;
}

torch::Tensor all2all_comm_combine(
  void* comms,
  torch::Tensor& weights, // shape [num_tokens, experts_per_token]
  torch::Tensor& expert_meta, // shape [num_local_experts, max_recv, meta_dim] (expert_id, src_rank, src_token_id, topk_offset)
  torch::Tensor& expert_y, // shape [num_local_experts, max_recv, hidden_dim]
  torch::Tensor& expert_num_tokens, // shape [num_local_experts]
  float s = 1.0f
) {
  muillm_comm_p2p_t* comm = (muillm_comm_p2p_t*) comms;
  muillm_comm_p2p_stream_context_t* ctx = nullptr;

  if (muillm_comm_p2p_get_stream_context(comm, &ctx) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting stream context");
  }

  auto device = expert_meta.device();
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device.index());

  return all2all_comm_combine_on_stream(
    comm,
    ctx,
    stream,
    weights,
    expert_meta,
    expert_y,
    expert_num_tokens,
    s
  );
}

torch::Tensor all2all_comm_single_stream(
  void* comms_,
  torch::Tensor& x, // shape [num_tokens, hidden_dim]
  torch::Tensor& indices, // shape [num_tokens, experts_per_token]
  torch::Tensor& weights, // shape [num_tokens, experts_per_token]
  int num_local_experts,
  int max_recv
) {

  muillm_comm_p2p_t* comm = (muillm_comm_p2p_t*) comms_;
  muillm_comm_p2p_stream_context_t* ctx = nullptr;

  if (muillm_comm_p2p_get_stream_context(comm, &ctx) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting stream context");
  }

  auto device = x.device();
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device.index());

  int local_rank = comm->local_rank;

  // First dispatch
  auto dispatch_outputs = all2all_comm_dispatch_on_stream(
    comm,
    ctx,
    stream,
    x,
    indices,
    num_local_experts,
    max_recv
  );

  auto expert_num_tokens = std::get<0>(dispatch_outputs);
  auto expert_x = std::get<1>(dispatch_outputs);
  auto expert_meta = std::get<2>(dispatch_outputs);

  // Nota:
  // I am not sure if fusing compute in combine is in the spirit of the
  // competition, but I am pretty the top submissions will be doing it.
  bool fuse_compute_in_combine = true;

  if (!fuse_compute_in_combine) {
    // Then compute
    expert_x = all2all_compute_on_stream(
      stream,
      expert_num_tokens,
      expert_x,
      comm->rank
    );
  }

  // Finally combine
  float s = fuse_compute_in_combine ? (1.0f + local_rank) : 1.0f;
  return all2all_comm_combine_on_stream(
    comm,
    ctx,
    stream,
    weights,
    expert_meta,
    expert_x,
    expert_num_tokens,
    s
  );
}

torch::Tensor all2all_comm_multi_stream(
  void* comms_,
  torch::Tensor& x, // shape [num_tokens, hidden_dim]
  torch::Tensor& indices, // shape [num_tokens, experts_per_token]
  torch::Tensor& weights, // shape [num_tokens, experts_per_token]
  int num_local_experts,
  int max_recv
) {

  muillm_comm_p2p_t* comm = (muillm_comm_p2p_t*) comms_;

  // std::cout<<"rank "<<comm->rank<<" all2all multi stream start"<<std::endl;

  muillm_comm_p2p_stream_context_t* first_ctx = nullptr;
  muillm_comm_p2p_stream_context_t* second_ctx = nullptr;

  if (muillm_comm_p2p_get_stream_context(comm, &first_ctx) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting stream context");
  }

  if (muillm_comm_p2p_get_stream_context(comm, &second_ctx) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when getting stream context");
  }
  
  // todo: use different streams
  auto device = x.device();
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device.index());

  at::cuda::CUDAStream first_stream = stream;
  at::cuda::CUDAStream second_stream = at::cuda::getStreamFromExternal(second_ctx->stream, device.index());

  int local_rank = comm->local_rank;

  // split the inputs into two halves
  int num_tokens = x.size(0);
  int half_max_recv = max_recv / 2;

  int first_half_num_tokens = num_tokens / 2;
  int second_half_num_tokens = num_tokens - first_half_num_tokens;

  auto x1 = x.narrow(0, 0, first_half_num_tokens);
  auto x2 = x.narrow(0, first_half_num_tokens, second_half_num_tokens);

  auto indices1 = indices.narrow(0, 0, first_half_num_tokens);
  auto indices2 = indices.narrow(0, first_half_num_tokens, second_half_num_tokens);

  auto weights1 = weights.narrow(0, 0, first_half_num_tokens);
  auto weights2 = weights.narrow(0, first_half_num_tokens, second_half_num_tokens);

  // First dispatch
  //std::cout<<"rank "<<comm->rank<<" first dispatch"<<std::endl;

  // record an event on the main stream
  if (hipEventRecord(first_ctx->stream_sync_event, first_stream) != hipSuccess) {
    TORCH_CHECK(false, "an error happened when recording event");
  }

  auto dispatch_outputs1 = all2all_comm_dispatch_on_stream(
    comm,
    first_ctx,
    first_stream,
    x1,
    indices1,
    num_local_experts,
    half_max_recv
  );

  //std::cout<<"rank "<<comm->rank<<" second dispatch"<<std::endl;

  // make the second stream wait for the event
  if (hipStreamWaitEvent(second_stream, first_ctx->stream_sync_event, 0) != hipSuccess) {
    TORCH_CHECK(false, "an error happened when waiting for event");
  }

  auto dispatch_outputs2 = all2all_comm_dispatch_on_stream(
    comm,
    second_ctx,
    second_stream,
    x2,
    indices2,
    num_local_experts,
    half_max_recv
  );

  auto expert_num_tokens1 = std::get<0>(dispatch_outputs1);
  auto expert_x1 = std::get<1>(dispatch_outputs1);
  auto expert_meta1 = std::get<2>(dispatch_outputs1);

  auto expert_num_tokens2 = std::get<0>(dispatch_outputs2);
  auto expert_x2 = std::get<1>(dispatch_outputs2);
  auto expert_meta2 = std::get<2>(dispatch_outputs2);

  // Nota:
  // I am not sure if fusing compute in combine is in the spirit of the
  // competition, but I am pretty the top submissions will be doing it.
  bool fuse_compute_in_combine = true;

  if (!fuse_compute_in_combine) {
    // Then compute
    //std::cout<<"rank "<<comm->rank<<" first compute"<<std::endl;
    expert_x1 = all2all_compute_on_stream(
      first_stream,
      expert_num_tokens1,
      expert_x1,
      comm->rank
    );

    //std::cout<<"rank "<<comm->rank<<" second compute"<<std::endl;
    expert_x2 = all2all_compute_on_stream(
      second_stream,
      expert_num_tokens2,
      expert_x2,
      comm->rank
    );
  }

  // Finally combine
  float s = fuse_compute_in_combine ? (1.0f + local_rank) : 1.0f;
  //std::cout<<"rank "<<comm->rank<<" first combine"<<std::endl;
  torch::Tensor out1 = all2all_comm_combine_on_stream(
    comm,
    first_ctx,
    first_stream,
    weights1,
    expert_meta1,
    expert_x1,
    expert_num_tokens1,
    s
  );

  //std::cout<<"rank "<<comm->rank<<" second combine"<<std::endl;
  torch::Tensor out2 = all2all_comm_combine_on_stream(
    comm,
    second_ctx,
    second_stream,
    weights2,
    expert_meta2,
    expert_x2,
    expert_num_tokens2,
    s
  );

  // make the second stream record an event to indicate it is done
  if (hipEventRecord(second_ctx->stream_sync_event, second_stream) != hipSuccess) {
    TORCH_CHECK(false, "an error happened when recording event");
  }

  //std::cout<<"rank "<<comm->rank<<" all2all multi stream done"<<std::endl;
  // TODO: sync streams

  at::cuda::setCurrentCUDAStream(stream);

  // make the first stream wait for the second stream to be done
  if (hipStreamWaitEvent(first_stream, second_ctx->stream_sync_event, 0) != hipSuccess) {
    TORCH_CHECK(false, "an error happened when waiting for event");
  }

  // Concatenate the outputs
  return torch::cat({out1, out2}, 0);
}

#define ALL2ALL_COMM_MULTI_STREAM_THRESHOLD 64

torch::Tensor all2all_comm(
  void* comms_,
  torch::Tensor& x, // shape [num_tokens, hidden_dim]
  torch::Tensor& indices, // shape [num_tokens, experts_per_token]
  torch::Tensor& weights, // shape [num_tokens, experts_per_token]
  int num_local_experts,
  int max_recv
) {
  int num_tokens = x.size(0);
  int experts_per_token = indices.size(1);
  int tot_tokens = num_tokens * experts_per_token;

  if (false) { //(num_tokens < ALL2ALL_COMM_MULTI_STREAM_THRESHOLD) {
    return all2all_comm_single_stream(
      comms_,
      x,
      indices,
      weights,
      num_local_experts,
      max_recv
    );
  } else {
    return all2all_comm_multi_stream(
      comms_,
      x,
      indices,
      weights,
      num_local_experts,
      max_recv
    );
  }
}
"""

COMM_KERNELS_CUDA_CODE = """
#include <stdint.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <iostream>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define MUILLM_MAX_GPUS 8

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

__inline__ __device__ int __block_broadcast(int val, int threads_per_block=THREADS_PER_BLOCK) {
  int lane_id = threadIdx.x % warpSize;

  if (threads_per_block > warpSize) {
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

typedef struct half8 {
  half x, y, z, w, a, b, c, d;
} half8;

typedef struct half4 {
  half x, y, z, w;
} half4;

__global__ void __muillm_inc_value_p2p_kernel(
  uint32_t* signal
) {
  if (threadIdx.x == 0) {
    atomicAdd_system(signal, 1);
  }
}

muillm_comm_error_t __mui_stream_inc_value(hipStream_t stream, uint32_t* signal) {
  __muillm_inc_value_p2p_kernel<<<1, 1, 0, stream>>>(signal);
  // HIP_CHECK(hipGetLastError());
  // HIP_CHECK(hipDeviceSynchronize());
  return MUILLM_COMM_SUCCESS;
}

__device__ void __do_inc_wait_value_p2p(
  volatile uint32_t* signal,
  uint32_t seq_no
) {
  if (threadIdx.x == 0) {
    // increment the value
    uint32_t value = atomicAdd_system((uint32_t*) signal, 1) + 1;

    // wait for the other ranks
    // we need the comparison to be >= as one GPU might already increment the value before all the other GPUs
    // have seen the previous one
    if (value < seq_no) {
      while (*signal < seq_no) {
        // sleep to avoid congesting the bus
        __builtin_amdgcn_s_sleep(64);
      }
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
  // HIP_CHECK(hipGetLastError());
  // HIP_CHECK(hipDeviceSynchronize());
  return MUILLM_COMM_SUCCESS;
}

__global__ void __muillm_inc_wait_value_cache_val_p2p_kernel(
  volatile uint32_t* signal,
  uint32_t seq_no,
  const uint32_t* __restrict__ uncached_val,
  uint32_t* __restrict__ cached_val
) {
  __do_inc_wait_value_p2p(signal, seq_no);

  if (threadIdx.x == 0) {
    *cached_val = *uncached_val;
  }
}

muillm_comm_error_t __mui_stream_inc_wait_value_cache_val(
  hipStream_t stream,
  uint32_t* signal,
  uint32_t seq_no,
  const uint32_t* __restrict__ uncached_val,
  uint32_t* __restrict__ cached_val
) {
  __muillm_inc_wait_value_cache_val_p2p_kernel<<<1, 1, 0, stream>>>(signal, seq_no, uncached_val, cached_val);
  // HIP_CHECK(hipGetLastError());
  // HIP_CHECK(hipDeviceSynchronize());
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

  // HIP_CHECK(hipGetLastError());
  // HIP_CHECK(hipDeviceSynchronize());
  if (hipPeekAtLastError() != hipSuccess) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}

#define META_DIM 4

//
// All2All dispatch kernels
//

void __global__ all2all_dispatch_compute_send_counts_kernel(
  const int32_t* __restrict__ indices,
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
  __shared__ uint32_t send_counts_shared[MUILLM_MAX_GPUS];

  if (threadIdx.x < local_size) {
    send_counts_shared[threadIdx.x] = 0;
  }

  __syncthreads();

  if (local_rank == 0 && threadIdx.x < local_size) {
    // clear the counters for the next use
    next_counters[threadIdx.x] = 0;
  }

  // each thread handles one token expert
  for (int i = threadIdx.x; i < total_send; i += THREADS_PER_BLOCK) {
    // TODO: avoid division if num_local_experts is power of 2
    int32_t dst_expert = indices[i];
    int32_t dst_rank = dst_expert / num_local_experts;
    atomicAdd((uint32_t*)&send_counts_shared[dst_rank], 1);
  }

  __syncthreads();

  // do a single thread prefix sum to compute send offsets
  if (threadIdx.x < local_size) {
    int rank = threadIdx.x;
    uint32_t send_count = send_counts_shared[rank];
    uint32_t send_offset = atomicAdd_system((uint32_t*) &counters[rank], send_count);
    send_offsets[rank] = send_offset;
  }
}

void all2all_dispatch_compute_send_counts(
    hipStream_t stream,
    const int32_t* __restrict__ indices,
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
    send_offsets,
    counters,
    next_counters,
    num_local_experts,
    total_send,
    local_size,
    local_rank
  );
  // HIP_CHECK(hipGetLastError());
  // HIP_CHECK(hipDeviceSynchronize());
}

void __global__ all2all_dispatch_pack_send_buffers_fp32_kernel(
  const float* __restrict__ x,
  const int32_t* __restrict__ indices,
  uint32_t* __restrict__ send_offsets,
  uint32_t* __restrict__ expert_num_tokens,
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

  if (blockIdx.x == 0 && blockIdx.y == 0) {
    // zero expert_num_tokens
    for (int i = threadIdx.x; i < num_local_experts; i+= THREADS_PER_BLOCK) {
      expert_num_tokens[i] = 0;
    }
  }

  x = &x[token_idx * hidden_dim];

  // determine where to write out
  // TODO: avoid division if num_local_experts is power of 2
  int32_t dst_expert = indices[token_idx * num_experts_per_token + expert_idx];
  int32_t dst_rank = dst_expert / num_local_experts;

  int32_t send_pos = -1;
  if (threadIdx.x == 0) {
    // TODO: move to previous kernel to avoid global atomics by computing a global offset array
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
    uint32_t* __restrict__ expert_num_tokens,
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
    expert_num_tokens,
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
  uint32_t* __restrict__ expert_num_tokens,
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

  if (blockIdx.x == 0 && blockIdx.y == 0) {
    // zero expert_num_tokens
    for (int i = threadIdx.x; i < num_local_experts; i+= THREADS_PER_BLOCK) {
      expert_num_tokens[i] = 0;
    }
  }

  x = &x[token_idx * hidden_dim];

  // determine where to write out
  // TODO: avoid division if num_local_experts is power of 2
  int32_t dst_expert = indices[token_idx * num_experts_per_token + expert_idx];
  int32_t dst_rank = dst_expert / num_local_experts;

  int32_t send_pos = -1;
  if (threadIdx.x == 0) {
    // TODO: move to previous kernel to avoid global atomics by computing a global offset array
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
    uint32_t* __restrict__ expert_num_tokens,
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
    expert_num_tokens,
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

void __global__ all2all_dispatch_cache_recv_count_kernel(
    const uint32_t* __restrict__ recv_counts,
    uint32_t* __restrict__ local_count_cache
) {
  if (threadIdx.x == 0) {
    local_count_cache[0] = recv_counts[0];
  }
}

#define DISPATCH_UNPACK_THREADS_PER_BLOCK 256

#define DISPATCH_UNPACK_FP16_ELEMENTS_PER_THREAD 8
#define DISPATCH_UNPACK_FP16_ELEMENTS_PER_BLOCK_LOOP (DISPATCH_UNPACK_FP16_ELEMENTS_PER_THREAD * DISPATCH_UNPACK_THREADS_PER_BLOCK)

// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_dispatch_unpack_fp16_kernel(
    const half* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    half* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset) {

  int token_idx = blockIdx.x;

  if (token_idx >= recv_counts[0]) {
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
    // TODO: make a previous single block kernel to read the meta and compute offsets
    // to avoid global atomics
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
  int d = threadIdx.x * DISPATCH_UNPACK_FP16_ELEMENTS_PER_THREAD;
  for (; d + (DISPATCH_UNPACK_FP16_ELEMENTS_PER_THREAD - 1) < hidden_dim; d += DISPATCH_UNPACK_FP16_ELEMENTS_PER_BLOCK_LOOP) {
    const half8 recv = *((half8*)&recv_buf[d]);
    *(half8*)&expert_x[d] = recv;
  }
  // one thread handles the remaining elements
  for (; d < hidden_dim; d++) {
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
    uint32_t* __restrict__ local_count_cache,
    int hidden_dim,
    int max_recv,
    int local_expert_offset,
    int max_total_recv) {

  const int threads_per_block = DISPATCH_UNPACK_THREADS_PER_BLOCK;
  const int blocks = max_total_recv;

  all2all_dispatch_unpack_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    recv_buf,
    recv_meta,
    expert_num_tokens,
    expert_x,
    expert_meta,
    local_count_cache,
    hidden_dim,
    max_recv,
    local_expert_offset
  );
  // HIP_CHECK(hipGetLastError());
  // HIP_CHECK(hipDeviceSynchronize());
}

#define DISPATCH_UNPACK_FP32_ELEMENTS_PER_THREAD 4
#define DISPATCH_UNPACK_FP32_ELEMENTS_PER_BLOCK_LOOP (DISPATCH_UNPACK_FP32_ELEMENTS_PER_THREAD * DISPATCH_UNPACK_THREADS_PER_BLOCK)

// expected to be launched with 1D grid with total_recv blocks
// each block handles one token
void __global__ all2all_dispatch_unpack_fp32_kernel(
    const float* __restrict__ recv_buf,
    const int32_t* __restrict__ recv_meta,
    int32_t* __restrict__ expert_num_tokens,
    float* __restrict__ expert_x,
    int32_t* __restrict__ expert_meta,
    const uint32_t* __restrict__ recv_counts,
    int hidden_dim,
    int max_recv,
    int local_expert_offset) {
 
  int lane_id = threadIdx.x % warpSize;

  int token_idx = blockIdx.x;

  if (token_idx >= recv_counts[0]) {
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
    // TODO: make a previous single block kernel to read the meta and compute offsets
    // to avoid global atomics
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
  int d = threadIdx.x * DISPATCH_UNPACK_FP32_ELEMENTS_PER_THREAD;
  for (; d + (DISPATCH_UNPACK_FP32_ELEMENTS_PER_THREAD - 1) < hidden_dim; d += DISPATCH_UNPACK_FP32_ELEMENTS_PER_BLOCK_LOOP) {
    float4 recv = *((float4*)&recv_buf[d]);
    *((float4*)&expert_x[d]) = recv;
  }
  // one thread handles the remaining elements
  for (; d < hidden_dim; d++) {
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
    uint32_t* __restrict__ local_count_cache,
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
    local_count_cache,
    hidden_dim,
    max_recv,
    local_expert_offset
  );
  // HIP_CHECK(hipGetLastError());
  // HIP_CHECK(hipDeviceSynchronize());
}

#define COMPUTE_PER_THREAD_FP32 4
#define COMPUTE_PER_BLOCK_FP32_LOOP (COMPUTE_PER_THREAD_FP32 * THREADS_PER_BLOCK)

void __global__ all2all_compute_fp32_kernel(
  const int32_t* __restrict__ expert_num_tokens,
  const float* __restrict__ expert_x,
  float* __restrict__ expert_y,
  int max_recv,
  int hidden_dim,
  float s
) {

  int token_idx = blockIdx.x;
  int local_expert_idx = blockIdx.y;

  int num_local_tokens = expert_num_tokens[local_expert_idx];
  if (token_idx >= num_local_tokens) {
    return;
  }


  // realign the pointers
  expert_x = &expert_x[(local_expert_idx * max_recv + token_idx) * hidden_dim];
  expert_y = &expert_y[(local_expert_idx * max_recv + token_idx) * hidden_dim];

  int i = threadIdx.x * COMPUTE_PER_THREAD_FP32;
  for (; i + (COMPUTE_PER_THREAD_FP32 - 1) < hidden_dim; i += COMPUTE_PER_BLOCK_FP32_LOOP) {
    float4 x = *((float4*)&expert_x[i]);
    expert_y[i + 0] = x.x * s;
    expert_y[i + 1] = x.y * s;
    expert_y[i + 2] = x.z * s;
    expert_y[i + 3] = x.w * s;
  }
  // handle the remaining elements
  for (; i < hidden_dim; i++) {
    expert_y[i] = expert_x[i] * s;
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

  float s = 1.0f + rank;
  all2all_compute_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_x,
    expert_y,
    max_recv,
    hidden_dim,
    s
  );
  // HIP_CHECK(hipGetLastError());
  // HIP_CHECK(hipDeviceSynchronize());
}

#define COMPUTE_PER_THREAD_FP16 8
#define COMPUTE_PER_BLOCK_FP16_LOOP (COMPUTE_PER_THREAD_FP16 * THREADS_PER_BLOCK)

void __global__ all2all_compute_fp16_kernel(
  const int32_t* __restrict__ expert_num_tokens,
  const half* __restrict__ expert_x,
  half* __restrict__ expert_y,
  int max_recv,
  int hidden_dim,
  float s
) {

  int token_idx = blockIdx.x;
  int local_expert_idx = blockIdx.y;

  int num_local_tokens = expert_num_tokens[local_expert_idx];
  if (token_idx >= num_local_tokens) {
    return;
  }

  // realign the pointers
  expert_x = &expert_x[(local_expert_idx * max_recv + token_idx) * hidden_dim];
  expert_y = &expert_y[(local_expert_idx * max_recv + token_idx) * hidden_dim];

  int i = threadIdx.x * COMPUTE_PER_THREAD_FP16;
  for (; i + (COMPUTE_PER_THREAD_FP16 - 1) < hidden_dim; i += COMPUTE_PER_BLOCK_FP16_LOOP) {
    half8 x = *(half8*)&expert_x[i];
    expert_y[i + 0] = __float2half_rn(__half2float(x.x) * s);
    expert_y[i + 1] = __float2half_rn(__half2float(x.y) * s);
    expert_y[i + 2] = __float2half_rn(__half2float(x.z) * s);
    expert_y[i + 3] = __float2half_rn(__half2float(x.w) * s);
    expert_y[i + 4] = __float2half_rn(__half2float(x.a) * s);
    expert_y[i + 5] = __float2half_rn(__half2float(x.b) * s);
    expert_y[i + 6] = __float2half_rn(__half2float(x.c) * s);
    expert_y[i + 7] = __float2half_rn(__half2float(x.d) * s);
  }
  // handle the remaining elements
  for (; i < hidden_dim; i++) {
    expert_y[i] = __float2half_rn(__half2float(expert_x[i]) * s);
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

  float s = 1.0f + rank;
  all2all_compute_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_x,
    expert_y,
    max_recv,
    hidden_dim,
    s
  );
  // HIP_CHECK(hipGetLastError());
  // HIP_CHECK(hipDeviceSynchronize());
}

//
// All2All combine kernels
//

// Send kernels

#define THREADS_PER_BLOCK_COMBINE_PACK_SEND 256

void __global__ all2all_combine_pack_send_buffers_fp16_kernel(
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const half* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    half* __restrict__ send_buf0, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf1, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf2, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf3, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf4, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf5, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf6, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf7, // shape [max_num_tokens, experts_per_token, hidden_dim]
    int max_recv,
    int experts_per_token,
    int hidden_dim,
    float s
) {
  /* META contains: 
    meta_ptr[0] = dispatched_expert;
    meta_ptr[1] = orig_rank;
    meta_ptr[2] = token_idx;
    meta_ptr[3] = expert_idx;
  */

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
  int32_t dst_token_idx = local_expert_meta[2]; // token_idx
  int32_t dst_expert_idx = local_expert_meta[3]; // expert_idx

  int32_t send_pos = dst_token_idx * experts_per_token + dst_expert_idx;

  half* send_buffs[8] = {send_buf0, send_buf1, send_buf2, send_buf3, send_buf4, send_buf5, send_buf6, send_buf7};
  half* send_buf = send_buffs[dst_rank];

  // write to send_buf
  half* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  {
    int i = 8 * threadIdx.x;
    // vectorized part
    for (; i + 7 < hidden_dim; i += 8 * THREADS_PER_BLOCK_COMBINE_PACK_SEND) {
      half8 y = *((half8*) &local_expert_y[i]);
      half8* t = (half8*)&send_buf_ptr[i];
      t->x = __float2half_rn(__half2float(y.x) * s);
      t->y = __float2half_rn(__half2float(y.y) * s);
      t->z = __float2half_rn(__half2float(y.z) * s);
      t->w = __float2half_rn(__half2float(y.w) * s);
      t->a = __float2half_rn(__half2float(y.a) * s);
      t->b = __float2half_rn(__half2float(y.b) * s);
      t->c = __float2half_rn(__half2float(y.c) * s);
      t->d = __float2half_rn(__half2float(y.d) * s);
    }
    // remainders
    if (i + 3 < hidden_dim) {
      half4 y = *((half4*) &local_expert_y[i]);
      half4* t = (half4*)&send_buf_ptr[i];
      t->x = __float2half_rn(__half2float(y.x) * s);
      t->y = __float2half_rn(__half2float(y.y) * s);
      t->z = __float2half_rn(__half2float(y.z) * s);
      t->w = __float2half_rn(__half2float(y.w) * s);
      i += 4;
    }
    if (i + 1 < hidden_dim) {
      half2 y = *((half2*) &local_expert_y[i]);
      half2* t = (half2*)&send_buf_ptr[i];
      t->x = __float2half_rn(__half2float(y.x) * s);
      t->y = __float2half_rn(__half2float(y.y) * s);
      i += 2;
    }
    if (i < hidden_dim) {
      half y0 = local_expert_y[i + 0];
      send_buf_ptr[i + 0] = __float2half_rn(__half2float(y0) * s);
    }
  }
}

void all2all_combine_pack_send_buffers_fp16(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const half* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    half* __restrict__ send_buf0, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf1, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf2, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf3, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf4, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf5, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf6, // shape [max_num_tokens, experts_per_token, hidden_dim]
    half* __restrict__ send_buf7, // shape [max_num_tokens, experts_per_token, hidden_dim]
    int num_local_experts,
    int max_recv,
    int experts_per_token,
    int hidden_dim,
    float s
) {
  // call kernel to do the packing, with a 2D grid of size (max_recv, num_local_experts)
  const int threads_per_block = THREADS_PER_BLOCK_COMBINE_PACK_SEND;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_combine_pack_send_buffers_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_meta,
    expert_y,
    send_buf0,
    send_buf1,
    send_buf2,
    send_buf3,
    send_buf4,
    send_buf5,
    send_buf6,
    send_buf7,
    max_recv,
    experts_per_token,
    hidden_dim,
    s
  );
  // HIP_CHECK(hipGetLastError());
  // HIP_CHECK(hipDeviceSynchronize());
}

void __global__ all2all_combine_pack_send_buffers_fp32_kernel(
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const float* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    float* __restrict__ send_buf0, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf1, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf2, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf3, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf4, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf5, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf6, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf7, // shape [max_num_tokens, experts_per_token, hidden_dim]
    int max_recv,
    int experts_per_token,
    int hidden_dim,
    float s
) {
  /* META contains: 
    meta_ptr[0] = dispatched_expert;
    meta_ptr[1] = orig_rank;
    meta_ptr[2] = token_idx;
    meta_ptr[3] = expert_idx;
  */

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
  int32_t dst_token_idx = local_expert_meta[2]; // token_idx
  int32_t dst_expert_idx = local_expert_meta[3]; // expert_idx

  int32_t send_pos = dst_token_idx * experts_per_token + dst_expert_idx;

  float* send_buffs[8] = {send_buf0, send_buf1, send_buf2, send_buf3, send_buf4, send_buf5, send_buf6, send_buf7};
  float* send_buf = send_buffs[dst_rank];

  // write to send_buf
  float* send_buf_ptr = &send_buf[send_pos * hidden_dim];
  {
    int i = 4 * threadIdx.x;
    // vectorized part
    for (; i + 3 < hidden_dim; i += 4 * THREADS_PER_BLOCK_COMBINE_PACK_SEND) {
      float4 y = *((float4*) &local_expert_y[i]);
      *((float4*)&send_buf_ptr[i]) = s * y;
    }
    if (i + 1 < hidden_dim) {
      float2 y = *((float2*) &local_expert_y[i]);
      *((float2*)&send_buf_ptr[i]) = s * y;
      i += 2;
    }
    if (i < hidden_dim) {
      float y0 = local_expert_y[i + 0];
      send_buf_ptr[i + 0] = s * y0;
    }
  }
}

void all2all_combine_pack_send_buffers_fp32(
    hipStream_t stream,
    const int32_t* __restrict__ expert_num_tokens, // shape [num_local_experts]
    const int32_t* __restrict__ expert_meta, // shape [num_local_experts, max_recv, META_DIM]
    const float* __restrict__ expert_y, // shape [num_local_experts, max_recv, hidden_dim]
    float* __restrict__ send_buf0, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf1, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf2, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf3, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf4, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf5, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf6, // shape [max_num_tokens, experts_per_token, hidden_dim]
    float* __restrict__ send_buf7, // shape [max_num_tokens, experts_per_token, hidden_dim]
    int num_local_experts,
    int max_recv,
    int experts_per_token,
    int hidden_dim,
    float s
) {
  // call kernel to do the packing, with a 2D grid of size (max_recv, num_local_experts)
  const int threads_per_block = THREADS_PER_BLOCK_COMBINE_PACK_SEND;
  const dim3 blocks(max_recv, num_local_experts);

  all2all_combine_pack_send_buffers_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
    expert_num_tokens,
    expert_meta,
    expert_y,
    send_buf0,
    send_buf1,
    send_buf2,
    send_buf3,
    send_buf4,
    send_buf5,
    send_buf6,
    send_buf7,
    max_recv,
    experts_per_token,
    hidden_dim,
    s
  );
  // HIP_CHECK(hipGetLastError());
  // HIP_CHECK(hipDeviceSynchronize());
}

#define COMBINE_WRITE_BACK_THREADS_PER_BLOCK 256
#define COMBINE_WRITE_BACK_ELEMENTS_PER_THREAD 4
#define COMBINE_WRITE_BACK_ELEMENTS_PER_BLOCK_LOOP (COMBINE_WRITE_BACK_THREADS_PER_BLOCK * COMBINE_WRITE_BACK_ELEMENTS_PER_THREAD)

void __global__ all2all_combine_write_back_fp32_kernel(
  const float* __restrict__ recv_buf, // shape [num_tokens, num_experts_per_token, hidden_dim]
  const float* __restrict__ weights, // shape [num_tokens, experts_per_token]
  float* __restrict__ output,
  int hidden_dim,
  int experts_per_token
) {
  int token_idx = blockIdx.x;

  // realign input pointers
  recv_buf = &recv_buf[token_idx * experts_per_token * hidden_dim];

  weights = &weights[token_idx * experts_per_token + 0];

  // realign output pointer
  output = &output[token_idx * hidden_dim];

  // combine the expert outputs into output
  int d = COMBINE_WRITE_BACK_ELEMENTS_PER_THREAD * threadIdx.x;
  for (; d + (COMBINE_WRITE_BACK_ELEMENTS_PER_THREAD - 1) < hidden_dim; d += COMBINE_WRITE_BACK_ELEMENTS_PER_BLOCK_LOOP) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (int k = 0; k < experts_per_token; k++) {
      float w = weights[k];
      sum0 += w * recv_buf[k * hidden_dim + d + 0];
      sum1 += w * recv_buf[k * hidden_dim + d + 1];
      sum2 += w * recv_buf[k * hidden_dim + d + 2];
      sum3 += w * recv_buf[k * hidden_dim + d + 3];
    }
    output[d + 0] = sum0;
    output[d + 1] = sum1;
    output[d + 2] = sum2;
    output[d + 3] = sum3;
  }
  // remainder
  for (; d < hidden_dim; d++) {
    float sum = 0.0f;
    for (int k = 0; k < experts_per_token; k++) {
      float w = weights[k];
      sum += w * recv_buf[k * hidden_dim + d];
    }
    output[d] = sum;
  }
}

void all2all_combine_unpack_fp32(
    hipStream_t stream,
    const float* __restrict__ recv_buf, // shape [num_tokens, experts_per_token, hidden_dim]
    const float* __restrict__ weights, // shape [num_tokens, experts_per_token]
    float* __restrict__ output, // shape [max_num_tokens, hidden_dim]
    int hidden_dim,
    int num_tokens,
    int experts_per_token
) {
  const int threads_per_block = COMBINE_WRITE_BACK_THREADS_PER_BLOCK;
  const int blocks = num_tokens;

  if (num_tokens > 0) {
    all2all_combine_write_back_fp32_kernel<<<blocks, threads_per_block, 0, stream>>>(
      recv_buf,
      weights,
      output,
      hidden_dim,
      experts_per_token
    );
    // HIP_CHECK(hipGetLastError());
    // HIP_CHECK(hipDeviceSynchronize());
  }
}

void __global__ all2all_combine_write_back_fp16_kernel(
  const half* __restrict__ recv_buf, // shape [num_tokens, num_experts_per_token, hidden_dim]
  const float* __restrict__ weights, // shape [num_tokens, num_experts_per_token]
  half* __restrict__ output,
  int hidden_dim,
  int experts_per_token
) {
  int token_idx = blockIdx.x;

  // realign input pointers
  recv_buf = &recv_buf[token_idx * experts_per_token * hidden_dim];

  weights = &weights[token_idx * experts_per_token + 0];

  // realign output pointer
  output = &output[token_idx * hidden_dim];

  // combine the expert outputs into output
  int d = COMBINE_WRITE_BACK_ELEMENTS_PER_THREAD * threadIdx.x;
  for (; d + (COMBINE_WRITE_BACK_ELEMENTS_PER_THREAD - 1) < hidden_dim; d += COMBINE_WRITE_BACK_ELEMENTS_PER_BLOCK_LOOP) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (int k = 0; k < experts_per_token; k++) {
      float w = weights[k];
      sum0 += w * __half2float(recv_buf[k * hidden_dim + d + 0]);
      sum1 += w * __half2float(recv_buf[k * hidden_dim + d + 1]);
      sum2 += w * __half2float(recv_buf[k * hidden_dim + d + 2]);
      sum3 += w * __half2float(recv_buf[k * hidden_dim + d + 3]);
    }
    output[d + 0] = __float2half(sum0);
    output[d + 1] = __float2half(sum1);
    output[d + 2] = __float2half(sum2);
    output[d + 3] = __float2half(sum3);
  }
  // remainder
  for (; d < hidden_dim; d++) {
    float sum = 0.0f;
    for (int k = 0; k < experts_per_token; k++) {
      float w = weights[k];
      sum += w * __half2float(recv_buf[k * hidden_dim + d]);
    }
    output[d] = __float2half(sum);
  }
}

void all2all_combine_unpack_fp16(
    hipStream_t stream,
    const half* __restrict__ recv_buf, // shape [total_recv, hidden_sim]
    const float* __restrict__ weights, // shape [num_tokens, experts_per_token]
    half* __restrict__ output, // shape [max_num_tokens, hidden_dim]
    int hidden_dim,
    int num_tokens,
    int experts_per_token
) {
  const int threads_per_block = COMBINE_WRITE_BACK_THREADS_PER_BLOCK;
  const int blocks = num_tokens;

  if (num_tokens > 0) {
    all2all_combine_write_back_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
      recv_buf,
      weights,
      output,
      hidden_dim,
      experts_per_token
    );
    // HIP_CHECK(hipGetLastError());
    // HIP_CHECK(hipDeviceSynchronize());
  }
}
"""


class All2AllCommKernels:
    def __init__(self):
        from torch.utils.cpp_extension import load_inline, _TORCH_PATH

        # limit the architectures to avoid long compile time
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx908,gfx942"

        # measure the time to compile the kernels
        import time

        start_time = time.time()
        # print("Loading All2All comm kernels...")

        try:
            self.comm_kernels = load_inline(
                name="all2all_comm_kernels",
                cpp_sources=[COMM_KERNELS_CPP_CODE],
                cuda_sources=[COMM_KERNELS_CUDA_CODE],
                functions=[
                    "all2all_comm_init",
                    "all2all_comm_destroy",
                    "all2all_comm_dispatch",
                    "all2all_compute",
                    "all2all_comm_combine",
                    "all2all_comm",
                ],
                extra_include_paths=[
                    os.path.join(_TORCH_PATH, "include", "torch", "csrc")
                ],
                with_cuda=True,
                no_implicit_headers=True,
            )
        except Exception as e:
            print(
                "Failed to load All2All comm kernels, falling back to PyTorch implementation."
            )
            print(e)
            self.comm_kernels = None
        end_time = time.time()
        # print(
        #     f"All2All comm kernels loaded in {end_time - start_time:.2f} seconds.",
        #     flush=True,
        # )


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
        # import time

        # start_time = time.time()
        # print(f"Initializing All2AllComm on rank {rank}...")

        self.comm_kernels = get_global_comm_kernels()

        if self.comm_kernels is not None:
            import torch.distributed as dist

            # use custom kernel
            self.comms = self.comm_kernels.all2all_comm_init(
                world_size,
                rank,
                # we use the torch distributed comms to bootstrap our comms
                dist.group.WORLD,
            )
        else:
            self.comms = None

        if self.comms is None:
            raise ValueError("All2AllComm initialization failed.")

        # end_time = time.time()
        # print(
        #     f"All2AllComm initialized in {end_time - start_time:.2f} seconds on rank {rank}.",
        #     flush=True,
        # )

    def destroy(self):
        if self.comm_kernels is not None and self.comms is not None:
            self.comm_kernels.all2all_comm_destroy(self.comms)
            self.comms = None

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
        return self.comm_kernels.all2all_comm_dispatch(
            self.comms,
            dp_x,
            indices,
            num_local_experts,
            max_recv,
        )

    def compute(
        self,
        expert_num_tokens: torch.Tensor,
        expert_x: torch.Tensor,
    ) -> torch.Tensor:
        return self.comm_kernels.all2all_compute(expert_num_tokens, expert_x, self.rank)

    def combine(
        self,
        weights: torch.Tensor,  # topk weight shape (num tokens, num_experts_per_token)
        expert_meta: torch.Tensor,  # input shape (num_local_experts, max_recv, META_DIM)
        expert_y: torch.Tensor,  # input shape (num_local_experts, max_recv, hidden_dim)
        expert_num_tokens: torch.Tensor,
    ) -> torch.Tensor:  # output shape (max num tokens, token dim)
        return self.comm_kernels.all2all_comm_combine(
            self.comms,
            weights,
            expert_meta,
            expert_y,
            expert_num_tokens,
            1.0,  # scale factor
        )

    def all2all(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        num_local_experts: int,
        max_recv: int,
    ) -> torch.Tensor:
        return self.comm_kernels.all2all_comm(
            self.comms,
            x,
            indices,
            weights,
            num_local_experts,
            max_recv,
        )


_global_all2all_comm = None


def get_global_all2all_comm(rank: int, world_size: int):
    global _global_all2all_comm
    if _global_all2all_comm is None:
        _global_all2all_comm = All2AllComm(rank, world_size)
    return _global_all2all_comm


def destroy_global_all2all_comm():
    global _global_all2all_comm
    if _global_all2all_comm is not None:
        _global_all2all_comm.destroy()
        _global_all2all_comm = None


_monkey_patched = False


def apply_monkey_patch():
    # For some reason our comms get corrupted due to the way the benchmarking/testing
    # harness is written (probably destroy_process_group closes a lot of HIP resources including the events we use)
    # so we hook into destroy_process_group to close our comms

    global _monkey_patched

    if _monkey_patched:
        return

    # disable GC to avoid CPU variability during the benchmarking
    import gc

    gc.disable()

    import torch.distributed as dist

    # Save the original function
    _original_destroy = dist.destroy_process_group

    def custom_destroy_process_group(*args, **kwargs):
        destroy_global_all2all_comm()

        # Call the original destroy
        _original_destroy(*args, **kwargs)

    # Monkey-patch it
    dist.destroy_process_group = custom_destroy_process_group
    _monkey_patched = True


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

        apply_monkey_patch()

    # ---------- dispatch ----------

    def dispatch(
        self,
        dp_x: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.comms.dispatch(dp_x, indices, self.num_local_experts, self.max_recv)

    # ---------- combine ----------
    def combine(
        self,
        weights: torch.Tensor,  # topk weight
        expert_meta: torch.Tensor,  # input
        expert_y: torch.Tensor,  # input, (num_local_experts, max_num_tokens * num_dp, token_dim)
        expert_num_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.comms.combine(
            weights=weights,
            expert_meta=expert_meta,
            expert_y=expert_y,
            expert_num_tokens=expert_num_tokens,
        )

    def compute(
        self,
        expert_num_tokens: torch.Tensor,
        expert_x: torch.Tensor,
    ) -> torch.Tensor:
        expert_y = self.comms.compute(expert_num_tokens, expert_x)
        return expert_y

    def all2all(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.comms.all2all(
            x, indices, weights, self.num_local_experts, self.max_recv
        )


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data

    ata = PyTorchAllToAll(cfg, rank, world_size)

    # Having three different methos dispatch, compute and combine incurs CPU overheads
    # due to Python
    # while they were all implemented and correct, for performance we provide a single
    # all2all method that does all three steps in one C++ call

    # Three different calls:
    # expert_num_tokens, expert_x, expert_meta = ata.dispatch(
    #     dp_x=rank_data.x, indices=rank_data.indices
    # )

    # expert_y = ata.compute(
    #     expert_num_tokens=expert_num_tokens,
    #     expert_x=expert_x,
    # )

    # return ata.combine(
    #     weights=rank_data.weights,
    #     expert_meta=expert_meta,
    #     expert_y=expert_y,
    #     expert_num_tokens=expert_num_tokens,
    # )

    # Everything combined into a single call:
    ata_comms = ata.comms
    return ata_comms.comm_kernels.all2all_comm(
        ata_comms.comms,
        rank_data.x,
        rank_data.indices,
        rank_data.weights,
        ata.num_local_experts,
        ata.max_recv,
    )
