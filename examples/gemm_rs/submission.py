import os
from typing import Optional, Tuple
from task import input_t, output_t
import torch


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

muillm_comm_error_t muillm_comm_p2p_destroy_comm(
    muillm_comm_p2p_t* comm
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

muillm_comm_error_t muillm_comm_p2p_destroy_comm(
    muillm_comm_p2p_t* comm
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

  // gpu barrier
  if ((error = __mui_gpu_barrier(comm, /*stream*/ 0)) != MUILLM_COMM_SUCCESS) {
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

  // free buffer sets
  if ((error = __free_buffer_set(comm, comm->first_buffers, /*sync*/ false)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to free first buffer set"<<std::endl;
    return error;
  }
  delete comm->first_buffers;

  if ((error = __free_buffer_set(comm, comm->second_buffers, /*sync*/ false)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to free second buffer set"<<std::endl;
    return error;
  }
  delete comm->second_buffers;

  // free signal memory
  if (comm->signal_host != nullptr) {
    __deallocate_locked_shared_cpu_mem(
      comm,
      comm->signal_host
    );
    comm->signal_host = nullptr;
    comm->signal = nullptr;
  }

  // destroy cache flush event
  if (hipEventDestroy(comm->cache_flush_event) != hipSuccess) {
    std::cout<<"rank "<<local_rank<<" failed to destroy cache flush event"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
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
        std::cout<<"rank "<<local_rank<<" gpu barrier failed because hipEventRecord failed"<<std::endl;
        hipError_t err = hipGetLastError();
        const char* errStr = hipGetErrorString(err);
        std::cout<<"Last HIP error: "<<errStr<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }

    // write the values
    if ((muillm_error = __mui_stream_inc_wait_value(stream, comm->signal, seq_no)) != MUILLM_COMM_SUCCESS) {
      std::cout<<"rank "<<local_rank<<" gpu barrier failed because __mui_stream_inc_wait_value failed"<<std::endl;
      return muillm_error;
    }
  } else {
    std::cout<<"rank "<<local_rank<<" gpu barrier failed because there is no signal memory"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}

muillm_comm_error_t __muillm_gpu_copy(void* dst, const void* src, size_t count, hipStream_t stream);

muillm_comm_error_t __muillm_scatter_all(
  hipStream_t stream,
  // inputs
  const void* src,
  int scattered_size_bytes,
  int local_size,
  int local_rank,
  // outputs
  void* dst0,
  void* dst1,
  void* dst2,
  void* dst3,
  void* dst4,
  void* dst5,
  void* dst6,
  void* dst7
);

muillm_comm_error_t __muillm_reduce(
  hipStream_t stream,
  // inputs
  const void* src, // shape [local_size, scattered_M, N]
  int scattered_count,
  int local_size,
  muillm_comm_datatype_t datatype,
  // outputs
  void* dst
);

muillm_comm_error_t muillm_comm_reduce_scatter_ll(
  muillm_comm_p2p_t* comm,
  hipStream_t stream,
  void* input_buffer,
  int M,
  int scattered_M,
  int N,
  muillm_comm_datatype_t datatype,
  void* output_buffer
) {
  muillm_comm_error_t error;

  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  int scattered_count = scattered_M * N;
  int scattered_size = __comm_size(datatype, scattered_count);

  //
  // First make sure we have enough buffer space
  //
  int capacity = scattered_size * local_size;

  muillm_comm_p2p_buffer_set_t* buffer_set;
  if ((error = muillm_comm_p2p_get_buffer_set(comm, capacity, &buffer_set, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to get buffer set"<<std::endl;
    return error;
  }

  // Copy our data to the other GPUs buffers
  if ((error = __muillm_scatter_all(
    stream,
    input_buffer,
    scattered_size,
    local_size,
    local_rank,
    buffer_set->buffers[0],
    buffer_set->buffers[1],
    buffer_set->buffers[2],
    buffer_set->buffers[3],
    buffer_set->buffers[4],
    buffer_set->buffers[5],
    buffer_set->buffers[6],
    buffer_set->buffers[7]
  )) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to scatter data to other GPUs"<<std::endl;
    return error;
  }

  // Synchronize GPUs
  if (__mui_gpu_barrier(comm, stream) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when doing gpu barrier");
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  // Finally reduce the data on each GPU
  if ((error = __muillm_reduce(
    stream,
    buffer_set->buffers[local_rank],
    scattered_count,
    local_size,
    datatype,
    output_buffer
  )) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to reduce data"<<std::endl;
    return error;
  }

  return MUILLM_COMM_SUCCESS;
}

#define MUILLM_REDUCE_SCATTER_LL_TRESHOLD (16 * 1024 * 1024) // 16M elements

// torch extension

#include <tuple>

#include <ATen/cuda/CUDAContext.h>

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

// output shape [M, N]
torch::Tensor all2all_comm_gemm_reduce_scatter(
  void* comms,
  torch::Tensor& input, // shape [M, local_K]
  torch::Tensor& weights, // shape [N, local_K]
  std::optional<torch::Tensor> bias // shape [N]
) {
  muillm_comm_p2p_t* comm = (muillm_comm_p2p_t*) comms;

  CHECK_INPUT(input);
  CHECK_INPUT(weights);

  auto device = input.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  int M = input.size(0);
  int N = weights.size(0);
  int K = input.size(1);

  int scattered_M = M / local_size;

  auto dtype = input.dtype();

  muillm_comm_datatype_t datatype;
  if (dtype == torch::kFloat16) {
    datatype = MUILLM_COMM_FP16;
  } else if (dtype == torch::kBFloat16) {
    datatype = MUILLM_COMM_BF16;
  } else {
    TORCH_CHECK(false, "unsupported data type");
    return torch::Tensor();
  }

  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);
                          
  torch::Tensor output = torch::empty({M, N}, output_options); // shape [M, N]
  
  torch::linear_out(output, input, weights, bias); // shape [M, N]
        
  auto rs_output = torch::empty({scattered_M, N}, output_options);

  int total_size = scattered_M * N;
  if (true) { // total_size <= MUILLM_REDUCE_SCATTER_LL_TRESHOLD) {
    // use our custom reduce-scatter implementation
    if (muillm_comm_reduce_scatter_ll(
      comm,
      stream,
      output.data_ptr(),
      M,
      scattered_M,
      N,
      datatype,
      rs_output.data_ptr()
    ) != MUILLM_COMM_SUCCESS) {
      TORCH_CHECK(false, "an error happened when doing reduce-scatter");
    }
  } else {
    std::vector<torch::Tensor> rs_output_vector = {rs_output};
    std::vector<torch::Tensor> input_vector = {output};
    comm->process_group->reduce_scatter_tensor_coalesced(
      rs_output_vector,
      input_vector,
      c10d::ReduceScatterOptions()
    );
  }

  return rs_output;
}

muillm_comm_error_t __muillm_reduce_pull(
  hipStream_t stream,
  // inputs
  const void* src0, // shape [M, N] needs to be offset
  const void* src1, // shape [M, N] needs to be offset
  const void* src2, // shape [M, N] needs to be offset
  const void* src3, // shape [M, N] needs to be offset
  const void* src4, // shape [M, N] needs to be offset
  const void* src5, // shape [M, N] needs to be offset
  const void* src6, // shape [M, N] needs to be offset
  const void* src7, // shape [M, N] needs to be offset
  int scattered_count,
  int local_size,
  int local_rank,
  muillm_comm_datatype_t datatype,
  // outputs
  void* dst // shape [scattered_M, N]
);

torch::Tensor all2all_comm_gemm_reduce_scatter_pull(
  void* comms,
  torch::Tensor& input, // shape [M, local_K]
  torch::Tensor& weights, // shape [N, local_K]
  std::optional<torch::Tensor> bias // shape [N]
) {
  muillm_comm_p2p_t* comm = (muillm_comm_p2p_t*) comms;
  muillm_comm_error_t error;

  CHECK_INPUT(input);
  CHECK_INPUT(weights);

  auto device = input.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  int M = input.size(0);
  int N = weights.size(0);
  int K = input.size(1);

  int scattered_M = M / local_size;

  auto dtype = input.dtype();

  muillm_comm_datatype_t datatype;
  if (dtype == torch::kFloat16) {
    datatype = MUILLM_COMM_FP16;
  } else if (dtype == torch::kBFloat16) {
    datatype = MUILLM_COMM_BF16;
  } else {
    TORCH_CHECK(false, "unsupported data type");
    return torch::Tensor();
  }

  int scattered_count = scattered_M * N;

  //
  // First make sure we have enough buffer space
  //
  int capacity = __comm_size(datatype, M * N); // need to hold the full output

  muillm_comm_p2p_buffer_set_t* buffer_set;
  if ((error = muillm_comm_p2p_get_buffer_set(comm, capacity, &buffer_set, stream)) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to get buffer set"<<std::endl;
    TORCH_CHECK(false, "an error happened when getting buffer set");
    return torch::Tensor();
  }

  // then we compute the local GEMM placing the output in the send/recv buffers

  auto output_options = at::TensorOptions()
                            .dtype(dtype)
                            .layout(at::kStrided)
                            .device(device) // same output device as inputs
                            .requires_grad(false);

  torch::Tensor output = torch::from_blob(buffer_set->buffers[local_rank], {M, N}, output_options); // shape [M, N]

  torch::linear_out(output, input, weights, bias); // shape [M, N]

  // Synchronize GPUs
  if (__mui_gpu_barrier(comm, stream) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "an error happened when doing gpu barrier");
    return torch::Tensor();
  }

  auto rs_output = torch::empty({scattered_M, N}, output_options);

  // Finally gather/reduce the data on each GPU
  if ((error = __muillm_reduce_pull(
    stream,
    buffer_set->buffers[0],
    buffer_set->buffers[1],
    buffer_set->buffers[2],
    buffer_set->buffers[3],
    buffer_set->buffers[4],
    buffer_set->buffers[5],
    buffer_set->buffers[6],
    buffer_set->buffers[7],
    scattered_count,
    local_size,
    local_rank,
    datatype,
    rs_output.data_ptr() // shape [scattered_M, N]
  )) != MUILLM_COMM_SUCCESS) {
    std::cout<<"rank "<<local_rank<<" failed to reduce_pull data"<<std::endl;
    TORCH_CHECK(false, "an error happened when doing reduce_pull");
    return torch::Tensor();
  }

  return rs_output;
}
"""

COMM_KERNELS_CUDA_CODE = """
#include <stdint.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <iostream>
#include <algorithm>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != cudaSuccess) { \
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
#define ALIGN_UP(a, b) (DIV_ROUND_UP(a, b) * (b))

typedef struct half8 {
  half x, y, z, w, a, b, c, d;
} half8;

typedef struct half4 {
  half x, y, z, w;
} half4;

typedef struct __hip_bfloat164 {
  __hip_bfloat16 x, y, z, w;
} __hip_bfloat164;

typedef struct __hip_bfloat168 {
  __hip_bfloat16 x, y, z, w, a, b, c, d;
} __hip_bfloat168;


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
#define BYTES_PER_BLOCK_LOOP (THREADS_PER_BLOCK * BYTES_PER_THREAD)

typedef struct uint32x4{
uint32_t x, y, z, w;
} uint32x4_t;

__global__ void __muillm_copy_p2p_kernel(
  const uint8_t* src_ptr,
  uint8_t* dst_ptr,
  unsigned N
) {
  unsigned i = blockIdx.x * BYTES_PER_BLOCK_LOOP + (threadIdx.x * BYTES_PER_THREAD);
  if (i + (BYTES_PER_THREAD - 1) < N) {
    // can copy 16 bytes

    const uint32x4_t* src_x16_ptr = (const uint32x4_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr = (uint32x4_t*)(&dst_ptr[i]);
    *dst_x16_ptr = *src_x16_ptr;

    i += BYTES_PER_BLOCK_LOOP;
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
  const int num_blocks = DIV_ROUND_UP(count, BYTES_PER_BLOCK_LOOP);

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

void __global__ scatter_all_tp8_kernel(
  const uint8_t* src,
  uint8_t* dst0,
  uint8_t* dst1,
  uint8_t* dst2,
  uint8_t* dst3,
  uint8_t* dst4,
  uint8_t* dst5,
  uint8_t* dst6,
  uint8_t* dst7,
  int bytes_per_block,
  int N,
  int local_rank
) {
  unsigned block_idx = blockIdx.y;

  unsigned block_start = blockIdx.x * bytes_per_block;
  unsigned block_end = std::min(block_start + bytes_per_block, N);
  unsigned i = block_start + (threadIdx.x * BYTES_PER_THREAD);

  uint8_t* dsts[8] = {dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7};
  uint8_t* dst = dsts[block_idx];

  // realign src and dst pointers according to the block we are working on
  src += block_idx * N;
  dst += local_rank * N;

  // TODO: unroll more, e.g. 4x more
  for (; i + (BYTES_PER_THREAD - 1) < block_end; i += BYTES_PER_BLOCK_LOOP) {
    // can copy 16 bytes, which is 4kB per iteration
    const uint32x4_t* src_x16_ptr = (const uint32x4_t*)(&src[i]);
    uint32x4_t* dst_x16_ptr = (uint32x4_t*)(&dst[i]);

    uint32x4_t v = *src_x16_ptr;
    *dst_x16_ptr = v;
  }
  
  // loop remainder
  if (i < block_end) {
    // only one thread will execute this at max
    // non vectorized copy
    for (unsigned b = 0; b < BYTES_PER_THREAD; b++) {
      if (i < block_end) {
        uint8_t v = src[i];
        dst[i] = v;
        i++;
      }
    }
  }
}

#define MAX_REDUCE_X_BLOCKS 8

muillm_comm_error_t __muillm_scatter_all(
  hipStream_t stream,
  // inputs
  const void* src,
  int scattered_size_bytes,
  int local_size,
  int local_rank,
  // outputs
  void* dst0,
  void* dst1,
  void* dst2,
  void* dst3,
  void* dst4,
  void* dst5,
  void* dst6,
  void* dst7
) {

  const int threads_per_blocks = THREADS_PER_BLOCK;
  // we want to avoid spawning too many blocks to copy the data and want instead
  // to make blocks process more data when we have more than MAX_REDUCE_X_BLOCKS
  //
  int num_small_x_blocks = DIV_ROUND_UP(scattered_size_bytes, BYTES_PER_BLOCK_LOOP);
  int num_x_blocks = std::min(num_small_x_blocks, MAX_REDUCE_X_BLOCKS);
  const dim3 num_blocks = dim3(num_x_blocks, local_size);

  int bytes_per_block = ALIGN_UP(DIV_ROUND_UP(scattered_size_bytes, num_x_blocks), 4096);

  scatter_all_tp8_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint8_t*) src,
    (uint8_t*) dst0,
    (uint8_t*) dst1,
    (uint8_t*) dst2,
    (uint8_t*) dst3,
    (uint8_t*) dst4,
    (uint8_t*) dst5,
    (uint8_t*) dst6,
    (uint8_t*) dst7,
    bytes_per_block,
    scattered_size_bytes,
    local_rank
  );

  return MUILLM_COMM_SUCCESS;
}

#define REDUCE_PER_THREAD 8
#define REDUCE_PER_BLOCK (THREADS_PER_BLOCK * REDUCE_PER_THREAD)

void __global__ reduce_x8_fp16_kernel(
  const half* __restrict__ src0, // shape [scattered_M, N]
  const half* __restrict__ src1, // shape [scattered_M, N]
  const half* __restrict__ src2, // shape [scattered_M, N]
  const half* __restrict__ src3, // shape [scattered_M, N]
  const half* __restrict__ src4, // shape [scattered_M, N]
  const half* __restrict__ src5, // shape [scattered_M, N]
  const half* __restrict__ src6, // shape [scattered_M, N]
  const half* __restrict__ src7, // shape [scattered_M, N]
  half* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    half8* dst_h8_ptr = (half8*)(&dst[i]);

    const half8* src0_h8_ptr = (const half8*)(&src0[i]);
    const half8* src1_h8_ptr = (const half8*)(&src1[i]);
    const half8* src2_h8_ptr = (const half8*)(&src2[i]);
    const half8* src3_h8_ptr = (const half8*)(&src3[i]);
    const half8* src4_h8_ptr = (const half8*)(&src4[i]);
    const half8* src5_h8_ptr = (const half8*)(&src5[i]);
    const half8* src6_h8_ptr = (const half8*)(&src6[i]);
    const half8* src7_h8_ptr = (const half8*)(&src7[i]);

    half8 v0 = *src0_h8_ptr;
    half8 v1 = *src1_h8_ptr;
    half8 v2 = *src2_h8_ptr;
    half8 v3 = *src3_h8_ptr;
    half8 v4 = *src4_h8_ptr;
    half8 v5 = *src5_h8_ptr;
    half8 v6 = *src6_h8_ptr;
    half8 v7 = *src7_h8_ptr;

    half8 r;
    r.x = __hadd(v0.x, __hadd(v1.x, __hadd(v2.x, __hadd(v3.x, __hadd(v4.x, __hadd(v5.x, __hadd(v6.x, v7.x)))))));
    r.y = __hadd(v0.y, __hadd(v1.y, __hadd(v2.y, __hadd(v3.y, __hadd(v4.y, __hadd(v5.y, __hadd(v6.y, v7.y)))))));
    r.z = __hadd(v0.z, __hadd(v1.z, __hadd(v2.z, __hadd(v3.z, __hadd(v4.z, __hadd(v5.z, __hadd(v6.z, v7.z)))))));
    r.w = __hadd(v0.w, __hadd(v1.w, __hadd(v2.w, __hadd(v3.w, __hadd(v4.w, __hadd(v5.w, __hadd(v6.w, v7.w)))))));
    r.a = __hadd(v0.a, __hadd(v1.a, __hadd(v2.a, __hadd(v3.a, __hadd(v4.a, __hadd(v5.a, __hadd(v6.a, v7.a)))))));
    r.b = __hadd(v0.b, __hadd(v1.b, __hadd(v2.b, __hadd(v3.b, __hadd(v4.b, __hadd(v5.b, __hadd(v6.b, v7.b)))))));
    r.c = __hadd(v0.c, __hadd(v1.c, __hadd(v2.c, __hadd(v3.c, __hadd(v4.c, __hadd(v5.c, __hadd(v6.c, v7.c)))))));
    r.d = __hadd(v0.d, __hadd(v1.d, __hadd(v2.d, __hadd(v3.d, __hadd(v4.d, __hadd(v5.d, __hadd(v6.d, v7.d)))))));

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        half v0 = src0[i];
        half v1 = src1[i];
        half v2 = src2[i];
        half v3 = src3[i];
        half v4 = src4[i];
        half v5 = src5[i];
        half v6 = src6[i];
        half v7 = src7[i];

        half r = __hadd(v0, __hadd(v1, __hadd(v2, __hadd(v3, __hadd(v4, __hadd(v5, __hadd(v6, v7)))))));

        dst[i] = r;
        i++;
      }
    }
  }
}

void __global__ reduce_x4_fp16_kernel(
  const half* __restrict__ src0, // shape [scattered_M, N]
  const half* __restrict__ src1, // shape [scattered_M, N]
  const half* __restrict__ src2, // shape [scattered_M, N]
  const half* __restrict__ src3, // shape [scattered_M, N]
  half* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    half8* dst_h8_ptr = (half8*)(&dst[i]);

    const half8* src0_h8_ptr = (const half8*)(&src0[i]);
    const half8* src1_h8_ptr = (const half8*)(&src1[i]);
    const half8* src2_h8_ptr = (const half8*)(&src2[i]);
    const half8* src3_h8_ptr = (const half8*)(&src3[i]);

    half8 v0 = *src0_h8_ptr;
    half8 v1 = *src1_h8_ptr;
    half8 v2 = *src2_h8_ptr;
    half8 v3 = *src3_h8_ptr;

    half8 r;
    r.x = __hadd(v0.x, __hadd(v1.x, __hadd(v2.x, v3.x)));
    r.y = __hadd(v0.y, __hadd(v1.y, __hadd(v2.y, v3.y)));
    r.z = __hadd(v0.z, __hadd(v1.z, __hadd(v2.z, v3.z)));
    r.w = __hadd(v0.w, __hadd(v1.w, __hadd(v2.w, v3.w)));
    r.a = __hadd(v0.a, __hadd(v1.a, __hadd(v2.a, v3.a)));
    r.b = __hadd(v0.b, __hadd(v1.b, __hadd(v2.b, v3.b)));
    r.c = __hadd(v0.c, __hadd(v1.c, __hadd(v2.c, v3.c)));
    r.d = __hadd(v0.d, __hadd(v1.d, __hadd(v2.d, v3.d)));

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        half v0 = src0[i];
        half v1 = src1[i];
        half v2 = src2[i];
        half v3 = src3[i];

        half r = __hadd(v0, __hadd(v1, __hadd(v2, v3)));

        dst[i] = r;
        i++;
      }
    }
  }
}


void __global__ reduce_x2_fp16_kernel(
  const half* __restrict__ src0, // shape [scattered_M, N]
  const half* __restrict__ src1, // shape [scattered_M, N]
  half* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    half8* dst_h8_ptr = (half8*)(&dst[i]);

    const half8* src0_h8_ptr = (const half8*)(&src0[i]);
    const half8* src1_h8_ptr = (const half8*)(&src1[i]);

    half8 v0 = *src0_h8_ptr;
    half8 v1 = *src1_h8_ptr;

    half8 r;
    r.x = __hadd(v0.x, v1.x);
    r.y = __hadd(v0.y, v1.y);
    r.z = __hadd(v0.z, v1.z);
    r.w = __hadd(v0.w, v1.w);
    r.a = __hadd(v0.a, v1.a);
    r.b = __hadd(v0.b, v1.b);
    r.c = __hadd(v0.c, v1.c);
    r.d = __hadd(v0.d, v1.d);

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        half v0 = src0[i];
        half v1 = src1[i];

        half r = __hadd(v0, v1);

        dst[i] = r;
        i++;
      }
    }
  }
}

muillm_comm_error_t __muillm_reduce_fp16(
  hipStream_t stream,
  // inputs
  const half* src, // shape [local_size, scattered_M, N]
  int scattered_count,
  int local_size,
  // outputs
  half* dst
) {

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(scattered_count, REDUCE_PER_BLOCK);

  if (local_size == 8) {
    // compute the src pointers by applying the offsets
    const half* src0 = src + (0 * scattered_count);
    const half* src1 = src + (1 * scattered_count);
    const half* src2 = src + (2 * scattered_count);
    const half* src3 = src + (3 * scattered_count);
    const half* src4 = src + (4 * scattered_count);
    const half* src5 = src + (5 * scattered_count);
    const half* src6 = src + (6 * scattered_count);
    const half* src7 = src + (7 * scattered_count);

    reduce_x8_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      src4,
      src5,
      src6,
      src7,
      dst,
      scattered_count
    );
  } else if (local_size == 4) {
    // compute the src pointers by applying the offsets
    const half* src0 = src + (0 * scattered_count);
    const half* src1 = src + (1 * scattered_count);
    const half* src2 = src + (2 * scattered_count);
    const half* src3 = src + (3 * scattered_count);

    reduce_x4_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      dst,
      scattered_count
    );
  } else if (local_size == 2) {
    // compute the src pointers by applying the offsets
    const half* src0 = src + (0 * scattered_count);
    const half* src1 = src + (1 * scattered_count);

    reduce_x2_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      dst,
      scattered_count
    );
  } else {
    return MUILLM_COMM_UNSUPPORTED_SIZE;
  }

  return MUILLM_COMM_SUCCESS;
}


void __global__ reduce_x8_bf16_kernel(
  const __hip_bfloat16* __restrict__ src0, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src1, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src2, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src3, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src4, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src5, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src6, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src7, // shape [scattered_M, N]
  __hip_bfloat16* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    __hip_bfloat168* dst_h8_ptr = (__hip_bfloat168*)(&dst[i]);

    const __hip_bfloat168* src0_h8_ptr = (const __hip_bfloat168*)(&src0[i]);
    const __hip_bfloat168* src1_h8_ptr = (const __hip_bfloat168*)(&src1[i]);
    const __hip_bfloat168* src2_h8_ptr = (const __hip_bfloat168*)(&src2[i]);
    const __hip_bfloat168* src3_h8_ptr = (const __hip_bfloat168*)(&src3[i]);
    const __hip_bfloat168* src4_h8_ptr = (const __hip_bfloat168*)(&src4[i]);
    const __hip_bfloat168* src5_h8_ptr = (const __hip_bfloat168*)(&src5[i]);
    const __hip_bfloat168* src6_h8_ptr = (const __hip_bfloat168*)(&src6[i]);
    const __hip_bfloat168* src7_h8_ptr = (const __hip_bfloat168*)(&src7[i]);

    __hip_bfloat168 v0 = *src0_h8_ptr;
    __hip_bfloat168 v1 = *src1_h8_ptr;
    __hip_bfloat168 v2 = *src2_h8_ptr;
    __hip_bfloat168 v3 = *src3_h8_ptr;
    __hip_bfloat168 v4 = *src4_h8_ptr;
    __hip_bfloat168 v5 = *src5_h8_ptr;
    __hip_bfloat168 v6 = *src6_h8_ptr;
    __hip_bfloat168 v7 = *src7_h8_ptr;

    __hip_bfloat168 r;
    r.x = __hadd(v0.x, __hadd(v1.x, __hadd(v2.x, __hadd(v3.x, __hadd(v4.x, __hadd(v5.x, __hadd(v6.x, v7.x)))))));
    r.y = __hadd(v0.y, __hadd(v1.y, __hadd(v2.y, __hadd(v3.y, __hadd(v4.y, __hadd(v5.y, __hadd(v6.y, v7.y)))))));
    r.z = __hadd(v0.z, __hadd(v1.z, __hadd(v2.z, __hadd(v3.z, __hadd(v4.z, __hadd(v5.z, __hadd(v6.z, v7.z)))))));
    r.w = __hadd(v0.w, __hadd(v1.w, __hadd(v2.w, __hadd(v3.w, __hadd(v4.w, __hadd(v5.w, __hadd(v6.w, v7.w)))))));
    r.a = __hadd(v0.a, __hadd(v1.a, __hadd(v2.a, __hadd(v3.a, __hadd(v4.a, __hadd(v5.a, __hadd(v6.a, v7.a)))))));
    r.b = __hadd(v0.b, __hadd(v1.b, __hadd(v2.b, __hadd(v3.b, __hadd(v4.b, __hadd(v5.b, __hadd(v6.b, v7.b)))))));
    r.c = __hadd(v0.c, __hadd(v1.c, __hadd(v2.c, __hadd(v3.c, __hadd(v4.c, __hadd(v5.c, __hadd(v6.c, v7.c)))))));
    r.d = __hadd(v0.d, __hadd(v1.d, __hadd(v2.d, __hadd(v3.d, __hadd(v4.d, __hadd(v5.d, __hadd(v6.d, v7.d)))))));

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        __hip_bfloat16 v0 = src0[i];
        __hip_bfloat16 v1 = src1[i];
        __hip_bfloat16 v2 = src2[i];
        __hip_bfloat16 v3 = src3[i];
        __hip_bfloat16 v4 = src4[i];
        __hip_bfloat16 v5 = src5[i];
        __hip_bfloat16 v6 = src6[i];
        __hip_bfloat16 v7 = src7[i];

        __hip_bfloat16 r = __hadd(v0, __hadd(v1, __hadd(v2, __hadd(v3, __hadd(v4, __hadd(v5, __hadd(v6, v7)))))));

        dst[i] = r;
        i++;
      }
    }
  }
}

void __global__ reduce_x4_bf16_kernel(
  const __hip_bfloat16* __restrict__ src0, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src1, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src2, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src3, // shape [scattered_M, N]
  __hip_bfloat16* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    __hip_bfloat168* dst_h8_ptr = (__hip_bfloat168*)(&dst[i]);

    const __hip_bfloat168* src0_h8_ptr = (const __hip_bfloat168*)(&src0[i]);
    const __hip_bfloat168* src1_h8_ptr = (const __hip_bfloat168*)(&src1[i]);
    const __hip_bfloat168* src2_h8_ptr = (const __hip_bfloat168*)(&src2[i]);
    const __hip_bfloat168* src3_h8_ptr = (const __hip_bfloat168*)(&src3[i]);

    __hip_bfloat168 v0 = *src0_h8_ptr;
    __hip_bfloat168 v1 = *src1_h8_ptr;
    __hip_bfloat168 v2 = *src2_h8_ptr;
    __hip_bfloat168 v3 = *src3_h8_ptr;

    __hip_bfloat168 r;
    r.x = __hadd(v0.x, __hadd(v1.x, __hadd(v2.x, v3.x)));
    r.y = __hadd(v0.y, __hadd(v1.y, __hadd(v2.y, v3.y)));
    r.z = __hadd(v0.z, __hadd(v1.z, __hadd(v2.z, v3.z)));
    r.w = __hadd(v0.w, __hadd(v1.w, __hadd(v2.w, v3.w)));
    r.a = __hadd(v0.a, __hadd(v1.a, __hadd(v2.a, v3.a)));
    r.b = __hadd(v0.b, __hadd(v1.b, __hadd(v2.b, v3.b)));
    r.c = __hadd(v0.c, __hadd(v1.c, __hadd(v2.c, v3.c)));
    r.d = __hadd(v0.d, __hadd(v1.d, __hadd(v2.d, v3.d)));

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        __hip_bfloat16 v0 = src0[i];
        __hip_bfloat16 v1 = src1[i];
        __hip_bfloat16 v2 = src2[i];
        __hip_bfloat16 v3 = src3[i];

        __hip_bfloat16 r = __hadd(v0, __hadd(v1, __hadd(v2, v3)));

        dst[i] = r;
        i++;
      }
    }
  }
}


void __global__ reduce_x2_bf16_kernel(
  const __hip_bfloat16* __restrict__ src0, // shape [scattered_M, N]
  const __hip_bfloat16* __restrict__ src1, // shape [scattered_M, N]
  __hip_bfloat16* __restrict__ dst,
  int N
) {

  unsigned i = blockIdx.x * REDUCE_PER_BLOCK + (threadIdx.x * REDUCE_PER_THREAD);
  // TODO: make copy more like for scatter
  if (i + (REDUCE_PER_THREAD - 1) < N) {
    // can reduce 8 elements
    __hip_bfloat168* dst_h8_ptr = (__hip_bfloat168*)(&dst[i]);

    const __hip_bfloat168* src0_h8_ptr = (const __hip_bfloat168*)(&src0[i]);
    const __hip_bfloat168* src1_h8_ptr = (const __hip_bfloat168*)(&src1[i]);

    __hip_bfloat168 v0 = *src0_h8_ptr;
    __hip_bfloat168 v1 = *src1_h8_ptr;

    __hip_bfloat168 r;
    r.x = __hadd(v0.x, v1.x);
    r.y = __hadd(v0.y, v1.y);
    r.z = __hadd(v0.z, v1.z);
    r.w = __hadd(v0.w, v1.w);
    r.a = __hadd(v0.a, v1.a);
    r.b = __hadd(v0.b, v1.b);
    r.c = __hadd(v0.c, v1.c);
    r.d = __hadd(v0.d, v1.d);

    *dst_h8_ptr = r;
  } else {
    // non vectorized reduce
    for (unsigned r = 0; r < REDUCE_PER_THREAD; r++) {
      if (i < N) {
        __hip_bfloat16 v0 = src0[i];
        __hip_bfloat16 v1 = src1[i];

        __hip_bfloat16 r = __hadd(v0, v1);

        dst[i] = r;
        i++;
      }
    }
  }
}

muillm_comm_error_t __muillm_reduce_bf16(
  hipStream_t stream,
  // inputs
  const __hip_bfloat16* src, // shape [local_size, scattered_M, N]
  int scattered_count,
  int local_size,
  // outputs
  __hip_bfloat16* dst
) {

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(scattered_count, REDUCE_PER_BLOCK);

  if (local_size == 8) {
    // compute the src pointers by applying the offsets
    const __hip_bfloat16* src0 = src + (0 * scattered_count);
    const __hip_bfloat16* src1 = src + (1 * scattered_count);
    const __hip_bfloat16* src2 = src + (2 * scattered_count);
    const __hip_bfloat16* src3 = src + (3 * scattered_count);
    const __hip_bfloat16* src4 = src + (4 * scattered_count);
    const __hip_bfloat16* src5 = src + (5 * scattered_count);
    const __hip_bfloat16* src6 = src + (6 * scattered_count);
    const __hip_bfloat16* src7 = src + (7 * scattered_count);

    reduce_x8_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      src4,
      src5,
      src6,
      src7,
      dst,
      scattered_count
    );
  } else if (local_size == 4) {
    // compute the src pointers by applying the offsets
    const __hip_bfloat16* src0 = src + (0 * scattered_count);
    const __hip_bfloat16* src1 = src + (1 * scattered_count);
    const __hip_bfloat16* src2 = src + (2 * scattered_count);
    const __hip_bfloat16* src3 = src + (3 * scattered_count);

    reduce_x4_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      dst,
      scattered_count
    );
  } else if (local_size == 2) {
    // compute the src pointers by applying the offsets
    const __hip_bfloat16* src0 = src + (0 * scattered_count);
    const __hip_bfloat16* src1 = src + (1 * scattered_count);

    reduce_x2_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      dst,
      scattered_count
    );
  } else {
    return MUILLM_COMM_UNSUPPORTED_SIZE;
  }

  return MUILLM_COMM_SUCCESS;
}


muillm_comm_error_t __muillm_reduce(
  hipStream_t stream,
  // inputs
  const void* src, // shape [local_size, scattered_M, N]
  int scattered_count,
  int local_size,
  muillm_comm_datatype_t datatype,
  // outputs
  void* dst
) {
  if (datatype == MUILLM_COMM_FP16) {
    return __muillm_reduce_fp16(
      stream,
      (const half*) src,
      scattered_count,
      local_size,
      (half*) dst
    );
  } else if (datatype == MUILLM_COMM_BF16) {
    return __muillm_reduce_bf16(
      stream,
      (const __hip_bfloat16*) src,
      scattered_count,
      local_size,
      (__hip_bfloat16*) dst
    );
  } else {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
}

muillm_comm_error_t __muillm_reduce_pull_fp16(
  hipStream_t stream,
  // inputs
  const half* src0, // shape [M, N] needs to be offset
  const half* src1, // shape [M, N] needs to be offset
  const half* src2, // shape [M, N] needs to be offset
  const half* src3, // shape [M, N] needs to be offset
  const half* src4, // shape [M, N] needs to be offset
  const half* src5, // shape [M, N] needs to be offset
  const half* src6, // shape [M, N] needs to be offset
  const half* src7, // shape [M, N] needs to be offset
  int scattered_count,
  int local_size,
  int local_rank,
  // outputs
  half* dst
) {

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(scattered_count, REDUCE_PER_BLOCK);

  int offset = (local_rank * scattered_count);
  if (local_size == 8) {
    // compute the src pointers by applying the offsets
    src0 += offset;
    src1 += offset;
    src2 += offset;
    src3 += offset;
    src4 += offset;
    src5 += offset;
    src6 += offset;
    src7 += offset;

    reduce_x8_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      src4,
      src5,
      src6,
      src7,
      dst,
      scattered_count
    );
  } else if (local_size == 4) {
    // compute the src pointers by applying the offsets
    src0 += offset;
    src1 += offset;
    src2 += offset;
    src3 += offset;

    reduce_x4_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      dst,
      scattered_count
    );
  } else if (local_size == 2) {
    // compute the src pointers by applying the offsets
    src0 += offset;
    src1 += offset;

    reduce_x2_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      dst,
      scattered_count
    );
  } else {
    return MUILLM_COMM_UNSUPPORTED_SIZE;
  }

  return MUILLM_COMM_SUCCESS;
}


muillm_comm_error_t __muillm_reduce_pull_bf16(
  hipStream_t stream,
  // inputs
  const __hip_bfloat16* src0, // shape [M, N] needs to be offset
  const __hip_bfloat16* src1, // shape [M, N] needs to be offset
  const __hip_bfloat16* src2, // shape [M, N] needs to be offset
  const __hip_bfloat16* src3, // shape [M, N] needs to be offset
  const __hip_bfloat16* src4, // shape [M, N] needs to be offset
  const __hip_bfloat16* src5, // shape [M, N] needs to be offset
  const __hip_bfloat16* src6, // shape [M, N] needs to be offset
  const __hip_bfloat16* src7, // shape [M, N] needs to be offset
  int scattered_count,
  int local_size,
  int local_rank,
  // outputs
  __hip_bfloat16* dst
) {

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(scattered_count, REDUCE_PER_BLOCK);

  int offset = (local_rank * scattered_count);
  if (local_size == 8) {
    // compute the src pointers by applying the offsets
    src0 += offset;
    src1 += offset;
    src2 += offset;
    src3 += offset;
    src4 += offset;
    src5 += offset;
    src6 += offset;
    src7 += offset;

    reduce_x8_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      src4,
      src5,
      src6,
      src7,
      dst,
      scattered_count
    );
  } else if (local_size == 4) {
    // compute the src pointers by applying the offsets
    src0 += offset;
    src1 += offset;
    src2 += offset;
    src3 += offset;

    reduce_x4_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      src2,
      src3,
      dst,
      scattered_count
    );
  } else if (local_size == 2) {
    // compute the src pointers by applying the offsets
    src0 += offset;
    src1 += offset;

    reduce_x2_bf16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      src0,
      src1,
      dst,
      scattered_count
    );
  } else {
    return MUILLM_COMM_UNSUPPORTED_SIZE;
  }

  return MUILLM_COMM_SUCCESS;
}


muillm_comm_error_t __muillm_reduce_pull(
  hipStream_t stream,
  // inputs
  const void* src0, // shape [M, N] needs to be offset
  const void* src1, // shape [M, N] needs to be offset
  const void* src2, // shape [M, N] needs to be offset
  const void* src3, // shape [M, N] needs to be offset
  const void* src4, // shape [M, N] needs to be offset
  const void* src5, // shape [M, N] needs to be offset
  const void* src6, // shape [M, N] needs to be offset
  const void* src7, // shape [M, N] needs to be offset
  int scattered_count,
  int local_size,
  int local_rank,
  muillm_comm_datatype_t datatype,
  // outputs
  void* dst // shape [scattered_M, N]
) {
  if (datatype == MUILLM_COMM_FP16) {
    return __muillm_reduce_pull_fp16(
      stream,
      (const half*) src0,
      (const half*) src1,
      (const half*) src2,
      (const half*) src3,
      (const half*) src4,
      (const half*) src5,
      (const half*) src6,
      (const half*) src7,
      scattered_count,
      local_size,
      local_rank,
      (half*) dst
    );
  } else if (datatype == MUILLM_COMM_BF16) {
    return __muillm_reduce_pull_bf16(
      stream,
      (const __hip_bfloat16*) src0,
      (const __hip_bfloat16*) src1,
      (const __hip_bfloat16*) src2,
      (const __hip_bfloat16*) src3,
      (const __hip_bfloat16*) src4,
      (const __hip_bfloat16*) src5,
      (const __hip_bfloat16*) src6,
      (const __hip_bfloat16*) src7,
      scattered_count,
      local_size,
      local_rank,
      (__hip_bfloat16*) dst
    );
  } else {
    return MUILLM_COMM_UNKNOWN_ERROR;
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
        print("Loading All2All comm kernels...")

        try:
            self.comm_kernels = load_inline(
                name="all2all_comm_kernels",
                cpp_sources=[COMM_KERNELS_CPP_CODE],
                cuda_sources=[COMM_KERNELS_CUDA_CODE],
                functions=[
                    "all2all_comm_init",
                    "all2all_comm_destroy",
                    "all2all_comm_gemm_reduce_scatter",
                    "all2all_comm_gemm_reduce_scatter_pull",
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

        end_time = time.time()

        apply_monkey_patch()

        print(
            f"All2AllComm initialized in {end_time - start_time:.2f} seconds on rank {rank}.",
            flush=True,
        )

    def destroy(self):
        if self.comm_kernels is not None and self.comms is not None:
            self.comm_kernels.all2all_comm_destroy(self.comms)
            self.comms = None

    def gemm_reduce_scatter(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.comm_kernels.all2all_comm_gemm_reduce_scatter(
            self.comms,
            input,
            weight,
            bias,
        )

    def gemm_reduce_scatter_pull(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.comm_kernels.all2all_comm_gemm_reduce_scatter_pull(
            self.comms,
            input,
            weight,
            bias,
        )


_global_all2all_comm = None


def get_global_all2all_comm():
    global _global_all2all_comm
    if _global_all2all_comm is None:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        _global_all2all_comm = All2AllComm(rank, world_size)

    return _global_all2all_comm


def destroy_global_all2all_comm():
    global _global_all2all_comm
    if _global_all2all_comm is not None:
        _global_all2all_comm.destroy()
        _global_all2all_comm = None


def custom_kernel(data: input_t) -> output_t:
    """
    Reference kernel for Gemm-ReduceScatter operation.

    Args:
        data: Tuple of (input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor])
            - input: Local input tensor of shape [M, local_K].
            - weight: Weight tensor of shape [N, local_K].
            - bias: Optional bias tensor of shape [N] or None.
    Returns:
        Tuple containing:
            - output: Resulting tensor of shape [M // world_size, N].
    """
    input, weight, bias = data

    comms = get_global_all2all_comm()

    return comms.gemm_reduce_scatter_pull(
        input=input,
        weight=weight,
        bias=bias,
    )
