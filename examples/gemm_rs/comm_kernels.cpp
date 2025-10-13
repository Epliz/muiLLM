#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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

void __allocate_locked_cpu_mem(
    size_t size,
    void** host_addr_ptr,
    void** device_ptr_ptr
  ) {
  *host_addr_ptr = nullptr;
  *device_ptr_ptr = nullptr;

  void *host_addr = malloc(size);

  if (mlock(host_addr, size) != 0) {
    // TODO: return error code
    host_addr = nullptr;
  }

  // register the memory for use with HIP
  if (hipHostRegister(host_addr, size, hipHostRegisterPortable | hipHostRegisterMapped) != hipSuccess) {
    // TODO: return error code
    TORCH_CHECK(false, "an error happened when registering shared memory with HIP");
    return;
  }

  // get the device pointer after registration
  if (hipHostGetDevicePointer((void**)device_ptr_ptr, host_addr, 0) != hipSuccess) {
    // TODO: return error code
    TORCH_CHECK(false, "an error happened when getting device pointer for shared memory");
    return;
  }

  // return
  *host_addr_ptr = host_addr;
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

void __deallocate_locked_cpu_mem(
    muillm_comm_t* comm,
    void* host_addr
  ) {
  int local_rank = comm->local_rank;

  if (hipHostUnregister(host_addr) != hipSuccess) {
    // TODO: return error code
    TORCH_CHECK(false, "an error happened when unregistering shared memory from HIP");
    return;
  }
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


typedef struct muillm_comm_p2p: muillm_comm {

  muillm_comm_p2p(at::cuda::CUDAStream second_stream): second_stream(second_stream) {

  }

  // reduction buffer sets
  muillm_comm_p2p_buffer_set_t* first_buffers;
  muillm_comm_p2p_buffer_set_t* second_buffers;

  // shared signal memory to synchronize GPUs
  uint32_t* signal_host;
  uint32_t* signal;

  at::cuda::CUDAStream second_stream;

  // local stream signal memory to synchronize GPUs
  uint32_t* stream_signal_host;
  uint32_t* stream_signal;

  uint32_t signal_seq_no;
  uint32_t stream_signal_seq_no;

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
  hipStream_t stream,
  at::cuda::CUDAStream second_stream
) {
  if (world_size != local_size) {
    // we currently ony support single machine, so
    // we should fail
    return MUILLM_COMM_UNSUPPORTED_SIZE;
  }

  muillm_comm_error_t muillm_error;

  muillm_comm_method_t transfer_method = MUILLM_COMM_METHOD_P2P_TRANSFER;

  // create the comm object
  muillm_comm_p2p_t* comm = new muillm_comm_p2p_t(second_stream);
  comm->transfer_method = transfer_method;

  comm->world_size = world_size;
  comm->local_size = local_size;
  comm->rank = rank;
  comm->local_rank = local_rank;

  comm->signal_host = nullptr;
  comm->signal = nullptr;
  comm->signal_seq_no = 0;

  comm->stream_signal_host = nullptr;
  comm->stream_signal = nullptr;
  comm->stream_signal_seq_no = 0;

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

  // allocate shared signal memory
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

  // allocate signal memory
  __allocate_locked_cpu_mem(
    sizeof(uint64_t), // alloc 8 bytes even though we use only 4
    (void**) &comm->stream_signal_host,
    (void**) &comm->stream_signal
  );

  if (comm->stream_signal_host == nullptr || comm->stream_signal == nullptr) {
    return MUILLM_COMM_UNKNOWN_ERROR;
  }
  // initialize to 0
  if (hipMemset(comm->stream_signal, 0, sizeof(uint64_t)) != hipSuccess) {
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

  // free stream signal memory
  if (comm->stream_signal_host != nullptr) {
    __deallocate_locked_cpu_mem(
      comm,
      comm->stream_signal_host
    );
    comm->stream_signal_host = nullptr;
    comm->stream_signal = nullptr;
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

muillm_comm_error_t __mui_stream_wait_value(hipStream_t stream, uint32_t* signal, uint32_t seq_no);

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

static muillm_comm_error_t __mui_signal_stream_gpu_barrier(muillm_comm_p2p_t* comm, hipStream_t stream) {
  int local_rank = comm->local_rank;

  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  if (comm->stream_signal != nullptr) {
    // GPU barrier: all GPUs wait on each other
    if (comm->cant_skip_cache_flush_event) {
      // on MI100, we get a crash if not putting this event here
      // record an event to flush caches
      if (hipEventRecord(comm->cache_flush_event, stream) != hipSuccess) {
        std::cout<<"rank "<<local_rank<<" signal stream gpu barrier failed because hipEventRecord failed"<<std::endl;
        hipError_t err = hipGetLastError();
        const char* errStr = hipGetErrorString(err);
        std::cout<<"Last HIP error: "<<errStr<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }

    // write the values
    if ((muillm_error = __mui_stream_inc_value(stream, comm->stream_signal)) != MUILLM_COMM_SUCCESS) {
      std::cout<<"rank "<<local_rank<<" signal stream gpu barrier failed because __mui_stream_inc_wait_value failed"<<std::endl;
      return muillm_error;
    }
  } else {
    std::cout<<"rank "<<local_rank<<" signal stream gpu barrier failed because there is no signal memory"<<std::endl;
    return MUILLM_COMM_UNKNOWN_ERROR;
  }

  return MUILLM_COMM_SUCCESS;
}

static muillm_comm_error_t __mui_wait_stream_gpu_barrier(muillm_comm_p2p_t* comm, hipStream_t stream, uint64_t seq_no) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  hipError_t hip_error;
  muillm_comm_error_t muillm_error;

  if (comm->stream_signal != nullptr) {
    // was increment by __mui_signal_stream_gpu_barrier

    // GPU barrier: all GPUs wait on each other
    if (comm->cant_skip_cache_flush_event) {
      // on MI100, we get a crash if not putting this event here
      // record an event to flush caches
      if (hipEventRecord(comm->cache_flush_event, stream) != hipSuccess) {
        std::cout<<"rank "<<local_rank<<" wait stream gpu barrier failed because hipEventRecord failed"<<std::endl;
        hipError_t err = hipGetLastError();
        const char* errStr = hipGetErrorString(err);
        std::cout<<"Last HIP error: "<<errStr<<std::endl;
        return MUILLM_COMM_UNKNOWN_ERROR;
      }
    }

    // write the values
    if ((muillm_error = __mui_stream_wait_value(stream, comm->stream_signal, seq_no)) != MUILLM_COMM_SUCCESS) {
      std::cout<<"rank "<<local_rank<<" wait stream gpu barrier failed because __mui_stream_wait_value failed"<<std::endl;
      return muillm_error;
    }
  } else {
    std::cout<<"rank "<<local_rank<<" wait stream gpu barrier failed because there is no signal memory"<<std::endl;
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

muillm_comm_error_t __muillm_scatter_all_chunk(
  hipStream_t stream,
  // inputs
  const void* src,
  int scattered_chunk_size_bytes,
  int scattered_chunk_offset,
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

muillm_comm_error_t __muillm_reduce_chunks(
  hipStream_t stream,
  // inputs
  const void* src, // shape [num_chunks, local_size, M, chunk_N]
  int M,
  int chunk_N,
  int num_chunks,
  int local_size,
  muillm_comm_datatype_t datatype,
  // outputs
  void* dst // shape [M, chunk_N * num_chunks]
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

// torch extension

#include <tuple>

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

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  auto device_index = stream.device_index();
  at::cuda::CUDAStream second_stream = at::cuda::getStreamFromPool(false, device_index);

  muillm_comm_p2p_t* comm_ptr = nullptr;
  muillm_error = muillm_comm_p2p_init_comm(
    world_size,
    local_size,
    rank,
    local_rank,
    &local_socket,
    (muillm_comm_p2p_t**) &comm_ptr,
    stream,
    second_stream
  );

  TORCH_CHECK(muillm_error == MUILLM_COMM_SUCCESS, "an error happened when initializing mui comm");

  return (void*) comm_ptr;
}

void all2all_comm_destroy(void* comms) {
  muillm_comm_p2p_t* comm = (muillm_comm_p2p_t*) comms;

  muillm_comm_error_t muillm_error = muillm_comm_p2p_destroy_comm(comm);

  TORCH_CHECK(muillm_error == MUILLM_COMM_SUCCESS, "an error happened when destroying mui comm");
}

#define REDUCE_SCATTER_CHUNKED_THRESHOLD (4 * 1024 * 1024) // 4M elements

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
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device.index());
  at::cuda::CUDAStream second_stream = comm->second_stream;

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

  int total_size = scattered_M * N;
  // if the total size is big enough, we split the computation and communication
  // into two halves (divide the N dimension of inputs) to overlap them
  // we can't easily split the M dimension as it would be scattered wrongly
  // (first half of M has to go to first half of GPUs, not split across all GPUs)

  // always use this for debugging
  if (true) {//(total_size >= REDUCE_SCATTER_CHUNKED_THRESHOLD) {
    muillm_comm_error_t error;

    int num_chunks = 2; // TODO: Generalize

    if (N % num_chunks != 0) {
      TORCH_CHECK(false, "N must be divisible by num_chunks");
      return torch::Tensor();
    }

    int half_N = N / num_chunks;
    int second_half_N = N - half_N;

    // get first and seconds halves of the weights and biases
    auto first_weights_half = weights.narrow(0, 0, half_N); // shape [half_N, K]
    auto second_weights_half = weights.narrow(0, half_N, second_half_N); // shape [second_half_N, K]

    std::optional<torch::Tensor> first_bias_half;
    std::optional<torch::Tensor> second_bias_half;

    if (bias.has_value()) {
      auto bias_ = bias.value();
      first_bias_half = std::make_optional(bias_.narrow(0, 0, half_N));
      second_bias_half = std::make_optional(bias_.narrow(0, half_N, second_half_N));
    }

    auto rs_output = torch::empty({scattered_M, N}, output_options);

    //
    // First make sure we have enough buffer space
    //
    int scattered_count = scattered_M * N;
    int scattered_size = __comm_size(datatype, scattered_count);
    int capacity = scattered_size * local_size;

    int scattered_chunk_count = scattered_M * half_N;
    int scattered_chunk_size = __comm_size(datatype, scattered_chunk_count);
    int second_scattered_chunk_count = scattered_M * second_half_N;
    int second_scattered_chunk_size = __comm_size(datatype, second_scattered_chunk_count);

    muillm_comm_p2p_buffer_set_t* buffer_set;
    if ((error = muillm_comm_p2p_get_buffer_set(comm, capacity, &buffer_set, stream)) != MUILLM_COMM_SUCCESS) {
      std::cout<<"rank "<<local_rank<<" failed to get buffer set"<<std::endl;
      TORCH_CHECK(false, "an error happened when getting buffer set");
      return torch::Tensor();
    }

    //
    // Start the computations
    //

    comm->stream_signal_seq_no ++;
    // value that the second stream will wait on
    uint64_t first_barrier_seq_no = comm->stream_signal_seq_no;
    comm->stream_signal_seq_no ++;
    // value that the first stream will wait on
    uint64_t second_barrier_seq_no = comm->stream_signal_seq_no;

    torch::Tensor output_half; // to be used in the second stream
    torch::Tensor second_output_half; // to be used in the main stream

    { // Spawn work for the main stream

      // work on the first half
      output_half = torch::linear(input, first_weights_half, first_bias_half); // shape [M, half_N]

      // signal we are done with the first computations
      if (__mui_signal_stream_gpu_barrier(comm, stream) != MUILLM_COMM_SUCCESS) {
        TORCH_CHECK(false, "an error happened when doing signal stream gpu barrier");
      }

      // work on the second half
      second_output_half = torch::linear(input, second_weights_half, second_bias_half); // shape [M, second_half_N]

      int scatter_offset = local_size * scattered_chunk_size;
      // scatter
      if ((error = __muillm_scatter_all_chunk(
        stream,
        second_output_half.data_ptr(),
        second_scattered_chunk_size, // size of this chunk
        scatter_offset, // offset of this chunk is the size of the gathered previous one
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
        TORCH_CHECK(false, "an error happened when scattering data to other GPUs");
        return torch::Tensor();
      }
    }

    { // Spawn work for the second stream
      at::cuda::setCurrentCUDAStream(second_stream);

      // wait for the signal from the main stream
      // TODO: fuse in the scatter operation
      if (__mui_wait_stream_gpu_barrier(comm, second_stream, first_barrier_seq_no) != MUILLM_COMM_SUCCESS) {
        TORCH_CHECK(false, "an error happened when doing wait stream gpu barrier");
        return torch::Tensor();
      }

      int scatter_offset = 0;

      // scatter
      if ((error = __muillm_scatter_all_chunk(
        second_stream,
        output_half.data_ptr(),
        scattered_chunk_size, // size of this chunk
        0, // offset of this chunk
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
        TORCH_CHECK(false, "an error happened when scattering data to other GPUs");
        return torch::Tensor();
      }

      // signal we are done with the transfers
      if (__mui_signal_stream_gpu_barrier(comm, second_stream) != MUILLM_COMM_SUCCESS) {
        TORCH_CHECK(false, "an error happened when doing signal stream gpu barrier");
      }

      // switch back to the original stream
      at::cuda::setCurrentCUDAStream(stream);
    }

    // wait for the second stream to finish
    if (__mui_wait_stream_gpu_barrier(comm, stream, second_barrier_seq_no) != MUILLM_COMM_SUCCESS) {
      TORCH_CHECK(false, "an error happened when doing wait stream gpu barrier");
    }

    // Synchronize all the GPUs
    // TODO fuse with previous wait as well
    if (__mui_gpu_barrier(comm, stream) != MUILLM_COMM_SUCCESS) {
      TORCH_CHECK(false, "an error happened when doing gpu barrier");
      return torch::Tensor();
    }

    // Finally reduce the data on each GPU
    // We have two chunks to reduce
    if ((error = __muillm_reduce_chunks(
      stream,
      buffer_set->buffers[local_rank],
      scattered_M,
      half_N, // TODO change when num_chunks != 2
      num_chunks,
      local_size,
      datatype,
      rs_output.data_ptr()
    )) != MUILLM_COMM_SUCCESS) {
      std::cout<<"rank "<<local_rank<<" failed to reduce data"<<std::endl;
      TORCH_CHECK(false, "an error happened when reducing data");
      return torch::Tensor();
    }

    return rs_output;
  } else {
    // do it in one go
    auto output = torch::linear(input, weights, bias); // shape [M, N]

    auto rs_output = torch::empty({scattered_M, N}, output_options);

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

    return rs_output;
  }


  // equivalent to:
  /* {
    std::vector<torch::Tensor> rs_output_vector = {rs_output};
    std::vector<torch::Tensor> input_vector = {output};
    comm->process_group->reduce_scatter_tensor_coalesced(
      rs_output_vector,
      input_vector,
      c10d::ReduceScatterOptions()
    );
  } */
}