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

  ctx->signal_host = nullptr;
  ctx->signal = nullptr;
  ctx->signal_seq_no = 0;

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

    // write the values
    if ((muillm_error = __mui_stream_inc_wait_value(stream, ctx->signal, seq_no)) != MUILLM_COMM_SUCCESS) {
      std::cout<<"rank "<<local_rank<<" gpu barrier failed because __mui_stream_inc_wait_value failed"<<std::endl;
      return muillm_error;
    }
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

    // write the values
    if ((muillm_error = __mui_stream_inc_wait_value_cache_val(stream, ctx->signal, seq_no, uncached_val, cached_val)) != MUILLM_COMM_SUCCESS) {
      std::cout<<"rank "<<local_rank<<" caching gpu barrier failed because __mui_stream_inc_wait_value failed"<<std::endl;
      return muillm_error;
    }
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

  // print the pointers for debugging
  std::cout<<"rank "<<local_rank<<" num_local_experts "<<num_local_experts<<" num_tokens "<<num_tokens<<" num_experts_per_token "<<num_experts_per_token<<" hidden_dim "<<hidden_dim<<" max_recv "<<max_recv<<" max_total_recv "<<max_total_recv<<" total_send "<<total_send<<std::endl;
  std::cout<<"rank "<<local_rank<<" buffer pointers: ";
  for (int i = 0; i < 8; i++) {
    std::cout<<" "<<buffer_set->buffers[i];
  }
  std::cout<<std::endl;

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

  HIP_CHECK(local_rank, hipGetLastError());
  HIP_CHECK(local_rank, hipDeviceSynchronize());

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

  std::cout<<"rank "<<comm->rank<<" all2all multi stream start"<<std::endl;

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
  // TODO: use different streams
  at::cuda::CUDAStream first_stream = stream;
  at::cuda::CUDAStream second_stream = stream;

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
  std::cout<<"rank "<<comm->rank<<" first dispatch"<<std::endl;
  auto dispatch_outputs1 = all2all_comm_dispatch_on_stream(
    comm,
    first_ctx,
    first_stream,
    x1,
    indices1,
    num_local_experts,
    half_max_recv
  );

  std::cout<<"rank "<<comm->rank<<" second dispatch"<<std::endl;
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
    std::cout<<"rank "<<comm->rank<<" first compute"<<std::endl;
    expert_x1 = all2all_compute_on_stream(
      first_stream,
      expert_num_tokens1,
      expert_x1,
      comm->rank
    );

  std::cout<<"rank "<<comm->rank<<" second compute"<<std::endl;
    expert_x2 = all2all_compute_on_stream(
      second_stream,
      expert_num_tokens2,
      expert_x2,
      comm->rank
    );
  }

  // Finally combine
  float s = fuse_compute_in_combine ? (1.0f + local_rank) : 1.0f;
  std::cout<<"rank "<<comm->rank<<" first combine"<<std::endl;
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

  std::cout<<"rank "<<comm->rank<<" second combine"<<std::endl;
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

  std::cout<<"rank "<<comm->rank<<" all2all multi stream done"<<std::endl;
  // TODO: sync streams

  at::cuda::setCurrentCUDAStream(stream);

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