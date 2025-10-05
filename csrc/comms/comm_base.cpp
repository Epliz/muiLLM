#include "comm_base.h"


#include <hip/hip_runtime.h>

#include <string.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <errno.h>

#include <stdio.h>

#include <torch/torch.h>
#include <distributed/c10d/ProcessGroup.hpp>

// creates the domain sockets used to do cpu side exchanges
// (e.g. to exchange memory IPC handles)
muillm_comm_error_t __open_local_socket(
    int local_size,
    int local_rank,
    std::shared_ptr<c10d::ProcessGroup>& process_group,
    muillm_comm_local_socket_t* local_socket
) {
  local_socket->process_group = process_group;
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
  torch::Tensor cpu_tensor = torch::empty({byte_count}, tensor_options);

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
  torch::Tensor cpu_tensor = torch::empty({byte_count}, tensor_options);

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
    auto out_tensor = torch::empty({byte_count}, out_tensor_options);
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
      printf("(rank %d) could not get SHM id\n", local_rank);
      return;
    }
    
    shm_addr = shmat(shm_id, NULL, 0);
    if (shm_addr == (void *) -1) {
      // TODO: return error code
      printf("(rank %d) could not get SHM addr\n", local_rank);
      return;
    }

    // make the memory be deleted once all processes have detached from it
    // (memory is automatically detached on process exit)
    if (shmctl(shm_id, IPC_RMID, NULL) != 0) {
      // TODO: return error code
      return;
    }

    if (mlock(shm_addr, size) != 0) {
      // TODO: return error code
      printf("(rank %d) could not lock shared memory\n", local_rank);
      shm_id = - 1;
      // go to the broadcast
    }
  }

  // get the share memory ID on all ranks
  __local_socket_broadcast(comm, /*src*/ 0, &shm_id, sizeof(int));

  if (shm_id < 0) {
    // TODO: return error code
    printf("(rank %d) could not get SHM id\n", local_rank);
    return;
  }

  if (local_rank != 0) {
    shm_addr = shmat(shm_id, NULL, 0);
    if (shm_addr == (void *) -1) {
      // TODO: return error code
      printf("(rank %d) could not get SHM addr\n", local_rank);
      return;
    }
  }

  // register the memory for use with HIP
  if (hipHostRegister(shm_addr, size, hipHostRegisterPortable | hipHostRegisterMapped) != hipSuccess) {
    // TODO: return error code
      printf("(rank %d) could not register host address\n", local_rank);
    return;
  }

  // get the device pointer after registration
  if (hipHostGetDevicePointer((void**)device_ptr_ptr, shm_addr, 0) != hipSuccess) {
    // TODO: return error code
      printf("(rank %d) could not get device pointer\n", local_rank);
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
    printf("(rank %d) could not unregister host address\n", local_rank);
    return;
  }

  if (shmdt(host_addr) != 0) {
    printf("(rank %d) could not detach shared address\n", local_rank);
    return;
  }
}