#include "comm_base.h"


#include <hip/hip_runtime.h>

#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <errno.h>
#include <poll.h>

#include <stdio.h>

#include <torch/torch.h>
#include <distributed/c10d/ProcessGroup.hpp>

#define MUILLM_COMM_SOCKET_PATH "/tmp/muillm_comm_socket"

static int full_read(int fd, void* ptr, size_t byte_count) {
  size_t to_read_count = byte_count;

  uint8_t* byte_ptr = (uint8_t*) ptr;
  while (to_read_count > 0) {
    int read_status = read(fd, byte_ptr, to_read_count);

    if (read_status < 0) {
      return -1;
    }

    byte_ptr += read_status;
    to_read_count -= read_status;
  }

  return byte_count;
}

static int full_write(int fd, const void* ptr, size_t byte_count) {
  size_t to_write_count = byte_count;

  const uint8_t* byte_ptr = (const uint8_t*) ptr;
  while (to_write_count > 0) {
    int write_status = write(fd, byte_ptr, to_write_count);

    if (write_status < 0) {
      return -1;
    }

    byte_ptr += write_status;
    to_write_count -= write_status;
  }

  return byte_count;
}

// creates the domain sockets used to do cpu side exchanges
// (e.g. to exchange memory IPC handles)
muillm_comm_error_t __open_local_socket(
    int local_size,
    int local_rank,
    std::shared_ptr<c10d::ProcessGroup>& process_group,
    muillm_comm_local_socket_t* local_socket
) {
  local_socket->server_fd = -1;
  local_socket->client_to_server_fd = -1;
  local_socket->server_to_client_fds = nullptr;
  local_socket->process_group = process_group;

  bool is_server;

  // try to become the server
  struct sockaddr_un server_addr;

  // Create socket
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return MUILLM_COMM_SOCKET_CREATION_FAILED;
  }

  // Try bindnig socket to address
  memset(&server_addr, 0, sizeof(struct sockaddr_un));
  server_addr.sun_family = AF_UNIX;
  strncpy(server_addr.sun_path, MUILLM_COMM_SOCKET_PATH, sizeof(server_addr.sun_path) - 1);

  // the server is always the rank 0
  is_server = local_rank == 0;

  if (is_server) {
    bool correctly_bound = bind(fd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_un)) == 0;

    if (!correctly_bound) {
      printf("(rank %d) Could not bind as server error was %s\n", local_rank, strerror(errno));
      return MUILLM_COMM_SOCKET_BIND_FAILED;
    }

    // if we managed to bind, we should connect to all other ranks and they will be the socket
    // clients
    local_socket->server_fd = fd;

    int num_clients = local_size -1;

    local_socket->server_to_client_fds = new int[local_size];
    local_socket->server_to_client_fds[local_rank] = -1;

    // Listen for connections
    if (listen(local_socket->server_fd, num_clients) == -1) {
      printf("(rank %d) listen failed\n", local_rank);
      return MUILLM_COMM_SOCKET_LISTEN_FAILED;
    }

    struct pollfd poll_fd;
    poll_fd.fd = local_socket->server_fd;
    poll_fd.events = POLLIN;

    for (int c = 0; c < num_clients; c++) {

      int poll_count = poll(&poll_fd, 1, -1);
      if (poll_count < 0) {
        printf("(rank %d) Polling error, error was %s\n", local_rank, strerror(errno));
        return MUILLM_COMM_SOCKET_ACCEPT_FAILED;
      }

      if (poll_fd.revents & POLLIN) {
        int client_fd = accept(local_socket->server_fd, NULL, NULL);
        if (client_fd == -1) {
          printf("(rank %d) Accept failed error was %s\n", local_rank, strerror(errno));
          return MUILLM_COMM_SOCKET_ACCEPT_FAILED;
        }
      
        // now read what rank this client corresponds to
        int client_rank = 0;

        if (full_read(client_fd, &client_rank, sizeof(int)) < 0) {
          printf("(rank %d) partial read\n", local_rank);
          return MUILLM_COMM_SOCKET_READ_ERROR;
        }

        local_socket->server_to_client_fds[client_rank] = client_fd;
      }
    }

    // we can already unlink as everyone has connected
    unlink(MUILLM_COMM_SOCKET_PATH);

  } else {
    // if we didn't manage to bind, we are a mere client
    // we just need to connect to the server
    local_socket->client_to_server_fd = fd;

    bool connected = false;

    while (!connected) {
      if (connect(local_socket->client_to_server_fd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_un)) == -1) {
        printf("(rank %d) Could connect to the server error was %s\n", local_rank, strerror(errno));
        sleep(10); // wait before trying to connect again
        continue;
      }

      connected = true;
    }

    // send to the server our rank
    if (full_write(local_socket->client_to_server_fd, &local_rank, sizeof(int)) < 0) {
      return MUILLM_COMM_SOCKET_WRITE_ERROR;
    }
  }

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