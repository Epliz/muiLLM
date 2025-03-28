
#include <hip/hip_runtime.h>

#include <iostream>

#include <unistd.h>
#include <sys/types.h>

__global__ void incrementCounter(int* remote_counter, int* local_counter) {
  if (threadIdx.x == 0) {
    atomicAdd_system(remote_counter, 1);

    __threadfence_system();

    // wait for our counter to be incremented
    // while (atomicAdd((int*) local_counter, 0) < 1) {
    //   __builtin_amdgcn_s_sleep(2);
    // }
    while (atomicAdd_system(local_counter, 0) < 1) {
      __builtin_amdgcn_s_sleep(2);
      __threadfence_system();
    }
  }
}

__global__ void printKernel(int rank, int* counter) {
  if (threadIdx.x == 0) {
    printf("(rank %d) Counter: %d\n", rank, *counter);
  }
}

int main(int argc, char** argv) {
  // pipe file descriptors to share data after forking
  int pipefd[2];
  if (pipe(pipefd) == -1) {
    std::cerr << "Failed to create pipe" << std::endl;
    return -1;
  }
  
  // fork the process to create multiple processes
  bool is_parent = true;
  if (fork() == 0) {
    // child process
    is_parent = false;
    std::cout << "Child process created" << std::endl;
  }

  // get the number of GPUs
  constexpr int num_gpus = 2;

  int deviceCount;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess) {
    std::cerr << "Failed to get the number of devices" << std::endl;
    return -1;
  }

  if (deviceCount < num_gpus) {
    std::cerr << "This example requires at least two devices" << std::endl;
    return -1;
  }

  int local_rank = is_parent ? 0 : 1;

  // allocate memory for counters on each GPU
  int* local_counter = nullptr;

  if (hipSetDevice(local_rank) != hipSuccess) {
    std::cerr << "Failed to set device " << local_rank << std::endl;
    return -1;
  }

  // enable peer to peer access
  for (int j = 0; j < num_gpus; j++) {
    if (local_rank != j) {
    int canAccessPeer;
    if (hipDeviceCanAccessPeer(&canAccessPeer, local_rank, j) != hipSuccess) {
        std::cerr << "Failed to check peer access between devices " << local_rank
                << " and " << j << std::endl;
        return -1;
    }

    if (canAccessPeer) {
        if (hipDeviceEnablePeerAccess(j, 0) != hipSuccess) {
        std::cerr << "Failed to enable peer access between devices " << local_rank
                    << " and " << j << std::endl;
        return -1;
        }
    } else {
        std::cerr << "Devices " << local_rank << " and " << j
                << " cannot access each other" << std::endl;
        return -1;
    }
    }
  }

  if (hipExtMallocWithFlags((void**)&local_counter, sizeof(int), hipDeviceMallocUncached) != hipSuccess) {
    std::cerr << "Failed to allocate memory for counter on device " << local_rank
            << std::endl;
    return -1;
  }

  hipPointerAttribute_t attributes;
  if (hipPointerGetAttributes(&attributes, local_counter) != hipSuccess) {
    std::cerr << "Failed to get pointer attributes for counter on device "
            << local_rank << std::endl;
    return -1;
  }

  std::cout<<"Pointer type for device "<<local_rank<<" is "<<attributes.type<<std::endl;

  std::cout << "Counter on device " << local_rank << " is at address "
            << local_counter << std::endl;

  // initialize the counters to zero
  if (hipMemset(local_counter, 0, sizeof(int)) != hipSuccess) {
    std::cerr << "Failed to set counter to zero on device " << local_counter << std::endl;
    return -1;
  }

  // synchronize the device
  if (hipDeviceSynchronize() != hipSuccess) {
    std::cerr << "Failed to synchronize device " << local_counter << std::endl;
    return -1;
  }

  std::cout<<"Allocated the memory"<<std::endl;

  // get the pointer from the other process
  int* remote_counter;

  // get the mem handle to share with the other process

  hipIpcMemHandle_t ipcHandle;
  if (hipIpcGetMemHandle(&ipcHandle, local_counter) != hipSuccess) {
    std::cerr << "Failed to get IPC handle for counter on device " << local_rank<< std::endl;
    return -1;
  }

  // get the memory handle from the other process
  if (write(pipefd[1], &ipcHandle, sizeof(hipIpcMemHandle_t)) < sizeof(hipIpcMemHandle_t)) {
    std::cerr << "Failed to write IPC handle to pipe" << std::endl;
    return -1;
  }

  hipIpcMemHandle_t remoteIpcHandle;
  if (read(pipefd[0], &remoteIpcHandle, sizeof(hipIpcMemHandle_t)) < sizeof(hipIpcMemHandle_t)) {
    std::cerr << "Failed to read IPC handle from pipe" << std::endl;
    return -1;
  }

  close(pipefd[0]);
  close(pipefd[1]);

  // open the remote memory handle
  if (hipIpcOpenMemHandle((void**)&remote_counter, remoteIpcHandle,
                            hipIpcMemLazyEnablePeerAccess) != hipSuccess) {
    std::cerr << "Failed to open IPC handle for counter on device " << local_rank
                << std::endl;
    return -1;
  }

  std::cout << "Remote counter on device " << local_rank << " is at address "
            << remote_counter << std::endl;

  // run the kernels

  std::cout<<"Launching kernels..."<<std::endl;

  // now launch the kernels to increment the counters on each GPU
  incrementCounter<<<1, 1>>>(remote_counter, local_counter);

  if (hipDeviceSynchronize() != hipSuccess) {
    std::cerr << "Failed to synchronize device " << local_rank << std::endl;
    return -1;
  }

  std::cout<<"DONE"<<std::endl;

  // now launch the kernels to print the counters on each GPU
  printKernel<<<1, 1>>>(local_rank, local_counter);

  // synchronize
  if (hipDeviceSynchronize() != hipSuccess) {
    std::cerr << "Failed to synchronize device " << local_rank << std::endl;
    return -1;
  }

  std::cout<<"DONE"<<std::endl;

  return 0;
}