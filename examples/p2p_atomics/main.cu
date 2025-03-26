
#include <hip/hip_runtime.h>

#include <iostream>

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
      __builtin_amdgcn_buffer_wbinvl1();
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

  // allocate memory for counters on each GPU
  int* counters[num_gpus];

  for (int i = 0; i < num_gpus; i++) {
    if (hipSetDevice(i) != hipSuccess) {
      std::cerr << "Failed to set device " << i << std::endl;
      return -1;
    }

    // enable peer to peer access
    for (int j = 0; j < num_gpus; j++) {
      if (i != j) {
        int canAccessPeer;
        if (hipDeviceCanAccessPeer(&canAccessPeer, i, j) != hipSuccess) {
          std::cerr << "Failed to check peer access between devices " << i
                    << " and " << j << std::endl;
          return -1;
        }

        if (canAccessPeer) {
          if (hipDeviceEnablePeerAccess(j, 0) != hipSuccess) {
            std::cerr << "Failed to enable peer access between devices " << i
                      << " and " << j << std::endl;
            return -1;
          }
        } else {
          std::cerr << "Devices " << i << " and " << j
                    << " cannot access each other" << std::endl;
          return -1;
        }
      }
    }

    if (hipExtMallocWithFlags((void**)&counters[i], sizeof(int), hipDeviceMallocUncached) != hipSuccess) {
      std::cerr << "Failed to allocate memory for counter on device " << i
                << std::endl;
      return -1;
    }

    hipPointerAttribute_t attributes;
    if (hipPointerGetAttributes(&attributes, counters[i]) != hipSuccess) {
      std::cerr << "Failed to get pointer attributes for counter on device "
                << i << std::endl;
      return -1;
    }

    std::cout<<"Pointer type for device "<<i<<" is "<<attributes.type<<std::endl;

    std::cout << "Counter on device " << i << " is at address "
              << counters[i] << std::endl;

    // initialize the counters to zero
    if (hipMemset(counters[i], 0, sizeof(int)) != hipSuccess) {
      std::cerr << "Failed to set counter to zero on device " << i << std::endl;
      return -1;
    }

    // synchronize the device
    if (hipDeviceSynchronize() != hipSuccess) {
      std::cerr << "Failed to synchronize device " << i << std::endl;
      return -1;
    }
  }

  // create streams for each GPU
  hipStream_t streams[num_gpus];
  for (int i = 0; i < 2; i++) {
    if (hipSetDevice(i) != hipSuccess) {
      std::cerr << "Failed to set device " << i << std::endl;
      return -1;
    }
    if (hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking) != hipSuccess) {
      std::cerr << "Failed to create stream on device " << i << std::endl;
      return -1;
    }
  }

  std::cout<<"Launching kernels..."<<std::endl;

  // now launch the kernels to increment the counters on each GPU
  for (int i = 0; i < num_gpus; i++) {    
    incrementCounter<<<1, 1, 0, streams[i]>>>(counters[num_gpus - 1 - i], counters[i]);
  }

  // synchronize the streams
  for (int i = 0; i < num_gpus; i++) {
    if (hipSetDevice(i) != hipSuccess) {
      std::cerr << "Failed to set device " << i << std::endl;
      return -1;
    }
    if (hipStreamSynchronize(streams[i]) != hipSuccess) {
      std::cerr << "Failed to synchronize stream on device " << i << std::endl;
      return -1;
    }
  }

  std::cout<<"DONE"<<std::endl;

  // now launch the kernels to print the counters on each GPU
  for (int i = 0; i < num_gpus; i++) {
    printKernel<<<1, 1, 0, streams[i]>>>(i, counters[i]);
  }

  // synchronize the streams
  for (int i = 0; i < num_gpus; i++) {
    if (hipSetDevice(i) != hipSuccess) {
      std::cerr << "Failed to set device " << i << std::endl;
      return -1;
    }
    if (hipStreamSynchronize(streams[i]) != hipSuccess) {
      std::cerr << "Failed to synchronize stream on device " << i << std::endl;
      return -1;
    }
  }

  std::cout<<"DONE"<<std::endl;

  return 0;
}