
#include <hip/hip_runtime.h>

#include <iostream>


#define THREADS_PER_BLOCK 256


// each threads can copy 16 bytes
#define COPY_BYTES_PER_THREAD 16
#define COPY_BYTES_PER_BLOCK (THREADS_PER_BLOCK * COPY_BYTES_PER_THREAD)

#define COPY_FP16S_PER_BLOCK (COPY_BYTES_PER_BLOCK / 2)
#define COPY_FP32S_PER_BLOCK (COPY_BYTES_PER_BLOCK / 4)

typedef struct uint32x4{
uint32_t x, y, z, w;
} uint32x4_t;

__device__ uint32x4_t load_nontemporal_uint4(const uint32_t* p) {
  uint32_t _v0 = __builtin_nontemporal_load(((const uint32_t*)p));
  uint32_t _v1 = __builtin_nontemporal_load(((const uint32_t*)p) + 1);
  uint32_t _v2 = __builtin_nontemporal_load(((const uint32_t*)p) + 2);
  uint32_t _v3 = __builtin_nontemporal_load(((const uint32_t*)p) + 3);

  uint32x4_t v;
  v.x = _v0;
  v.y = _v1;
  v.z = _v2;
  v.w = _v3;

  return v;
}

__device__ void __muillm_do_copy_p2p(
  const uint8_t* __restrict__ src_ptr,
  uint8_t* __restrict__ dst_ptr,
  unsigned N,
  unsigned block_chunk_size = COPY_BYTES_PER_BLOCK
) {
  unsigned blockStart = blockIdx.x * block_chunk_size;
  unsigned blockEnd = std::min(blockStart + block_chunk_size, N);

  unsigned i = blockStart + (threadIdx.x * COPY_BYTES_PER_THREAD);

  while (i + (16 * COPY_BYTES_PER_BLOCK) <= blockEnd) {
    const uint32_t* src_u32_ptr0 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr0 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr1 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr1 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr2 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr2 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr3 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr3 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr4 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr4 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr5 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr5 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr6 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr6 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr7 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr7 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr8 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr8 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr9 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr9 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr10 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr10 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr11 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr11 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr12 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr12 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr13 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr13 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr14 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr14 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr15 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr15 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    *dst_x16_ptr0 = load_nontemporal_uint4(src_u32_ptr0);
    *dst_x16_ptr1 = load_nontemporal_uint4(src_u32_ptr1);
    *dst_x16_ptr2 = load_nontemporal_uint4(src_u32_ptr2);
    *dst_x16_ptr3 = load_nontemporal_uint4(src_u32_ptr3);
    *dst_x16_ptr4 = load_nontemporal_uint4(src_u32_ptr4);
    *dst_x16_ptr5 = load_nontemporal_uint4(src_u32_ptr5);
    *dst_x16_ptr6 = load_nontemporal_uint4(src_u32_ptr6);
    *dst_x16_ptr7 = load_nontemporal_uint4(src_u32_ptr7);
    *dst_x16_ptr8 = load_nontemporal_uint4(src_u32_ptr8);
    *dst_x16_ptr9 = load_nontemporal_uint4(src_u32_ptr9);
    *dst_x16_ptr10 = load_nontemporal_uint4(src_u32_ptr10);
    *dst_x16_ptr11 = load_nontemporal_uint4(src_u32_ptr11);
    *dst_x16_ptr12 = load_nontemporal_uint4(src_u32_ptr12);
    *dst_x16_ptr13 = load_nontemporal_uint4(src_u32_ptr13);
    *dst_x16_ptr14 = load_nontemporal_uint4(src_u32_ptr14);
    *dst_x16_ptr15 = load_nontemporal_uint4(src_u32_ptr15);
  }

  if (i + (8 * COPY_BYTES_PER_BLOCK) <= blockEnd) {
    const uint32_t* src_u32_ptr0 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr0 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr1 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr1 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr2 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr2 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr3 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr3 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr4 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr4 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr5 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr5 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr6 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr6 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr7 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr7 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    *dst_x16_ptr0 = load_nontemporal_uint4(src_u32_ptr0);
    *dst_x16_ptr1 = load_nontemporal_uint4(src_u32_ptr1);
    *dst_x16_ptr2 = load_nontemporal_uint4(src_u32_ptr2);
    *dst_x16_ptr3 = load_nontemporal_uint4(src_u32_ptr3);
    *dst_x16_ptr4 = load_nontemporal_uint4(src_u32_ptr4);
    *dst_x16_ptr5 = load_nontemporal_uint4(src_u32_ptr5);
    *dst_x16_ptr6 = load_nontemporal_uint4(src_u32_ptr6);
    *dst_x16_ptr7 = load_nontemporal_uint4(src_u32_ptr7);
  }

  if (i + (4 * COPY_BYTES_PER_BLOCK) <= blockEnd) {
    const uint32_t* src_u32_ptr0 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr0 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr1 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr1 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr2 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr2 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr3 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr3 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    *dst_x16_ptr0 = load_nontemporal_uint4(src_u32_ptr0);
    *dst_x16_ptr1 = load_nontemporal_uint4(src_u32_ptr1);
    *dst_x16_ptr2 = load_nontemporal_uint4(src_u32_ptr2);
    *dst_x16_ptr3 = load_nontemporal_uint4(src_u32_ptr3);
  }

  if (i + (2 * COPY_BYTES_PER_BLOCK) <= blockEnd) {
    const uint32_t* src_u32_ptr0 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr0 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    const uint32_t* src_u32_ptr1 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr1 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    *dst_x16_ptr0 = load_nontemporal_uint4(src_u32_ptr0);
    *dst_x16_ptr1 = load_nontemporal_uint4(src_u32_ptr1);
  }

  if (i + COPY_BYTES_PER_THREAD <= blockEnd) {
    const uint32_t* src_u32_ptr0 = (const uint32_t*)(&src_ptr[i]);
    uint32x4_t* dst_x16_ptr0 = (uint32x4_t*)(&dst_ptr[i]);
    i += COPY_BYTES_PER_BLOCK;

    *dst_x16_ptr0 = load_nontemporal_uint4(src_u32_ptr0);
  }

  // non vectorized copy
  unsigned iEnd = std::min(i + COPY_BYTES_PER_THREAD, blockEnd);
  for (; i < iEnd; i++) {
    dst_ptr[i] = src_ptr[i];
  }
}


__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) __muillm_copy_p2p_kernel(
  const uint8_t* src_ptr,
  uint8_t* dst_ptr,
  unsigned N
) {
  __muillm_do_copy_p2p(src_ptr, dst_ptr, N);
}

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