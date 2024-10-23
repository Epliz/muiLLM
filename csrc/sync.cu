#include "sync.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <xmmintrin.h>  // for _mm_pause

#include <stdint.h>
#include <string.h>
#include <stdio.h>

struct muillm_synchronizer {
  // locked CPU memory to do transfers to
  void* staging_buffer;
  // locked CPU memory for the GPUs to signal completions
  volatile int* signal_buffer;

  // size of the staging buffer in bytes
  size_t staging_buffer_size;

  // current sequential number for the signal
  int seq_no;
};

static void __allocate_signal_buffer(muillm_synchronizer_t* sync) {
  if (hipHostMalloc((void**) &sync->signal_buffer, sizeof(int), 0) != hipSuccess) {
    printf("allocating the signal buffer failedd!\n");
  }
  // initialize to 0 as the first sequential number to wait on will be 1
  memset((void*) sync->signal_buffer, 0, sizeof(int));
}

static size_t next_power_of_2(size_t n) {
  size_t p = 1;

  while (p < n) {
    p *= 2;
  }

  return p;
}

static void __ensure_staging_buffer_capacity(
  muillm_synchronizer_t* sync,
  size_t count,
  hipStream_t stream
) {
  if (sync->staging_buffer_size >= count) {
    // enough space
    return;
  }

  // deallocate the previous memory
  if ((sync->staging_buffer != nullptr) && (hipHostFree(sync->staging_buffer) != hipSuccess)) {
    printf("freeing the staging buffer failedd!\n");
  }

  // find the next power of two for the size to not have to do re-allocations over and over
  count = next_power_of_2(count);

  // allocate a new buffer of the required size
  if (hipHostMalloc((void**) &sync->staging_buffer, count, 0) != hipSuccess) {
    printf("allocating the staging buffer failedd!\n");
  }
  sync->staging_buffer_size = count;
}

muillm_synchronizer_t* muillm_sync_init() {

  // create the comm object
  muillm_synchronizer_t* sync = new muillm_synchronizer_t;
  sync->staging_buffer = nullptr;
  sync->signal_buffer = nullptr;
  sync->staging_buffer_size = 0;
  sync->seq_no = 1; // first value we will wait on

  // allocate the wait buffer
  __allocate_signal_buffer(sync);

  // allocate an initial buffer with an initial size of 4kB
  __ensure_staging_buffer_capacity(sync, 4*1024, 0);

  return sync;
}

// make the GPU signal the completion
// (we use a kernel instead of the signals the HIP runtime uses as it seems that the
// dedicated HW block might hang sometimes?)
__global__ void __signal_kernel(
    volatile int* signal_buffer,
    int seq_no
) {
  if (threadIdx.x != 0) {
    return;
  }
  *signal_buffer = seq_no;
}

static void __spin_pause() {
  _mm_pause();
}

static void __spin_gpu_cpu_sync(
    muillm_synchronizer_t* sync,
    hipStream_t stream) {
  int wait_no = sync->seq_no;

  const int threads_per_blocks = 64;
  const int num_blocks = 1;
  __signal_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    sync->signal_buffer,
    wait_no
  );

  sync->seq_no++; // next sequential number we will wait on

  // spin until the GPU has signaled completion
  while (*sync->signal_buffer != wait_no) {
    __spin_pause();
  }
}


void muillm_sync_copy(
    muillm_synchronizer_t* sync,
    hipStream_t stream,
    void* dst,
    const void* src,
    size_t count
) {
  // ensure the staging buffer is big enough
  __ensure_staging_buffer_capacity(sync, count, stream);

  // do the copy to the staging buffer
  if (hipMemcpyAsync(sync->staging_buffer, src, count, hipMemcpyDeviceToHost, stream) != hipSuccess) {
    printf("async copy failed\n");
  }

  // sync the CPU with the GPU
  __spin_gpu_cpu_sync(sync, stream);

  // do the final copy from the staging buffer to the CPU memory
  memcpy(dst, sync->staging_buffer, count);
}