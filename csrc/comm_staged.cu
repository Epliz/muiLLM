#include "comm_staged.h"

#include "comm.h"
#include "comm_base.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <stdint.h>
#include <stdio.h>

#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <errno.h>
#include <poll.h>

static void __allocate_wait_buffers(
  muillm_comm_staged_t* comm
);

static void __init_staged_recv(
    muillm_comm_staged_t* comm
);

static void __reallocate_staged_recv_buffer(
    muillm_comm_staged_t* comm,
    size_t required_recv_buffer_size,
    hipStream_t stream
);

void __ensure_staged_buffer_capacity(
    muillm_comm_staged_t* comm,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  size_t required_recv_buffer_size = ALIGN_UP(__comm_size(datatype, count), GPU_CACHELINE_SIZE);

  if (comm->recv_buffer_size >= required_recv_buffer_size) {
    // we have sufficient buffer space
    return;
  }

  printf("(rank %d) Reallocating recv_buffers\n", local_rank);

  // TODO error handling
  __reallocate_staged_recv_buffer(
    (muillm_comm_staged_t*) comm,
    required_recv_buffer_size,
    stream
  );

  printf("(rank %d) Allocated receive buffers\n", local_rank);
}


muillm_comm_error_t __init_staged_comm(
    int world_size,
    int local_size,
    int rank,
    int local_rank,
    muillm_comm_staged_t** comm_ptr
) {
  if (world_size != local_size) {
    // we currently ony support single machine, so
    // we should fail
    return MUILLM_COMM_UNSUPPORTED_SIZE;
  }

  printf("(rank %d local_rank %d) Initializing comm for world_size %d local_size %d ...\n", rank, local_rank, world_size, local_size);

  muillm_comm_error_t error;

  muillm_comm_method_t transfer_method = MUILLM_COMM_METHOD_STAGED_TRANSFER;

  // create the comm object
  muillm_comm_staged_t* comm = nullptr;

  comm = new muillm_comm_staged_t;
  comm->transfer_method = transfer_method;

  comm->world_size = world_size;
  comm->local_size = local_size;
  comm->rank = rank;
  comm->local_rank = local_rank;

  // establish the local socket connection
  printf("(rank %d) Opening local socket...\n", local_rank);
  error = __open_local_socket(comm, local_size, local_rank);
  if (error != MUILLM_COMM_SUCCESS) {
    return error;
  }
  printf("(rank %d) Opened local socket\n", local_rank);

  
  if (hipSetDevice(local_rank) != hipSuccess) {
    printf("(rank %d) Failed to set device\n", local_rank);
    return MUILLM_UNKNOWN_ERROR;
  }

  // allocate the wait buffer
  // TODO: check for errors
  __allocate_wait_buffers((muillm_comm_staged_t*) comm);
  // TODO: check for errors
  __init_staged_recv((muillm_comm_staged_t*) comm);

  // allocate an initial buffer
  // TODO: check for errors
  __ensure_staged_buffer_capacity(comm, 1024*1024, MUILLM_COMM_FP32, 0);

  // return the comm object
  printf("(rank %d) Created comm %p\n", local_rank, comm);
  
  *comm_ptr = comm;

  return MUILLM_COMM_SUCCESS;
}

static inline void __swap_staged_recv_buffer_sets(muillm_comm_staged_t* comm) {
  void* staged_recv_buffer_temp = comm->staged_recv_buffer_set.staged_recv_buffer;
  void* staged_recv_buffer_cpu_temp = comm->staged_recv_buffer_set.staged_recv_buffer_cpu;

  comm->staged_recv_buffer_set.staged_recv_buffer = comm->second_staged_recv_buffer_set.staged_recv_buffer;
  comm->staged_recv_buffer_set.staged_recv_buffer_cpu = comm->second_staged_recv_buffer_set.staged_recv_buffer_cpu;

  comm->second_staged_recv_buffer_set.staged_recv_buffer = staged_recv_buffer_temp;
  comm->second_staged_recv_buffer_set.staged_recv_buffer_cpu = staged_recv_buffer_cpu_temp;
}

static void __allocate_locked_shared_cpu_mem(
    muillm_comm_staged_t* comm,
    size_t size,
    void** shm_addr_ptr,
    void** device_ptr_ptr
  ) {
  int local_rank = comm->local_rank;

  int shm_id;
  void *shm_addr;

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

    if (mlock(shm_addr, size) != 0) {
      // TODO: return error code
      printf("(rank %d) could not lock shared memory\n", local_rank);
      return;
    }

    // make the memory be deleted once all processes have detached from it
    // (memory is automatically detached on process exit)
    if (shmctl(shm_id, IPC_RMID, NULL) != 0) {
      // TODO: return error code
      return;
    }
    printf("(rank %d) allocated shared address\n", local_rank);
  }

  // get the share memory ID on all ranks
  __local_socket_broadcast(comm, /*src*/ 0, &shm_id, sizeof(int));

  if (local_rank != 0) {
    shm_addr = shmat(shm_id, NULL, 0);
    if (shm_addr == (void *) -1) {
      // TODO: return error code
      printf("(rank %d) could not get SHM addr\n", local_rank);
      return;
    }
    printf("(rank %d) got shared address\n", local_rank);
  }

  // register the memory for use with HIP
  if (hipHostRegister(shm_addr, size, hipHostRegisterPortable | hipHostRegisterMapped) != hipSuccess) {
    // TODO: return error code
      printf("(rank %d) could not register host address\n", local_rank);
    return;
  }

  printf("(rank %d) registered memory\n", local_rank);

  // get the device pointer after registration
  if (hipHostGetDevicePointer((void**)device_ptr_ptr, shm_addr, 0) != hipSuccess) {
    // TODO: return error code
      printf("(rank %d) could not get device pointer\n", local_rank);
    return;
  }

  printf("(rank %d) shared memory fully allocated\n", local_rank);
  
  // return
  *shm_addr_ptr = shm_addr;
}

void __deallocate_locked_shared_cpu_mem(
    muillm_comm_staged_t* comm,
    void* addr
  ) {
  int local_rank = comm->local_rank;

  if (hipHostUnregister(addr) != hipSuccess) {
    // TODO: return error code
    printf("(rank %d) could not unregister host address\n", local_rank);
    return;
  }

  if (shmdt(addr) != 0) {
    printf("(rank %d) could detach shared address\n", local_rank);
    return;
  }
}

static void __allocate_wait_buffers(
  muillm_comm_staged_t* comm
) {
  int local_rank = comm->local_rank;

  comm->wait_buffer_set.wait_buffer = nullptr;
  comm->second_wait_buffer_set.wait_buffer = nullptr;
  comm->seq_no = 1;

  // allocate some shared pinned memory on the host so that the GPUs
  // see the value change during kernel execution

  // we have to have each GPU write into different cache lines
  // to avoid lines overwrites
  size_t wait_buffer_size = CPU_CACHELINE_SIZE * comm->world_size;

  __allocate_locked_shared_cpu_mem(
    comm,
    wait_buffer_size,
    &comm->wait_buffer_set.wait_buffer_cpu,
    (void**)&comm->wait_buffer_set.wait_buffer
  );
  __allocate_locked_shared_cpu_mem(
    comm,
    wait_buffer_size,
    &comm->second_wait_buffer_set.wait_buffer_cpu,
    (void**)&comm->second_wait_buffer_set.wait_buffer
  );

  // initialize to 0 to avoid any bad suprise and missed syncs
  memset(comm->wait_buffer_set.wait_buffer_cpu, 0, wait_buffer_size);
  memset(comm->second_wait_buffer_set.wait_buffer_cpu, 0, wait_buffer_size);

  printf("(rank %d) Allocated wait buffers\n", local_rank);
}

static void __init_staged_recv(
    muillm_comm_staged_t* comm
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  printf("(rank %d) Initializing staged transfers\n", local_rank);

  comm->recv_buffer_size = 0;
  comm->staged_recv_buffer_set.staged_recv_buffer = nullptr;
  comm->staged_recv_buffer_set.staged_recv_buffer_cpu = nullptr;
  comm->second_staged_recv_buffer_set.staged_recv_buffer = nullptr;
  comm->second_staged_recv_buffer_set.staged_recv_buffer_cpu = nullptr;
}

// returns the number of receive buffers needed to do any of the operations
static inline size_t __comm_buffer_set_size(
    muillm_comm_staged_t* comm
) {
  // we need as many buffers as local ranks
  return comm->local_size;
}

__device__ void __device_wait(
    volatile int* wait_buffer,
    int seq_no,
    int local_size,
    int local_rank
) {
  // wait for the value (through uncached pinned CPU memory) to be posted
  // by other GPUs
  /*
  for (int remote_rank = 0; remote_rank < local_size; remote_rank++) {
    if (remote_rank == local_rank)
      continue;

    while (wait_buffer[remote_rank * INT_CACHELINE_SIZE] != seq_no);
  }
  */

  // wait for the value (through uncached pinned CPU memory) to be posted
  // by other GPUs
  volatile int* wait_slots[8];

  if (local_size > 8) {
    // error
    return;
  }

  int num_wait_slots = 0;

  // wait for the value (through uncached pinned CPU memory) to be posted
  // by other GPUs
  for (int remote_rank = 0; remote_rank < local_size; remote_rank++) {
    if (remote_rank == local_rank)
      continue;

    wait_slots[num_wait_slots] = &wait_buffer[remote_rank * INT_CACHELINE_SIZE];
    num_wait_slots++;
  }

  // now do the waits
  while (num_wait_slots == 8) {
    int val0 = *wait_slots[0];
    int val1 = *wait_slots[1];
    int val2 = *wait_slots[2];
    int val3 = *wait_slots[3];
    int val4 = *wait_slots[4];
    int val5 = *wait_slots[5];
    int val6 = *wait_slots[6];
    int val7 = *wait_slots[7];

    // check what waits are over, and recompact the wait list
    int written_slots = 0;

    if (val0 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[0];
      written_slots++;
    }

    if (val1 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[1];
      written_slots++;
    }

    if (val2 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[2];
      written_slots++;
    }

    if (val3 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[3];
      written_slots++;
    }

    if (val4 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[4];
      written_slots++;
    }

    if (val5 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[5];
      written_slots++;
    }

    if (val6 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[6];
      written_slots++;
    }

    if (val7 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[7];
      written_slots++;
    }
  }

  while (num_wait_slots == 7) {
    int val0 = *wait_slots[0];
    int val1 = *wait_slots[1];
    int val2 = *wait_slots[2];
    int val3 = *wait_slots[3];
    int val4 = *wait_slots[4];
    int val5 = *wait_slots[5];
    int val6 = *wait_slots[6];

    // check what waits are over, and recompact the wait list
    int written_slots = 0;

    if (val0 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[0];
      written_slots++;
    }

    if (val1 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[1];
      written_slots++;
    }

    if (val2 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[2];
      written_slots++;
    }

    if (val3 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[3];
      written_slots++;
    }

    if (val4 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[4];
      written_slots++;
    }

    if (val5 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[5];
      written_slots++;
    }

    if (val6 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[6];
      written_slots++;
    }
  }

  while (num_wait_slots == 6) {
    int val0 = *wait_slots[0];
    int val1 = *wait_slots[1];
    int val2 = *wait_slots[2];
    int val3 = *wait_slots[3];
    int val4 = *wait_slots[4];
    int val5 = *wait_slots[5];

    // check what waits are over, and recompact the wait list
    int written_slots = 0;

    if (val0 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[0];
      written_slots++;
    }

    if (val1 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[1];
      written_slots++;
    }

    if (val2 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[2];
      written_slots++;
    }

    if (val3 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[3];
      written_slots++;
    }

    if (val4 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[4];
      written_slots++;
    }

    if (val5 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[5];
      written_slots++;
    }
  }

  while (num_wait_slots == 5) {
    int val0 = *wait_slots[0];
    int val1 = *wait_slots[1];
    int val2 = *wait_slots[2];
    int val3 = *wait_slots[3];
    int val4 = *wait_slots[4];

    // check what waits are over, and recompact the wait list
    int written_slots = 0;

    if (val0 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[0];
      written_slots++;
    }

    if (val1 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[1];
      written_slots++;
    }

    if (val2 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[2];
      written_slots++;
    }

    if (val3 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[3];
      written_slots++;
    }

    if (val4 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[4];
      written_slots++;
    }
  }

  while (num_wait_slots == 4) {
    int val0 = *wait_slots[0];
    int val1 = *wait_slots[1];
    int val2 = *wait_slots[2];
    int val3 = *wait_slots[3];

    // check what waits are over, and recompact the wait list
    int written_slots = 0;

    if (val0 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[0];
      written_slots++;
    }

    if (val1 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[1];
      written_slots++;
    }

    if (val2 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[2];
      written_slots++;
    }

    if (val3 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[3];
      written_slots++;
    }
  }


  while (num_wait_slots == 3) {
    int val0 = *wait_slots[0];
    int val1 = *wait_slots[1];
    int val2 = *wait_slots[2];

    // check what waits are over, and recompact the wait list
    int written_slots = 0;

    if (val0 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[0];
      written_slots++;
    }

    if (val1 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[1];
      written_slots++;
    }

    if (val2 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[2];
      written_slots++;
    }
  }


  while (num_wait_slots == 2) {
    int val0 = *wait_slots[0];
    int val1 = *wait_slots[1];

    // check what waits are over, and recompact the wait list
    int written_slots = 0;

    if (val0 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[0];
      written_slots++;
    }

    if (val1 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    } else {
      // add it again in the list
      wait_slots[written_slots] = wait_slots[1];
      written_slots++;
    }
  }


  while (num_wait_slots == 1) {
    int val0 = *wait_slots[0];
    // check what waits are over
    if (val0 == seq_no) {
      // remove from the wait list by not writing it in the new list
      num_wait_slots--;
    }
  }
}

// make the GPU of the given rank to block until the others have reached the barrier
// too
__global__ void __barrier_kernel(
    volatile int* wait_buffer,
    int seq_no,
    int local_size,
    int local_rank
) {
  if (threadIdx.x != 0) {
    return;
  }
  //*
  // post the value to the other GPUs (through uncached pinned CPU memory)
  wait_buffer[local_rank * INT_CACHELINE_SIZE] = seq_no;

  __threadfence_system();

  __device_wait(
    wait_buffer,
    seq_no,
    local_size,
    local_rank
  );

  // wait done
}

void __local_staged_gpu_barrier(
    muillm_comm_staged_t* comm,
    hipStream_t stream) {
  // make the GPU of this rank to block until the others have reached the barrier
  // too
  const int threads_per_blocks = 64;
  const int num_blocks = 1;

  int local_rank = comm->local_rank;

  __barrier_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    comm->wait_buffer_set.wait_buffer,
    comm->seq_no,
    comm->local_size,
    comm->local_rank
  );


  comm->seq_no++;

  // swap the sets of wait buffers to avoid livelocks
  // when a GPU already writes the next seq_no while another GPU is waiting
  int* wait_buffer_temp = comm->wait_buffer_set.wait_buffer;
  void* wait_buffer_cpu_temp = comm->wait_buffer_set.wait_buffer_cpu;

  comm->wait_buffer_set.wait_buffer = comm->second_wait_buffer_set.wait_buffer;
  comm->wait_buffer_set.wait_buffer_cpu = comm->second_wait_buffer_set.wait_buffer_cpu;

  comm->second_wait_buffer_set.wait_buffer = wait_buffer_temp;
  comm->second_wait_buffer_set.wait_buffer_cpu = wait_buffer_cpu_temp;
}

static void __reallocate_staged_recv_buffer_set(
    muillm_comm_staged_t* comm,
    size_t required_recv_buffer_size,
    hipStream_t stream,
    muillm_comm_staged_recv_buffer_set_t* buffer_set
  ) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  // unsufficient buffer space: we have to allocate some bigger space
  if (buffer_set->staged_recv_buffer_cpu != nullptr) {
    // we need to synchronize the ranks and block the  CPU so that we can deallocate
    // the previous receive buffers
    __local_staged_gpu_barrier(comm, stream);

    // synchronize to make sure no GPU is going to reference the previous memory
    if (hipDeviceSynchronize() != hipSuccess) {
      printf("(rank %d) Error while synchronizing device\n", local_rank);
      return;
    }

    // deallocate the previous memory
    __deallocate_locked_shared_cpu_mem(
      comm,
      buffer_set->staged_recv_buffer_cpu
    );
  }

  // then allocate new receive buffers
  buffer_set->staged_recv_buffer = nullptr;
  buffer_set->staged_recv_buffer_cpu = nullptr;

  // each GPU will write in a different buffer
  size_t num_recv_buffers = __comm_buffer_set_size(comm);
  // align each buffer on a 2MiB as it is the shareable page size
  required_recv_buffer_size = ALIGN_UP(required_recv_buffer_size, GPU_SHAREABLE_PAGE_SIZE);

  size_t all_required_recv_buffer_size = num_recv_buffers * required_recv_buffer_size;


  __allocate_locked_shared_cpu_mem(
    comm,
    all_required_recv_buffer_size,
    &buffer_set->staged_recv_buffer_cpu,
    &buffer_set->staged_recv_buffer
  );

  comm->recv_buffer_size = all_required_recv_buffer_size / local_size;
}


static void __reallocate_staged_recv_buffer(
    muillm_comm_staged_t* comm,
    size_t required_recv_buffer_size,
    hipStream_t stream
  ) {
  __reallocate_staged_recv_buffer_set(comm, required_recv_buffer_size, stream, &comm->staged_recv_buffer_set);
  __reallocate_staged_recv_buffer_set(comm, required_recv_buffer_size, stream, &comm->second_staged_recv_buffer_set);
}

#define THREADS_PER_BLOCK 256


// each threads can copy 16 bytes
#define BYTES_PER_THREAD 16
#define BYTES_PER_BLOCK (THREADS_PER_BLOCK * BYTES_PER_THREAD)

typedef struct uint32x4{
  uint32_t x, y, z, w;
} uint32x4_t;

__global__ void muillm_copy_kernel(
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

static inline void __local_gpu_staged_send_async_copy(
    muillm_comm_staged_t* comm,
    void* src_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  // copy the memories around
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  size_t recv_buffer_size = __comm_size(datatype, count);
  // align to avoid cache line crossing
  // (might avoid correctness issues due to crossing PCIe writes)
  size_t padded_recv_buffer_size = ALIGN_UP(recv_buffer_size, GPU_CACHELINE_SIZE);

  // copy our memory to the CPU buffer
  // we have to write at this place:
  size_t dest_offset = padded_recv_buffer_size * local_rank;
  uint8_t* dst_ptr = ((uint8_t*)comm->staged_recv_buffer_set.staged_recv_buffer) + dest_offset;

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(recv_buffer_size, BYTES_PER_BLOCK);

  // a copy kernel is faster than a hipMemcpyAsync
  muillm_copy_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint8_t*) src_ptr,
    (uint8_t*) dst_ptr,
    recv_buffer_size
  );
}


__global__ void muillm_reduce_sum_staged_fp32_kernel(
    const float* remote_buffs,
    float* dest_buff,
    unsigned local_size,
    unsigned N,
    unsigned padded_recv_buffer_size
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float r = 0.f;

    // data to reduce is packed in the receive buffer
    // TODO: vectorize, unroll
    for (unsigned b = 0; b < local_size; b++) {
      size_t offset = padded_recv_buffer_size * b;
      const float* remote_buff = (const float*)((const uint8_t*)remote_buffs + offset);

      r += remote_buff[i];
    }

    dest_buff[i] = r;
  }
}

__global__ void muillm_reduce_sum_staged_fp16_kernel(
    const half* remote_buffs,
    half* dest_buff,
    unsigned local_size,
    unsigned N,
    unsigned padded_recv_buffer_size
) {
  int warpCounts = THREADS_PER_BLOCK / warpSize;
  int warpId = threadIdx.x / warpSize;
  int laneId = threadIdx.x % warpSize;

  unsigned i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (i < N) {
    float r =0.f;

    // data to reduce is packed in the receive buffer
    // TODO: vectorize, unroll
    for (unsigned b = 0; b < local_size; b++) {
      size_t offset = padded_recv_buffer_size * b;
      const half* remote_buff = (const half*)((const uint8_t*)remote_buffs + offset);

      r += __half2float(remote_buff[i]);
    }

    dest_buff[i] = __float2half(r);
  }
}

void __all_reduce_sum_staged(
    muillm_comm_staged_t* comm,
    void* src_ptr,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  // if (hipDeviceSynchronize() != hipSuccess) {
  //   printf("(rank %d) hipSync failed before starting reduce\n", local_rank);
  // }

  //printf("(rank %d) starting reduce sum...\n", local_rank);

  // first, make sure we have enough buffer space
  __ensure_staged_buffer_capacity(comm, count, datatype, stream);

  // gather all the pieces
  __local_gpu_staged_send_async_copy(
    comm,
    src_ptr,
    count,
    datatype,
    stream
  );

//   if (hipDeviceSynchronize() != hipSuccess) {
//     printf("(rank %d) hipSync failed after send\n", local_rank);
//   }

  // sync the GPUs
  __local_staged_gpu_barrier(comm, stream);

//   if (hipDeviceSynchronize() != hipSuccess) {
//     printf("(rank %d) hipSync failed after barried\n", local_rank);
//   }

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(count, threads_per_blocks);

  size_t recv_buffer_size = __comm_size(datatype, count);
  // align to avoid cache line crossing
  // (might avoid correctness issues due to crossing PCIe writes)
  size_t padded_recv_buffer_size = ALIGN_UP(recv_buffer_size, GPU_CACHELINE_SIZE);

  // do the reductions
  if (datatype == MUILLM_COMM_FP16) {
    muillm_reduce_sum_staged_fp16_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      (const half*)comm->staged_recv_buffer_set.staged_recv_buffer,
      (half*)dst_ptr,
      local_size,
      count,
      padded_recv_buffer_size
    );
  } else if (datatype == MUILLM_COMM_FP32) {
    muillm_reduce_sum_staged_fp32_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
      (const float*)comm->staged_recv_buffer_set.staged_recv_buffer,
      (float*)dst_ptr,
      local_size,
      count,
      padded_recv_buffer_size
    );
  } else {
    // TODO: error
    printf("unsupported type\n");
  }

  // sync the GPUs
  //__local_gpu_barrier(comm, stream);


  // if (hipDeviceSynchronize() != hipSuccess) {
  //   printf("(rank %d) hipSync failed after second barried\n", local_rank);
  // }

  // swap buffer sets to avoid overwrites
  __swap_staged_recv_buffer_sets(comm);


  //printf("(rank %d) finished reduce sum.\n", local_rank);
}


static inline void __local_gpu_staged_receive_async_copy(
    muillm_comm_staged_t* comm,
    int src_rank,
    void* dst_ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  // copy the memories around
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  size_t recv_buffer_size = __comm_size(datatype, count);
  // align to avoid cache line crossing
  // (might avoid correctness issues due to crossing PCIe writes)
  size_t padded_recv_buffer_size = ALIGN_UP(recv_buffer_size, GPU_CACHELINE_SIZE);

  // copy our memory from the CPU buffer
  // we have to read at this place:
  size_t src_offset = padded_recv_buffer_size * src_rank;
  uint8_t* src_ptr = ((uint8_t*)comm->staged_recv_buffer_set.staged_recv_buffer) + src_offset;

  const int threads_per_blocks = THREADS_PER_BLOCK;
  const int num_blocks = DIV_ROUND_UP(recv_buffer_size, BYTES_PER_BLOCK);

  // a copy kernel is faster than a hipMemcpyAsync
  muillm_copy_kernel<<<num_blocks, threads_per_blocks, 0, stream>>>(
    (const uint8_t*) src_ptr,
    (uint8_t*) dst_ptr,
    recv_buffer_size
  );
}

void __broadcast_staged(
    muillm_comm_staged_t* comm,
    int src_rank,
    void* ptr,
    size_t count,
    muillm_comm_datatype_t datatype,
    hipStream_t stream
) {
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  // if (hipDeviceSynchronize() != hipSuccess) {
  //   printf("(rank %d) hipSync before starting broadcasting\n", local_rank);
  // }

  //printf("(rank %d) starting broadcasting...\n", local_rank);

  // first, make sure we have enough buffer space
  __ensure_staged_buffer_capacity(comm, count, datatype, stream);

  // send if we are the broadcaster
  if (src_rank == local_rank) {
    __local_gpu_staged_send_async_copy(
      comm,
      ptr,
      count,
      datatype,
      stream
    );

    // if (hipDeviceSynchronize() != hipSuccess) {
    //   printf("(rank %d) hipSync after send\n", local_rank);
    // }
  }

  // sync the GPUs
  __local_staged_gpu_barrier(comm, stream);

//   if (hipDeviceSynchronize() != hipSuccess) {
//     printf("(rank %d) hipSync after barrier\n", local_rank);
//   }

  // receive if we are not the broadcaster
  if (src_rank != local_rank) {
    __local_gpu_staged_receive_async_copy(
      comm,
      src_rank,
      ptr,
      count,
      datatype,
      stream
    );
  }

  // if (hipDeviceSynchronize() != hipSuccess) {
  //   printf("(rank %d) hipSync after receive\n", local_rank);
  // }

  // swap buffer sets to avoid overwrites
  __swap_staged_recv_buffer_sets(comm);

  //printf("(rank %d) finished broadcast\n", local_rank);
}