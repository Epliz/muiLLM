#ifndef __MUILLM_SYNC_HPP__
#define __MUILLM_SYNC_HPP__

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <stddef.h>

typedef struct muillm_synchronizer muillm_synchronizer_t;

muillm_synchronizer_t* muillm_sync_init();

void muillm_sync_copy(
    muillm_synchronizer_t* sync,
    hipStream_t stream,
    void* dst,
    const void* src,
    size_t count
);

#endif // __MUILLM_SYNC_HPP__