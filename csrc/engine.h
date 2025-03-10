#ifndef __MUILLM_ENGINE_H__
#define __MUILLM_ENGINE_H__

#include "base.h"
#include "gpu_info.h"

#include <hip/hip_runtime.h>

typedef struct muillm_engine {
  muillm_gpu_info_t* gpu_infos[MUILLM_MAX_GPUS];
} muillm_engine_t;
  
// needed because Pybind11 can't seem to be able to deal with opaque pointers
struct muillm_engine_ptr {
  muillm_engine_t* engine_ptr;
};


muillm_error_t muillm_engine_init(
  muillm_engine_t** engine
);

#endif /* __MUILLM_ENGINE_H__ */