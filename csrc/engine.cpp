#include "engine.h"

muillm_error_t muillm_engine_init(
  muillm_engine_t** engine_ptr
) {
  muillm_engine_t* engine = new muillm_engine_t;

  for (int d = 0; d < MUILLM_MAX_GPUS; d++) {
    engine->gpu_infos[d] = nullptr;
  }

  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess) {
    return MUILLM_UNKNOWN_ERROR;
  }

  for (int d = 0; d < device_count; d++) {
    engine->gpu_infos[d] = new muillm_gpu_info_t;
    if (muillm_detect_gpu_properties(d, engine->gpu_infos[d]) != MUILLM_SUCCESS) {
      return MUILLM_UNKNOWN_ERROR;
    }
  }

  *engine_ptr = engine;
  return MUILLM_SUCCESS;
}