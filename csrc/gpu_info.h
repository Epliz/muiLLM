#ifndef __MUILLM_GPU_INFO_H__
#define __MUILLM_GPU_INFO_H__

#include "base.h"

typedef enum muillm_gpu_family {
  MUILLM_GPU_FAMILY_UNKNOWN = 0,
  MUILLM_GPU_FAMILY_RDNA,
  MUILLM_GPU_FAMILY_CDNA
} muillm_gpu_family_t;

typedef enum muillm_gpu_arch {
  MUILLM_GPU_ARCH_UNKNOWN = 0,
  MUILLM_GPU_ARCH_RDNA1,
  MUILLM_GPU_ARCH_RDNA2,
  MUILLM_GPU_ARCH_RDNA3,
  MUILLM_GPU_ARCH_RDNA4,
  MUILLM_GPU_ARCH_MI100,
  MUILLM_GPU_ARCH_MI200,
  MUILLM_GPU_ARCH_MI300,
  MUILLM_GPU_ARCH_MI400
} muillm_gpu_arch_t;

typedef struct muillm_gpu_info {
  muillm_gpu_arch_t arch;
  muillm_gpu_family_t family;
  // number of threads in warp: 64 for gfx9, 32 for gfx10+
  int warp_size;
  // number of simd lanes on a device ("cuda cores")
  int simd_lanes;
} muillm_gpu_info_t;

muillm_error_t muillm_detect_gpu_properties(
  int device,
  muillm_gpu_info_t* gpu_info
);

#endif /* __MUILLM_GPU_INFO_H__ */