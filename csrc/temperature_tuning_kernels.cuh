#ifndef __MUILLM_TEMPERATURETUNING_KERNELS_H__
#define __MUILLM_TEMPERATURETUNING_KERNELS_H__

#include <torch/extension.h>

at::Tensor muillm_apply_temperature_tuning(
    torch::Tensor query_states,
    torch::Tensor cache_position,
    float attn_scale,
    float floor_scale
);

#endif /* __MUILLM_TEMPERATURETUNING_KERNELS_H__ */
