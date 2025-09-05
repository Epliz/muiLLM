#ifndef __MUILLM_DYNAMIC_KVCACHE_KERNELS_H__
#define __MUILLM_DYNAMIC_KVCACHE_KERNELS_H__

#include "../rope/rotary_position_layout.h"

#include <torch/extension.h>
#include <tuple>
#include <stdint.h>

// out: (narrowed) key, (narrowed) value
// (need to return value as we narrow the tensor of the cache)
std::tuple<at::Tensor, at::Tensor> muillm_dynamic_kvcache_update(
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& prev_k_cache,
    torch::Tensor& prev_v_cache
);

// out: query, k_cache_out, v_cache_out
std::tuple<at::Tensor, at::Tensor, at::Tensor> muillm_rope_forward_dynamic_cache(
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& prev_k_cache,
    torch::Tensor& prev_v_cache
);

// out: query, k_cache_out, v_cache_out
std::tuple<at::Tensor, at::Tensor, at::Tensor> muillm_complex_rope_forward_dynamic_cache(
    torch::Tensor& position_embeds,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& prev_k_cache,
    torch::Tensor& prev_v_cache
);

#endif /* __MUILLM_DYNAMIC_KVCACHE_KERNELS_H__ */