#ifndef __MUILLM_STATIC_KVCACHE_KERNELS_H__
#define __MUILLM_STATIC_KVCACHE_KERNELS_H__

#include "../rope/rotary_position_layout.h"

#include <torch/extension.h>
#include <tuple>
#include <stdint.h>

// out: (narrowed) key, (narrowed) value
// (need to return value as we narrow the tensor of the cache)
std::tuple<at::Tensor, at::Tensor> muillm_static_kvcache_update(
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& cache_position,
    uint64_t seen_tokens
);

// out: query, (narrowed) key, (narrowed) value
// (need to return value as we narrow the tensor of the cache)
std::tuple<at::Tensor, at::Tensor, at::Tensor> muillm_rope_forward_static_cache(
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& cache_position,
    uint64_t seen_tokens
);

// out: query, (narrowed) key, (narrowed) value
// (need to return value as we narrow the tensor of the cache)
std::tuple<at::Tensor, at::Tensor, at::Tensor> muillm_complex_rope_forward_static_cache(
    torch::Tensor& position_embeds, // shape [B, T, embed_dim / 2, (2)] dtype complex float
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& cache_position,
    uint64_t seen_tokens
);

#endif /* __MUILLM_STATIC_KVCACHE_KERNELS_H__ */