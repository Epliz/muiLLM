#ifndef __MUILLM_SLIDING_KVCACHE_KERNELS_H__
#define __MUILLM_SLIDING_KVCACHE_KERNELS_H__

#include <torch/extension.h>
#include <tuple>
#include <stdint.h>

// out: (narrowed) key, (narrowed) value
// (need to return value as we narrow the tensor of the cache)
// (in some cases the returned tensors might be bigger than the window size
// e.g. large extra tokens coming in)
std::tuple<at::Tensor, at::Tensor> muillm_sliding_kvcache_update(
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& cache_position,
    uint64_t seen_tokens
);

#endif /* __MUILLM_SLIDING_KVCACHE_KERNELS_H__ */