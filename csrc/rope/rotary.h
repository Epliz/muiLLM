#ifndef __MUILLM_ROTARY_KERNELS_H__
#define __MUILLM_ROTARY_KERNELS_H__

#include <torch/extension.h>
#include <tuple>
#include <optional>
#include <stdint.h>

#include "rotary_position_layout.h"

// out: cos, sin
std::tuple<at::Tensor, at::Tensor> muillm_compute_rotary_embed_positions(
    torch::Tensor& x,
    torch::Tensor& position_ids, // shape [batch_size, seq_len]
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached
);

// out: query, key
std::tuple<at::Tensor, at::Tensor> muillm_rope_forward_no_cache(
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in
);

// out: query, key
// Rotary embedding (complex multiply, like apply_rotary_emb)
std::tuple<at::Tensor, at::Tensor> muillm_complex_rope_forward_no_cache(
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& position_embeds // always complex floats
);

#endif /* __MUILLM_ROTARY_KERNELS_H__ */