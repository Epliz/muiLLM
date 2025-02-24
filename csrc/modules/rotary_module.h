#ifndef __MUILLM_ROTARY_MODULE_H__
#define __MUILLM_ROTARY_MODULE_H__

#include "../engine.h"
#include "kvcache.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>


struct MuillmRotaryEmbedding {
  // fields
  muillm_engine_t* engine;

  torch::Tensor cos_cached;
  torch::Tensor sin_cached;

  int layer_idx;

  // method
  MuillmRotaryEmbedding(
    muillm_engine_t* engine,
    int layer_idx,
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached
  );

  ~MuillmRotaryEmbedding();

  // output: q, k, v
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
    MuillmKVCache* cache,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& position_ids,
    std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
    torch::Tensor& cache_positions
  );

  // output: cos, sin 
  std::tuple<torch::Tensor, torch::Tensor> compute_rotary_pos_emb(
    torch::Tensor& x,
    torch::Tensor& position_ids
  );  
};
  
// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_rotary_embedding_module_ptr {
    MuillmRotaryEmbedding* ptr;
} muillm_rotary_embedding_module_ptr_t;

// init
muillm_rotary_embedding_module_ptr_t muillm_rotary_embedding_module_init_trampoline(
  muillm_engine_ptr engine,
  int layer_idx,
  torch::Tensor& cos_cached,
  torch::Tensor& sin_cached
);
  
// deinit
void muillm_rotary_embedding_module_deinit_trampoline(
  muillm_rotary_embedding_module_ptr_t module_ptr
);

// forward
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> muillm_rotary_embedding_module_forward_trampoline(
  muillm_rotary_embedding_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache,
  torch::Tensor& q_in,
  torch::Tensor& k_in,
  torch::Tensor& v_in,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_ROTARY_MODULE_H__ */