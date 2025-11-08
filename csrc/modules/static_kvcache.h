#ifndef __MUILLM_STATIC_KVCACHE_H__
#define __MUILLM_STATIC_KVCACHE_H__

#include "kvcache.h"

#include "../engine.h"

#include <torch/torch.h>
#include <vector>

struct MuillmStaticKVCache: MuillmKVCache {
  // fields
  std::vector<torch::Tensor> key_cache;
  std::vector<torch::Tensor> value_cache;

  // method
  MuillmStaticKVCache(
    muillm_engine_t* engine,
    std::vector<torch::Tensor>& key_cache,
    std::vector<torch::Tensor>& value_cache,
    int seen_tokens
  );
  
  virtual ~MuillmStaticKVCache();
  
  std::tuple<torch::Tensor, torch::Tensor> update(
    torch::Tensor& key_states,
    torch::Tensor& value_states,
    torch::Tensor& cache_positions,
    int layer_index
  );

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rope_update(
    torch::Tensor& query_states,
    torch::Tensor& key_states,
    torch::Tensor& value_states,
    std::tuple<torch::Tensor, torch::Tensor>& position_embeddings,
    torch::Tensor& cache_positions,
    int layer_index
  );

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> complex_rope_update(
    torch::Tensor& query_states,
    torch::Tensor& key_states,
    torch::Tensor& value_states,
    torch::Tensor& position_embeddings,
    torch::Tensor& cache_positions,
    int layer_index
  );
};

// init
muillm_kvcache_module_ptr_t muillm_static_kvcache_module_init_trampoline(
  muillm_engine_ptr engine,
  std::vector<torch::Tensor>& key_cache,
  std::vector<torch::Tensor>& value_cache,
  int seen_tokens
);

// update
std::tuple<torch::Tensor, torch::Tensor> muillm_static_kvcache_module_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  torch::Tensor& cache_positions,
  int layer_index
);

// rope update
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> muillm_static_kvcache_module_rope_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& query_states,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  std::tuple<torch::Tensor, torch::Tensor> position_embeddings,
  torch::Tensor& cache_positions,
  int layer_index
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> muillm_static_kvcache_module_complex_rope_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& query_states,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  torch::Tensor& position_embeddings,
  torch::Tensor& cache_positions,
  int layer_index
);

// deinit
void muillm_static_kvcache_module_deinit_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
);

// sync
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> muillm_static_kvcache_module_sync_back_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
);

#endif /* __MUILLM_STATIC_KVCACHE_H__ */