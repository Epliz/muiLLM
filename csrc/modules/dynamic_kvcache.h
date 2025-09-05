#ifndef __MUILLM_DYNAMIC_KVCACHE_H__
#define __MUILLM_DYNAMIC_KVCACHE_H__

#include "kvcache.h"


#include "../engine.h"

#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <optional>

struct MuillmDynamicKVCache: MuillmKVCache {
  // fields
  std::vector<torch::Tensor> key_cache;
  std::vector<torch::Tensor> value_cache;

  // method
  MuillmDynamicKVCache(
    muillm_engine_t* engine,
    std::vector<torch::Tensor>& key_cache,
    std::vector<torch::Tensor>& value_cache,
    int seen_tokens
  );
  
  virtual ~MuillmDynamicKVCache();
  
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
muillm_kvcache_module_ptr_t muillm_dynamic_kvcache_module_init_trampoline(
  muillm_engine_ptr engine,
  std::vector<torch::Tensor>& key_cache,
  std::vector<torch::Tensor>& value_cache,
  int seen_tokens
);

// update
std::tuple<torch::Tensor, torch::Tensor> muillm_dynamic_kvcache_module_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  std::optional<torch::Tensor> cache_positions,
  int layer_index
);

// rope update
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> muillm_dynamic_kvcache_module_rope_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& query_states,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  std::tuple<torch::Tensor, torch::Tensor> position_embeddings,
  torch::Tensor& cache_positions,
  int layer_index
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> muillm_dynamic_kvcache_module_complex_rope_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& query_states,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  torch::Tensor& position_embeddings,
  torch::Tensor& cache_positions,
  int layer_index
);

// deinit
void muillm_dynamic_kvcache_module_deinit_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
);

// sync
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> muillm_dynamic_kvcache_module_sync_back_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
);


#endif /* __MUILLM_DYNAMIC_KVCACHE_H__ */