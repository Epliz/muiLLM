#ifndef __MUILLM_HYBRID_CHUNKED_KVCACHE_H__
#define __MUILLM_HYBRID_CHUNKED_KVCACHE_H__

#include "kvcache.h"

#include "../engine.h"

#include <torch/torch.h>
#include <vector>

struct MuillmHybridChunkedKVCache: MuillmKVCache {
  // fields
  std::vector<torch::Tensor> key_cache;
  std::vector<torch::Tensor> value_cache;
  std::vector<bool> is_sliding;

  int window_size;

  // method
  MuillmHybridChunkedKVCache(
    muillm_engine_t* engine,
    std::vector<torch::Tensor>& key_cache,
    std::vector<torch::Tensor>& value_cache,
    std::vector<bool>& is_sliding,
    int window_size,
    int seen_tokens
  );
  
  virtual ~MuillmHybridChunkedKVCache();


  std::tuple<torch::Tensor, torch::Tensor> update(
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor& cache_positions,
    int layer_index
  );
};

// init
muillm_kvcache_module_ptr_t muillm_hybrid_chunked_kvcache_module_init_trampoline(
  muillm_engine_ptr engine,
  std::vector<torch::Tensor>& key_cache,
  std::vector<torch::Tensor>& value_cache,
  std::vector<bool>& is_sliding,
  int window_size,
  int seen_tokens
);

// update
std::tuple<torch::Tensor, torch::Tensor> muillm_hybrid_chunked_kvcache_module_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor key_states,
  torch::Tensor value_states,
  torch::Tensor& cache_positions,
  int layer_index
);

// deinit
void muillm_hybrid_chunked_kvcache_module_deinit_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
);

// sync
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> muillm_hybrid_chunked_kvcache_module_sync_back_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
);

#endif /* __MUILLM_HYBRID_CHUNKED_KVCACHE_H__ */