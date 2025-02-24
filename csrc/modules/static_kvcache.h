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
  
};

// init
muillm_kvcache_module_ptr_t muillm_static_kvcache_module_init_trampoline(
  muillm_engine_ptr engine,
  std::vector<torch::Tensor>& key_cache,
  std::vector<torch::Tensor>& value_cache,
  int seen_tokens
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