#ifndef __MUILLM_DYNAMIC_KVCACHE_H__
#define __MUILLM_DYNAMIC_KVCACHE_H__

#include "kvcache.h"


#include "../engine.h"

#include <torch/torch.h>
#include <vector>
#include <tuple>

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
  
};

// init
muillm_kvcache_module_ptr_t muillm_dynamic_kvcache_module_init_trampoline(
  muillm_engine_ptr engine,
  std::vector<torch::Tensor>& key_cache,
  std::vector<torch::Tensor>& value_cache,
  int seen_tokens
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