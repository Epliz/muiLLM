#include "dynamic_kvcache.h"

MuillmDynamicKVCache::MuillmDynamicKVCache(
    muillm_engine_t* engine,
    std::vector<torch::Tensor>& key_cache,
    std::vector<torch::Tensor>& value_cache,
    int seen_tokens
) : MuillmKVCache(engine, MUILLM_DYNAMIC_KVCACHE, seen_tokens) {
  this->key_cache = key_cache;
  this->value_cache = value_cache;
}

MuillmDynamicKVCache::~MuillmDynamicKVCache() {
}


// init
muillm_kvcache_module_ptr_t muillm_dynamic_kvcache_module_init_trampoline(
  muillm_engine_ptr engine,
  std::vector<torch::Tensor>& key_cache,
  std::vector<torch::Tensor>& value_cache,
  int seen_tokens
) {
  muillm_kvcache_module_ptr_t ret;

  MuillmDynamicKVCache* cache = new MuillmDynamicKVCache(engine.engine_ptr, key_cache, value_cache, seen_tokens);

  ret.ptr = cache;
  return ret;
}
  
// deinit
void muillm_dynamic_kvcache_module_deinit_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}
  
// sync
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> muillm_dynamic_kvcache_module_sync_back_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
) {
  MuillmKVCache* cache = module_ptr.ptr;
  if (cache->type != MUILLM_DYNAMIC_KVCACHE) {
    TORCH_CHECK(false, "expected a dynamic cache");
  }

  MuillmDynamicKVCache* dynamic_cache =(MuillmDynamicKVCache*) cache;
  return std::make_tuple(dynamic_cache->key_cache, dynamic_cache->value_cache);
}