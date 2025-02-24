#include "static_kvcache.h"

MuillmStaticKVCache::MuillmStaticKVCache(
    muillm_engine_t* engine,
    std::vector<torch::Tensor>& key_cache,
    std::vector<torch::Tensor>& value_cache,
    int seen_tokens
) : MuillmKVCache(engine, MUILLM_STATIC_KVCACHE, seen_tokens) {
  this->key_cache = key_cache;
  this->value_cache = value_cache;
}

MuillmStaticKVCache::~MuillmStaticKVCache() {
}

// init
muillm_kvcache_module_ptr_t muillm_static_kvcache_module_init_trampoline(
  muillm_engine_ptr engine,
  std::vector<torch::Tensor>& key_cache,
  std::vector<torch::Tensor>& value_cache,
  int seen_tokens
) {
  muillm_kvcache_module_ptr_t ret;

  MuillmStaticKVCache* cache = new MuillmStaticKVCache(engine.engine_ptr, key_cache, value_cache, seen_tokens);

  ret.ptr = cache;
  return ret;
}
  
// deinit
void muillm_static_kvcache_module_deinit_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}
  
// sync
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> muillm_static_kvcache_module_sync_back_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
) {
  MuillmKVCache* cache = module_ptr.ptr;
  if (cache->type != MUILLM_STATIC_KVCACHE) {
    TORCH_CHECK(false, "expected a static cache");
  }

  MuillmStaticKVCache* static_cache = (MuillmStaticKVCache*) cache;
  return std::make_tuple(static_cache->key_cache, static_cache->value_cache);
}