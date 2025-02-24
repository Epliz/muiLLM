#include "kvcache.h"

MuillmKVCache::MuillmKVCache(
    muillm_engine_t* engine,
    muillm_kvcache_type_t type,
    int _seen_tokens
) {
  this->engine = engine;
  this->type = type;
  this->_seen_tokens = _seen_tokens;
}

MuillmKVCache::~MuillmKVCache() {
}

int muillm_kvcache_module_get_seen_tokens_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
) {
  return module_ptr.ptr->seen_tokens();
}