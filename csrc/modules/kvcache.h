#ifndef __MUILLM_KVCACHE_H__
#define __MUILLM_KVCACHE_H__

#include "../engine.h"

typedef enum muillm_kvcache_type {
  MUILLM_NO_KVCACHE = 0,
  MUILLM_STATIC_KVCACHE,
  MUILLM_DYNAMIC_KVCACHE,
  MUILLM_HYBRID_CHUNKED_KVCACHE
} muillm_kvcache_type_t;

struct MuillmKVCache {
  muillm_engine_t* engine;
  muillm_kvcache_type_t type;

  int _seen_tokens;
  
  MuillmKVCache(
    muillm_engine_t* engine,
    muillm_kvcache_type_t type,
    int seen_tokens
  );

  virtual ~MuillmKVCache();

  int get_set_seen_tokens(int seen_tokens) {
    if (seen_tokens > this->_seen_tokens) {
      // update the seen tokens only if the new value is greater
      this->_seen_tokens = seen_tokens;
    }
    return this->_seen_tokens;
  }

  int seen_tokens() {
    return this->_seen_tokens;
  }

  void seen_tokens(int seen_tokens) {
    this->_seen_tokens = seen_tokens;
  }

};

// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_kvcache_module_ptr {
  MuillmKVCache* ptr;
} muillm_kvcache_module_ptr_t;

// sync
int muillm_kvcache_module_get_set_seen_tokens_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  int seen_tokens
);

#endif /* __MUILLM_KVCACHE_H__ */