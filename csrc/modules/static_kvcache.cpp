#include "static_kvcache.h"

#include "../kvcaches/static_kvcache.hpp"

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

std::tuple<torch::Tensor, torch::Tensor> MuillmStaticKVCache::update(
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  torch::Tensor& cache_positions,
  int layer_index
) {

  int prev_seen_tokens;
  int seen_tokens;
  auto num_new_tokens = key_states.size(key_states.dim() - 2);

  if (layer_index == 0) {
    // update the token count only for the first layer
    prev_seen_tokens = this->seen_tokens();
    seen_tokens = prev_seen_tokens + num_new_tokens;
    this->seen_tokens(seen_tokens);
  } else {
    // we assume we already updated the count with the first layer
    seen_tokens = this->seen_tokens();
    prev_seen_tokens = seen_tokens - num_new_tokens;
  }

  // static cache
  return muillm_static_kvcache_update(
    key_states,
    value_states,
    this->key_cache[layer_index],
    this->value_cache[layer_index],
    cache_positions,
    this->seen_tokens()
  );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MuillmStaticKVCache::rope_update(
  torch::Tensor& query_states,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  std::tuple<torch::Tensor, torch::Tensor>& position_embeddings,
  torch::Tensor& cache_positions,
  int layer_index
) {

  auto undef_tensor = torch::Tensor();

  int prev_seen_tokens;
  int seen_tokens;
  auto num_new_tokens = key_states.size(key_states.dim() - 2);

  if (layer_index == 0) {
    // update the token count only for the first layer
    prev_seen_tokens = this->seen_tokens();
    seen_tokens = prev_seen_tokens + num_new_tokens;
    this->seen_tokens(seen_tokens);
  } else {
    // we assume we already updated the count with the first layer
    seen_tokens = this->seen_tokens();
    prev_seen_tokens = seen_tokens - num_new_tokens;
  }

  torch::Tensor cos = std::get<0>(position_embeddings);
  torch::Tensor sin = std::get<1>(position_embeddings);

  // static cache
  return muillm_rope_forward_static_cache(
    cos,
    sin,
    query_states,
    key_states,
    value_states,
    this->key_cache[layer_index],
    this->value_cache[layer_index],
    cache_positions,
    this->seen_tokens()
  );
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MuillmStaticKVCache::complex_rope_update(
  torch::Tensor& query_states,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  torch::Tensor& position_embeddings,
  torch::Tensor& cache_positions,
  int layer_index
) {

  int prev_seen_tokens;
  int seen_tokens;
  auto num_new_tokens = key_states.size(key_states.dim() - 2);

  if (layer_index == 0) {
    // update the token count only for the first layer
    prev_seen_tokens = this->seen_tokens();
    seen_tokens = prev_seen_tokens + num_new_tokens;
    this->seen_tokens(seen_tokens);
  } else {
    // we assume we already updated the count with the first layer
    seen_tokens = this->seen_tokens();
    prev_seen_tokens = seen_tokens - num_new_tokens;
  }

  // static cache
  return muillm_complex_rope_forward_static_cache(
    position_embeddings,
    query_states,
    key_states,
    value_states,
    this->key_cache[layer_index],
    this->value_cache[layer_index],
    cache_positions,
    this->seen_tokens()
  );
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

// update
std::tuple<torch::Tensor, torch::Tensor> muillm_static_kvcache_module_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  torch::Tensor& cache_positions,
  int layer_index
) {
  MuillmKVCache* cache = module_ptr.ptr;
  if (cache->type != MUILLM_STATIC_KVCACHE) {
    TORCH_CHECK(false, "expected a static cache");
  }

  MuillmStaticKVCache* static_cache = (MuillmStaticKVCache*) cache;
  return static_cache->update(
    key_states,
    value_states,
    cache_positions,
    layer_index
  );
}

// rope update
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> muillm_static_kvcache_module_rope_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& query_states,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  std::tuple<torch::Tensor, torch::Tensor> position_embeddings,
  torch::Tensor& cache_positions,
  int layer_index
) {
  MuillmKVCache* cache = module_ptr.ptr;
  if (cache->type != MUILLM_STATIC_KVCACHE) {
    TORCH_CHECK(false, "expected a static cache");
  }

  MuillmStaticKVCache* static_cache = (MuillmStaticKVCache*) cache;
  return static_cache->rope_update(
    query_states,
    key_states,
    value_states,
    position_embeddings,
    cache_positions,
    layer_index
  );
}

// complex rope update
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> muillm_static_kvcache_module_complex_rope_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& query_states,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  torch::Tensor& position_embeddings,
  torch::Tensor& cache_positions,
  int layer_index
) {
  MuillmKVCache* cache = module_ptr.ptr;
  if (cache->type != MUILLM_STATIC_KVCACHE) {
    TORCH_CHECK(false, "expected a static cache");
  }

  MuillmStaticKVCache* static_cache = (MuillmStaticKVCache*) cache;
  return static_cache->complex_rope_update(
    query_states,
    key_states,
    value_states,
    position_embeddings,
    cache_positions,
    layer_index
  );
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