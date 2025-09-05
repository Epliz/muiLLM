#include "dynamic_kvcache.h"

#include "../kvcaches/dynamic_kvcache.hpp"

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

std::tuple<torch::Tensor, torch::Tensor> MuillmDynamicKVCache::update(
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

  auto prev_k_cache = this->key_cache[layer_index];
  auto prev_v_cache = this->value_cache[layer_index];

  if (prev_k_cache.defined()) {

    auto k_out_v_out_tuple = muillm_dynamic_kvcache_update(
      key_states,
      value_states,
      prev_k_cache,
      prev_v_cache
    );

    auto k_cache_out = std::get<0>(k_out_v_out_tuple);
    auto v_cache_out = std::get<1>(k_out_v_out_tuple);
  
    // update the cache
    this->key_cache[layer_index] = k_cache_out;
    this->value_cache[layer_index] = v_cache_out;

    return k_out_v_out_tuple;
  } else {
    // no previous cache, use the passed tensors as caches
    this->key_cache[layer_index] = key_states;
    this->value_cache[layer_index] = value_states;


    return std::make_tuple(key_states, value_states);
  }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MuillmDynamicKVCache::rope_update(
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

  // TODO: allocate caches if needed
  auto prev_k_cache = this->key_cache[layer_index];
  auto prev_v_cache = this->value_cache[layer_index];

  auto q_k_out_v_out_tuple = muillm_rope_forward_dynamic_cache(
      cos,
      sin,
      query_states,
      key_states,
      value_states,
      prev_k_cache,
      prev_v_cache
    );


  auto k_cache_out = std::get<1>(q_k_out_v_out_tuple);
  auto v_cache_out = std::get<2>(q_k_out_v_out_tuple);

  // update the cache
  this->key_cache[layer_index] = k_cache_out;
  this->value_cache[layer_index] = v_cache_out;

  return q_k_out_v_out_tuple;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MuillmDynamicKVCache::complex_rope_update(
  torch::Tensor& query_states,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  torch::Tensor& position_embeddings,
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

  // TODO: allocate caches if needed
  auto prev_k_cache = this->key_cache[layer_index];
  auto prev_v_cache = this->value_cache[layer_index];

  auto q_k_out_v_out_tuple = muillm_complex_rope_forward_dynamic_cache(
      position_embeddings,
      query_states,
      key_states,
      value_states,
      prev_k_cache,
      prev_v_cache
    );


  auto k_cache_out = std::get<1>(q_k_out_v_out_tuple);
  auto v_cache_out = std::get<2>(q_k_out_v_out_tuple);

  // update the cache
  this->key_cache[layer_index] = k_cache_out;
  this->value_cache[layer_index] = v_cache_out;

  return q_k_out_v_out_tuple;
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


// update
std::tuple<torch::Tensor, torch::Tensor> muillm_dynamic_kvcache_module_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  std::optional<torch::Tensor> cache_positions_,
  int layer_index
) {
  MuillmKVCache* cache = module_ptr.ptr;
  if (cache->type != MUILLM_DYNAMIC_KVCACHE) {
    TORCH_CHECK(false, "expected a dynamic cache");
  }

  torch::Tensor cache_positions = cache_positions_.has_value() ? cache_positions_.value() : torch::Tensor();

  MuillmDynamicKVCache* dynamic_cache = (MuillmDynamicKVCache*) cache;

  return dynamic_cache->update(
    key_states,
    value_states,
    cache_positions,
    layer_index
  );
}

// rope update
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> muillm_dynamic_kvcache_module_rope_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& query_states,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  std::tuple<torch::Tensor, torch::Tensor> position_embeddings,
  torch::Tensor& cache_positions,
  int layer_index
) {
  MuillmKVCache* cache = module_ptr.ptr;
  if (cache->type != MUILLM_DYNAMIC_KVCACHE) {
    TORCH_CHECK(false, "expected a dynamic cache");
  }

  MuillmDynamicKVCache* dynamic_cache = (MuillmDynamicKVCache*) cache;
  return dynamic_cache->rope_update(
    query_states,
    key_states,
    value_states,
    position_embeddings,
    cache_positions,
    layer_index
  );
}

// complex rope update
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> muillm_dynamic_kvcache_module_complex_rope_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor& query_states,
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  torch::Tensor& position_embeddings,
  torch::Tensor& cache_positions,
  int layer_index
) {
  MuillmKVCache* cache = module_ptr.ptr;
  if (cache->type != MUILLM_DYNAMIC_KVCACHE) {
    TORCH_CHECK(false, "expected a dynamic cache");
  }

  MuillmDynamicKVCache* dynamic_cache = (MuillmDynamicKVCache*) cache;
  return dynamic_cache->complex_rope_update(
    query_states,
    key_states,
    value_states,
    position_embeddings,
    cache_positions,
    layer_index
  );
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