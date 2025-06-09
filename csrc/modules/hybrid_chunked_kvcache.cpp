#include "hybrid_chunked_kvcache.h"

#include "../kvcaches/sliding_kvcache_kernels.cuh"
#include "../kvcaches/static_kvcache_kernels.cuh"

MuillmHybridChunkedKVCache::MuillmHybridChunkedKVCache(
    muillm_engine_t* engine,
    std::vector<torch::Tensor>& key_cache,
    std::vector<torch::Tensor>& value_cache,
    std::vector<bool>& is_sliding,
    int window_size,
    int seen_tokens
) : MuillmKVCache(engine, MUILLM_HYBRID_CHUNKED_KVCACHE, seen_tokens) {
  this->key_cache = key_cache;
  this->value_cache = value_cache;
  this->is_sliding = is_sliding;
  this->window_size = window_size;
}

MuillmHybridChunkedKVCache::~MuillmHybridChunkedKVCache() {
}

std::tuple<torch::Tensor, torch::Tensor> MuillmHybridChunkedKVCache::update(
  torch::Tensor key_states,
  torch::Tensor value_states,
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

  if (this->is_sliding[layer_index]) {
    // sliding cache
    bool is_prefill = prev_seen_tokens == 0;
    bool is_full = seen_tokens > this->window_size;

    auto [k_out, v_out] = muillm_sliding_kvcache_update(
      key_states,
      value_states,
      this->key_cache[layer_index],
      this->value_cache[layer_index],
      cache_positions,
      seen_tokens
    );
    
    if (!is_prefill && is_full) {
      // the cache was re-allocated during the update
      this->key_cache[layer_index] = k_out;
      this->value_cache[layer_index] = v_out;
    }

    return std::make_tuple(k_out, v_out);
  } else {
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
}

// init
muillm_kvcache_module_ptr_t muillm_hybrid_chunked_kvcache_module_init_trampoline(
  muillm_engine_ptr engine,
  std::vector<torch::Tensor>& key_cache,
  std::vector<torch::Tensor>& value_cache,
  std::vector<bool>& is_sliding,
  int window_size,
  int seen_tokens
) {
  muillm_kvcache_module_ptr_t ret;

  MuillmHybridChunkedKVCache* cache = new MuillmHybridChunkedKVCache(
    engine.engine_ptr,
    key_cache,
    value_cache,
    is_sliding,
    window_size,
    seen_tokens
  );

  ret.ptr = cache;
  return ret;
}

// update
std::tuple<torch::Tensor, torch::Tensor> muillm_hybrid_chunked_kvcache_module_update_trampoline(
  muillm_kvcache_module_ptr_t module_ptr,
  torch::Tensor key_states,
  torch::Tensor value_states,
  torch::Tensor& cache_positions,
  int layer_index
) {
  MuillmKVCache* cache = module_ptr.ptr;
  if (cache->type != MUILLM_HYBRID_CHUNKED_KVCACHE) {
    TORCH_CHECK(false, "expected a hybrid chunked cache");
  }

  MuillmHybridChunkedKVCache* hybrid_cache = (MuillmHybridChunkedKVCache*) cache;
  return hybrid_cache->update(
    key_states,
    value_states,
    cache_positions,
    layer_index
  );
}

// deinit
void muillm_hybrid_chunked_kvcache_module_deinit_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}
  
// sync
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> muillm_hybrid_chunked_kvcache_module_sync_back_trampoline(
  muillm_kvcache_module_ptr_t module_ptr
) {
  MuillmKVCache* cache = module_ptr.ptr;
  if (cache->type != MUILLM_HYBRID_CHUNKED_KVCACHE) {
    TORCH_CHECK(false, "expected a hybrid chunked cache");
  }

  MuillmHybridChunkedKVCache* hybrid_cache = (MuillmHybridChunkedKVCache*) cache;
  return std::make_tuple(hybrid_cache->key_cache, hybrid_cache->value_cache);
}