#include "rotary_module.h"
#include "static_kvcache.h"
#include "dynamic_kvcache.h"

#include "../rope/rotary.h"

MuillmRotaryEmbedding::MuillmRotaryEmbedding(
  muillm_engine_t* engine,
  int layer_idx,
  torch::Tensor& cos_cached,
  torch::Tensor& sin_cached
) {
    this->engine = engine;

    this->cos_cached = cos_cached;
    this->sin_cached = sin_cached;

    this->layer_idx = layer_idx;
}

MuillmRotaryEmbedding::~MuillmRotaryEmbedding() {
}

// out: q, k, v
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MuillmRotaryEmbedding::forward(
  MuillmKVCache* cache,
  torch::Tensor& q_in,
  torch::Tensor& k_in,
  torch::Tensor& v_in,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
) {

  torch::Tensor cos;
  torch::Tensor sin;
  if (cos_sin.has_value()) {
    auto cos_sin_tuple = cos_sin.value();
    cos = std::get<0>(cos_sin_tuple);
    sin = std::get<1>(cos_sin_tuple);
  } else {
    // cos_sin were not provided, compute them
    auto cos_sin_tuple = this->compute_rotary_pos_emb(k_in, position_ids);
    cos = std::get<0>(cos_sin_tuple);
    sin = std::get<1>(cos_sin_tuple);
  }

  if (cache == nullptr) {
    // no cache
    auto qk_tuple = muillm_rope_forward_no_cache(
      position_ids,
      cos,
      sin,
      q_in,
      k_in
    );

    auto q = std::get<0>(qk_tuple);
    auto k = std::get<1>(qk_tuple);

    return std::make_tuple(q, k, v_in);
  }
  

  // cached cases

  auto layer_idx = this->layer_idx;
  if (layer_idx == 0) {
    // update the token count only for the first layer
    int prev_seen_tokens = cache->seen_tokens();
    auto T = k_in.size(k_in.dim() - 2);
    cache->seen_tokens(prev_seen_tokens + T);
  }

  if (cache->type == MUILLM_STATIC_KVCACHE) {
    MuillmStaticKVCache* static_kvcache = reinterpret_cast<MuillmStaticKVCache*>(cache);

    auto k_cache = static_kvcache->key_cache[layer_idx];
    auto v_cache = static_kvcache->value_cache[layer_idx];

    auto qkv_tuple = muillm_rope_forward_static_cache(
      position_ids,
      cos,
      sin,
      q_in,
      k_in,
      v_in,
      k_cache,
      v_cache,
      cache_positions,
      cache->seen_tokens()
    );

    return qkv_tuple;
  } else if (cache->type == MUILLM_DYNAMIC_KVCACHE) {
    MuillmDynamicKVCache* dynamic_kvcache = reinterpret_cast<MuillmDynamicKVCache*>(cache);

    auto prev_k_cache = dynamic_kvcache->key_cache[layer_idx];
    auto prev_v_cache = dynamic_kvcache->value_cache[layer_idx];

    auto q_k_out_v_out_tuple = muillm_rope_forward_dynamic_cache(
      position_ids,
      cos,
      sin,
      q_in,
      k_in,
      v_in,
      prev_k_cache,
      prev_v_cache
    );

    auto k_cache_out = std::get<1>(q_k_out_v_out_tuple);
    auto v_cache_out = std::get<2>(q_k_out_v_out_tuple);

    // update the cache
    dynamic_kvcache->key_cache[layer_idx] = k_cache_out;
    dynamic_kvcache->value_cache[layer_idx] = v_cache_out;

    // return
    return q_k_out_v_out_tuple;
  } else {
    // unknown cache type
    TORCH_CHECK(false, "unsupported cache type");
  }
}

std::tuple<torch::Tensor, torch::Tensor> MuillmRotaryEmbedding::compute_rotary_pos_emb(
  torch::Tensor& x,
  torch::Tensor& position_ids
) {
  TORCH_CHECK(false, "compute_rotary_pos_emb is not implemented");
}

muillm_rotary_embedding_module_ptr_t muillm_rotary_embedding_module_init_trampoline(
  muillm_engine_ptr engine,
  int layer_idx,
  torch::Tensor& cos_cached,
  torch::Tensor& sin_cached
) {
  MuillmRotaryEmbedding* module = new MuillmRotaryEmbedding(
    engine.engine_ptr,
    layer_idx,
    cos_cached,
    sin_cached
  );
  muillm_rotary_embedding_module_ptr_t module_ptr = {module};
  return module_ptr;
}

void muillm_rotary_embedding_module_deinit_trampoline(
  muillm_rotary_embedding_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> muillm_rotary_embedding_module_forward_trampoline(
  muillm_rotary_embedding_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& q_in,
  torch::Tensor& k_in,
  torch::Tensor& v_in,
  torch::Tensor& position_ids,
  std::optional<std::tuple<torch::Tensor, torch::Tensor>>& cos_sin,
  torch::Tensor& cache_positions
) {
  MuillmRotaryEmbedding* rotary_emb = module_ptr.ptr;
  MuillmKVCache* cache = cache_ptr.ptr;

  return rotary_emb->forward(cache, q_in, k_in, v_in, position_ids, cos_sin, cache_positions);
}