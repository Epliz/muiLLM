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

std::tuple<torch::Tensor, torch::Tensor> MuillmRotaryEmbedding::compute_rotary_pos_emb(
  torch::Tensor& x,
  torch::Tensor& position_ids
) {
  return muillm_compute_rotary_embed_positions(
    x,
    position_ids, 
    this->cos_cached,
    this->sin_cached
  );
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