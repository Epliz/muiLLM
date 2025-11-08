#ifndef __MUILLM_GEMMA3_DECODER_MODULE_H__
#define __MUILLM_GEMMA3_DECODER_MODULE_H__


#include "../engine.h"

#include "multilinear_module.h"
#include "gemma3_attention_module.h"
#include "gateup_module_interface.h"
#include "kvcache.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>

struct MuiLLMGemma3Decoder: torch::nn::Module {
  // fields
  muillm_engine_t* engine;

  MuiLLMMultiLinear* multilinear;
  MuiLLMGemma3Attention* attention;
  MuiLLMGateUpDownMLPInterface* mlp;

  bool sliding_layer;

  // post attention layer norm
  torch::Tensor post_attention_layer_norm_weight;
  float post_attention_layer_norm_epsilon;
  float post_attention_layer_norm_weights_offset;

  // post feedforward layer norm
  torch::Tensor post_feedforward_layer_norm_weight;
  float post_feedforward_layer_norm_epsilon;
  float post_feedforward_layer_norm_weights_offset;

  // methods
  MuiLLMGemma3Decoder(
    muillm_engine_t* engine,
    MuiLLMMultiLinear* multilinear,
    MuiLLMGemma3Attention* attention,
    MuiLLMGateUpDownMLPInterface* mlp,
    bool sliding_layer,
    torch::Tensor& post_attention_layer_norm_weight,
    float post_attention_layer_norm_epsilon,
    float post_attention_layer_norm_weights_offset,
    torch::Tensor& post_feedforward_layer_norm_weight,
    float post_feedforward_layer_norm_epsilon,
    float post_feedforward_layer_norm_weights_offset
  );

  virtual ~MuiLLMGemma3Decoder();

  torch::Tensor forward(
    MuillmKVCache* cache,
    torch::Tensor& h,
    torch::Tensor& mask,
    torch::Tensor& sliding_mask,
    std::tuple<torch::Tensor, torch::Tensor>& position_embeds_global,
    std::tuple<torch::Tensor, torch::Tensor>& position_embeds_local,
    torch::Tensor& cache_positions
  );
};

// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_gemma3_decoder_module_ptr {
  MuiLLMGemma3Decoder* ptr;
} muillm_gemma3_decoder_module_ptr_t;

muillm_gemma3_decoder_module_ptr_t muillm_gemma3_decoder_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_multilinear_module_ptr_t multilinear,
  muillm_gemma3_attention_module_ptr_t attention,
  muillm_igateupdownmlp_module_ptr_t mlp,
  bool sliding_layer,
  torch::Tensor& post_attention_layer_norm_weight,
  float post_attention_layer_norm_epsilon,
  float post_attention_layer_norm_weights_offset,
  torch::Tensor& post_feedforward_layer_norm_weight,
  float post_feedforward_layer_norm_epsilon,
  float post_feedforward_layer_norm_weights_offset
);

void muillm_gemma3_decoder_module_deinit_trampoline(
  muillm_gemma3_decoder_module_ptr_t module_ptr
);

at::Tensor muillm_gemma3_decoder_module_forward(
  muillm_gemma3_decoder_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& mask,
  std::optional<torch::Tensor>& sliding_mask,
  std::tuple<torch::Tensor, torch::Tensor> position_embeds_global,
  std::tuple<torch::Tensor, torch::Tensor> position_embeds_local,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_GEMMA3_DECODER_MODULE_H__ */