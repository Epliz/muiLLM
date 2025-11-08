#ifndef __MUILLM_PARALLEL_GEMMA3_DECODER_MODULE_H__
#define __MUILLM_PARALLEL_GEMMA3_DECODER_MODULE_H__


#include "../engine.h"

#include "parallel_multilinear_module.h"
#include "parallel_gemma3_attention_module.h"
#include "parallel_gateup_module_interface.h"
#include "kvcache.h"

#include <optional>
#include <tuple>

#include <torch/torch.h>

struct MuiLLMParallelGemma3Decoder: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  muillm_comm_t* comm;

  MuiLLMParallelMultiLinear* multilinear;
  MuiLLMParallelGemma3Attention* attention;
  MuiLLMParallelGateUpDownMLPInterface* mlp;

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
  MuiLLMParallelGemma3Decoder(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    MuiLLMParallelMultiLinear* multilinear,
    MuiLLMParallelGemma3Attention* attention,
    MuiLLMParallelGateUpDownMLPInterface* mlp,
    bool sliding_layer,
    torch::Tensor& post_attention_layer_norm_weight,
    float post_attention_layer_norm_epsilon,
    float post_attention_layer_norm_weights_offset,
    torch::Tensor& post_feedforward_layer_norm_weight,
    float post_feedforward_layer_norm_epsilon,
    float post_feedforward_layer_norm_weights_offset
  );

  virtual ~MuiLLMParallelGemma3Decoder();

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
typedef struct muillm_parallel_gemma3_decoder_module_ptr {
  MuiLLMParallelGemma3Decoder* ptr;
} muillm_parallel_gemma3_decoder_module_ptr_t;

muillm_parallel_gemma3_decoder_module_ptr_t muillm_parallel_gemma3_decoder_module_init_trampoline(
  muillm_engine_ptr engine,
  muillm_comm_ptr comm,
  muillm_parallel_multilinear_module_ptr_t multilinear,
  muillm_parallel_gemma3_attention_module_ptr_t attention,
  muillm_parallel_igateupdownmlp_module_ptr_t mlp,
  bool sliding_layer,
  torch::Tensor& post_attention_layer_norm_weight,
  float post_attention_layer_norm_epsilon,
  float post_attention_layer_norm_weights_offset,
  torch::Tensor& post_feedforward_layer_norm_weight,
  float post_feedforward_layer_norm_epsilon,
  float post_feedforward_layer_norm_weights_offset
);

void muillm_parallel_gemma3_decoder_module_deinit_trampoline(
  muillm_parallel_gemma3_decoder_module_ptr_t module_ptr
);

at::Tensor muillm_parallel_gemma3_decoder_module_forward(
  muillm_parallel_gemma3_decoder_module_ptr_t module_ptr,
  muillm_kvcache_module_ptr_t cache_ptr,
  torch::Tensor& h,
  std::optional<torch::Tensor>& mask,
  std::optional<torch::Tensor>& sliding_mask,
  std::tuple<torch::Tensor, torch::Tensor> position_embeds_global,
  std::tuple<torch::Tensor, torch::Tensor> position_embeds_local,
  torch::Tensor& cache_positions
);

#endif /* __MUILLM_PARALLEL_GEMMA3_DECODER_MODULE_H__ */