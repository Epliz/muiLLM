#ifndef __MUILLM_EMBEDDING_MODULE_H__
#define __MUILLM_EMBEDDING_MODULE_H__

#include "../engine.h"

#include <optional>

#include <torch/torch.h>

struct MuiLLMEmbedding: torch::nn::Module {
  // fields
  muillm_engine_t* engine;
  
  torch::Tensor weights{nullptr};

  bool dispatchable;

  // methods
  MuiLLMEmbedding(
    muillm_engine_t* engine,
    torch::Tensor& weights
  );

  virtual ~MuiLLMEmbedding();

  torch::Tensor forward(
    torch::Tensor& inputs
  );
};

// needed because Pybind11 can't seem to be able to deal with opaque pointers
typedef struct muillm_embedding_module_ptr {
  MuiLLMEmbedding* ptr;
} muillm_embedding_module_ptr_t;

// init
muillm_embedding_module_ptr_t muillm_embedding_module_init_trampoline(
  muillm_engine_ptr engine,
  torch::Tensor weights
);

// deinit
void muillm_embedding_module_deinit_trampoline(
  muillm_embedding_module_ptr_t module_ptr
);

// forward
at::Tensor muillm_embedding_module_forward_trampoline(
  muillm_embedding_module_ptr_t module_ptr,
  torch::Tensor& inputs
);

#endif /* __MUILLM_EMBEDDING_MODULE_H__ */