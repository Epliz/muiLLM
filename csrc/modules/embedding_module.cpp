#include "embedding_module.h"

#include "../embedding/embedding.cuh"

//
// actual module
//

MuiLLMEmbedding::MuiLLMEmbedding(
  muillm_engine_t* engine,
  torch::Tensor& weights
) {
  this->engine = engine;

  // we don't register as parameter in case it duplicates the memory
  this->weights = weights;

  auto wdtype = weights.dtype();
  bool dispatchable_type = (wdtype == torch::kFloat16) || (wdtype == torch::kBFloat16);
  bool dispatchable_device = weights.device().is_cuda();
  this->dispatchable = dispatchable_type && dispatchable_device;
}

MuiLLMEmbedding::~MuiLLMEmbedding() {
  // nothing to do
}

torch::Tensor MuiLLMEmbedding::forward(
    torch::Tensor& inputs
) {
  if (this->dispatchable) {
    return muillm_embedding_forward(
      this->engine,
      this->weights,
      inputs
    );
  } else {
    // embedding
    auto output = torch::nn::functional::embedding(inputs, this->weights);
    return output;
  }
}

//
// Python trampolines
//

// init
muillm_embedding_module_ptr_t muillm_embedding_module_init_trampoline(
  muillm_engine_ptr engine,
  torch::Tensor weights
) {

  MuiLLMEmbedding* m = new MuiLLMEmbedding(
    engine.engine_ptr,
    weights
  );

  muillm_embedding_module_ptr_t ret;
  ret.ptr = m;
  return ret;
}

// deinit
void muillm_embedding_module_deinit_trampoline(
  muillm_embedding_module_ptr_t module_ptr
) {
  delete module_ptr.ptr;
}

// forward
at::Tensor muillm_embedding_module_forward_trampoline(
  muillm_embedding_module_ptr_t module_ptr,
  torch::Tensor& inputs
) {

  MuiLLMEmbedding* m = module_ptr.ptr;

  return m->forward(inputs);
}
