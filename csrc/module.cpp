#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include <tuple>

#include "engine.h"

muillm_engine_ptr muillm_engine_init_trampoline(
) {

  muillm_engine_t* ptr = nullptr;
  muillm_error_t error = muillm_engine_init(&ptr);

  TORCH_CHECK(error == MUILLM_SUCCESS, "an error happened when initializing mui engine");

  muillm_engine_ptr ret;
  ret.engine_ptr = ptr;
  return ret;
}

#include "linear_kernels.cuh"

#include "int8_linear_kernels.cuh"


at::Tensor muillm_int8_dequantize_forward(
    torch::Tensor weights,
    torch::Tensor scales_min_vals,
    int group_size_shift);

at::Tensor muillm_int8_linear_forward_trampoline(
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor scales_min_vals,
    int group_size_shift,
    std::optional<torch::Tensor> norm_weights_,
    float epsilon,
    std::optional<torch::Tensor> mul_bias_,
    std::optional<torch::Tensor> add_bias_) {
    torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : torch::Tensor();
    torch::Tensor mul_bias = mul_bias_.has_value() ? mul_bias_.value() : torch::Tensor();
    torch::Tensor add_bias = add_bias_.has_value() ? add_bias_.value() : torch::Tensor();
    return muillm_int8_linear_activ_forward(
        norm_weights,
        epsilon,
        weights,
        scales_min_vals,
        group_size_shift,
        mui_activation::Identity,
        mul_bias,
        add_bias,
        x
    );
}

#include "gateup_kernels.cuh"

at::Tensor muillm_gateupsilu_forward_trampoline(
    muillm_engine_ptr engine,
    std::optional<torch::Tensor> norm_weights_,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor down_weights,
    std::optional<torch::Tensor> residual_,
    torch::Tensor x) {
    torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : torch::Tensor();
    torch::Tensor residual = residual_.has_value() ? residual_.value() : torch::Tensor();
    return muillm_gateupsilu_forward(
        engine.engine_ptr,
        norm_weights,
        epsilon,
        gate_weights,
        up_weights,
        down_weights,
        residual,
        x
    );
}

at::Tensor muillm_gateupsilu_split_forward_trampoline(
    muillm_engine_ptr engine,
    std::optional<torch::Tensor> norm_weights_,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor down_weights,
    std::optional<torch::Tensor> residual_,
    torch::Tensor x) {
    torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : torch::Tensor();
    torch::Tensor residual = residual_.has_value() ? residual_.value() : torch::Tensor();
    return muillm_gateupsilu_split_forward(
        engine.engine_ptr,
        norm_weights,
        epsilon,
        gate_weights,
        up_weights,
        down_weights,
        residual,
        x
    );
}

std::tuple<at::Tensor, at::Tensor> muillm_int8_gateupsilu_dequantize_forward(
    torch::Tensor gate_up_weights,
    torch::Tensor gate_up_scales_min_vals,
    int group_size_shift);

at::Tensor muillm_int8_gateupsilu_forward(
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor gate_up_weights,
    torch::Tensor gate_up_scales_min_vals,
    int group_size_shift,
    torch::Tensor x);

#include "rmsnorm_kernels.cuh"

#include "rotary_kernels.h"

#include "causal_transformer_decoding.cuh"

#include "sync.h"

// needed because Pybind11 can't seem to be able to deal with opaque pointers
struct muillm_synchronizer_ptr {
  muillm_synchronizer_t* sync_ptr;
};

muillm_synchronizer_ptr muillm_sync_init_trampoline(
) {
  muillm_synchronizer_t* ptr = muillm_sync_init();
  muillm_synchronizer_ptr ret;
  ret.sync_ptr = ptr;
  return ret;
}

bool muillm_item_bool(
    muillm_synchronizer_t* sync,
    torch::Tensor& tensor
);

half muillm_item_f16(
    muillm_synchronizer_t* sync,
    torch::Tensor& tensor
);

float muillm_item_f32(
    muillm_synchronizer_t* sync,
    torch::Tensor& tensor
);

at::Tensor muillm_to_cpu(
    muillm_synchronizer_t* sync,
    torch::Tensor& tensor
);

bool muillm_item_bool_trampoline(
    muillm_synchronizer_ptr sync,
    torch::Tensor& tensor
) {
  return muillm_item_bool(sync.sync_ptr, tensor);
}

float muillm_item_f16_trampoline(
    muillm_synchronizer_ptr sync,
    torch::Tensor& tensor
) {
  return __half2float(muillm_item_f16(sync.sync_ptr, tensor));
}

float muillm_item_f32_trampoline(
    muillm_synchronizer_ptr sync,
    torch::Tensor& tensor
) {
  return muillm_item_f32(sync.sync_ptr, tensor);
}

at::Tensor muillm_to_cpu_trampoline(
    muillm_synchronizer_ptr sync,
    torch::Tensor& tensor
) {
  return muillm_to_cpu(sync.sync_ptr, tensor);
}

#include "comm_torch.h"

#include "parallel_linear_kernels.cuh"

#include "modules/parallel_linear_module.h"

#include "modules/parallel_attention_module.h"

// parallel Gate/Up Silu (FFN)
at::Tensor muillm_parallel_gateupsilu_forward(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x);

at::Tensor muillm_parallel_gateupsilu_split_forward(
    muillm_engine_t* engine,
    muillm_comm_t* comm,
    torch::Tensor& norm_weights,
    float epsilon,
    torch::Tensor& gate_weights,
    torch::Tensor& up_weights,
    torch::Tensor& down_weights,
    torch::Tensor& residual,
    torch::Tensor& x);

at::Tensor muillm_parallel_gateupsilu_forward_trampoline(
    muillm_engine_ptr engine,
    muillm_comm_ptr comm,
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor down_weights,
    torch::Tensor residual,
    torch::Tensor x) {
  return muillm_parallel_gateupsilu_forward(
    engine.engine_ptr,
    comm.comm_ptr,
    norm_weights,
    epsilon,
    gate_weights,
    up_weights,
    down_weights,
    residual,
    x
  );
}

at::Tensor muillm_parallel_gateupsilu_split_forward_trampoline(
    muillm_engine_ptr engine,
    muillm_comm_ptr comm,
    torch::Tensor norm_weights,
    float epsilon,
    torch::Tensor gate_weights,
    torch::Tensor up_weights,
    torch::Tensor down_weights,
    torch::Tensor residual,
    torch::Tensor x) {
  return muillm_parallel_gateupsilu_split_forward(
    engine.engine_ptr,
    comm.comm_ptr,
    norm_weights,
    epsilon,
    gate_weights,
    up_weights,
    down_weights,
    residual,
    x
  );
}

#include "modules/kvcache.h"
#include "modules/static_kvcache.h"
#include "modules/dynamic_kvcache.h"

#include "modules/rotary_module.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("muillm_linear_forward", &muillm_linear_forward_trampoline, "muillm linear forward", py::arg("engine"), py::arg("x"), py::arg("weights"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none(), py::arg("residual") = py::none());
  m.def("muillm_parallel_linear_forward", &muillm_parallel_linear_forward_trampoline, "muillm parallel linear forward", py::arg("engine"), py::arg("comm"), py::arg("x"), py::arg("weights"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none(), py::arg("residual") = py::none(), py::arg("sharding_dim") = 1, py::arg("reduce") = false);
  m.def("muillm_int8_dequantize_forward", &muillm_int8_dequantize_forward, "muillm int8 dequantize forward");
  m.def("muillm_int8_linear_forward", &muillm_int8_linear_forward_trampoline, "muillm linear forward", py::arg("x"), py::arg("weights"), py::arg("scales_min_vals"), py::arg("group_size_shift"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none());
  m.def("muillm_gateupsilu_forward", &muillm_gateupsilu_forward_trampoline, "muillm gate up silu forward");
  m.def("muillm_parallel_gateupsilu_forward", &muillm_parallel_gateupsilu_forward_trampoline, "muillm parallel gate up silu forward");
  m.def("muillm_gateupsilu_split_forward", &muillm_gateupsilu_split_forward_trampoline, "muillm gate up silu split K forward");
  m.def("muillm_parallel_gateupsilu_split_forward", &muillm_parallel_gateupsilu_split_forward_trampoline, "muillm parallel gate up silu split K forward");
  m.def("muillm_int8_gateupsilu_dequantize_forward", &muillm_int8_gateupsilu_dequantize_forward, "muillm int8 gate up dequantize");
  m.def("muillm_int8_gateupsilu_forward", &muillm_int8_gateupsilu_forward, "muillm int8 gate up silu forward");
  m.def("muillm_rmsnorm_forward", &muillm_rmsnorm_forward, "muillm rmsnorm forward");
  // rotary
  m.def("muillm_rope_forward_no_cache", &muillm_rope_forward_no_cache, "muillm rotary forward no cache");
  m.def("muillm_rope_forward_dynamic_cache", &muillm_rope_forward_dynamic_cache, "muillm rotary forward dynamic cache");
  m.def("muillm_rope_forward_static_cache", &muillm_rope_forward_static_cache, "muillm rotary forward static cache");
  // causal transformer decoding
  m.def("muillm_causal_transformer_compute_softmax_scores_no_mask", &muillm_causal_transformer_compute_softmax_scores_no_mask, "muillm causal transformer compute softmax scores no mask");
  m.def("muillm_causal_transformer_apply_softmax_scores", &muillm_causal_transformer_apply_softmax_scores, "muillm causal transformer apply softmax scores");
  m.def("muillm_causal_transformer_decoding_no_mask", &muillm_causal_transformer_decoding_no_mask, "muillm causal transformer decoding no mask");
  m.def("muillm_causal_transformer_decoding_masked", &muillm_causal_transformer_decoding_masked, "muillm causal transformer decoding masked");

  // communication
  pybind11::class_<muillm_engine_ptr> cl_eng(m, "muillm_engine_ptr");
  cl_eng.def(pybind11::init<>());

  m.def("muillm_engine_init", &muillm_engine_init_trampoline, "muillm engine init");

  // synchronization
  pybind11::class_<muillm_synchronizer_ptr> cl_sync(m, "muillm_synchronizer_ptr");
  cl_sync.def(pybind11::init<>());

  m.def("muillm_sync_init", &muillm_sync_init_trampoline, "muillm sync init");
  m.def("muillm_item_bool", &muillm_item_bool_trampoline, "muillm item bool");
  m.def("muillm_item_f16", &muillm_item_f16_trampoline, "muillm item f16");
  m.def("muillm_item_f32", &muillm_item_f32_trampoline, "muillm item f32");
  m.def("muillm_to_cpu", &muillm_to_cpu_trampoline, "muillm to cpu");


  // communication
  pybind11::class_<muillm_comm_ptr> cl_comm(m, "muillm_comm_ptr");
  cl_comm.def(pybind11::init<>());

  m.def("muillm_comm_init", &muillm_comm_init_trampoline, "muillm comm init");
  m.def("muillm_all_reduce_sum", &muillm_all_reduce_sum_trampoline, "muillm all_reduce sum");
  m.def("muillm_broadcast", &muillm_broadcast_trampoline, "muillm broadcast");

  // modules

  // parallel linear
  pybind11::class_<muillm_parallel_linear_module_ptr_t> cl_parallel_linear_module(m, "muillm_parallel_linear_module_ptr");
  cl_parallel_linear_module.def(pybind11::init<>());

  m.def("muillm_parallel_linear_module_init", &muillm_parallel_linear_module_init_trampoline, "muillm parallel linear module init", py::arg("engine"), py::arg("comm"), py::arg("weights"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none(), py::arg("sharding_dim") = 1);
  m.def("muillm_parallel_linear_module_deinit", &muillm_parallel_linear_module_deinit_trampoline, "muillm parallel linear module deinit", py::arg("module"));
  m.def("muillm_parallel_linear_module_forward", &muillm_parallel_linear_module_forward_trampoline, "muillm parallel linear module forward", py::arg("module"), py::arg("inputs"), py::arg("residual") = py::none(), py::arg("reduce") = false);

  // KV cache
  pybind11::class_<muillm_kvcache_module_ptr_t> cl_kvcache_module(m, "muillm_kvcache_module_ptr");
  cl_kvcache_module.def(pybind11::init<>());

  m.def("muillm_kvcache_module_get_seen_tokens", &muillm_kvcache_module_get_seen_tokens_trampoline, "muillm kvcache module get seen tokens", py::arg("module"));
  
  // static KV cache
  m.def("muillm_static_kvcache_module_init", &muillm_static_kvcache_module_init_trampoline, "muillm static kvcache module init", py::arg("engine"), py::arg("key_cache"), py::arg("value_cache"), py::arg("seen_tokens"));
  m.def("muillm_static_kvcache_module_deinit", &muillm_static_kvcache_module_deinit_trampoline, "muillm static kvcache module deinit", py::arg("module"));
  m.def("muillm_static_kvcache_module_sync_back", &muillm_static_kvcache_module_sync_back_trampoline, "muillm static kvcache module sync back", py::arg("module"));

  // dynamic KV cache
  m.def("muillm_dynamic_kvcache_module_init", &muillm_dynamic_kvcache_module_init_trampoline, "muillm dynamic kvcache module init", py::arg("engine"), py::arg("key_cache"), py::arg("value_cache"), py::arg("seen_tokens"));
  m.def("muillm_dynamic_kvcache_module_deinit", &muillm_dynamic_kvcache_module_deinit_trampoline, "muillm dynamic kvcache module deinit", py::arg("module"));
  m.def("muillm_dynamic_kvcache_module_sync_back", &muillm_dynamic_kvcache_module_sync_back_trampoline, "muillm dynamic kvcache module sync back", py::arg("module"));

  // rotary embedding
  pybind11::class_<muillm_rotary_embedding_module_ptr_t> cl_rotary_embedding_module(m, "muillm_rotary_embedding_module_ptr");
  cl_rotary_embedding_module.def(pybind11::init<>());

  m.def("muillm_rotary_embedding_module_init", &muillm_rotary_embedding_module_init_trampoline, "muillm rotary embedding module init", py::arg("engine"), py::arg("layer_idx"), py::arg("cos_cached"), py::arg("sin_cached"));
  m.def("muillm_rotary_embedding_module_deinit", &muillm_rotary_embedding_module_deinit_trampoline, "muillm rotary embedding module deinit", py::arg("module"));
  m.def("muillm_rotary_embedding_module_forward", &muillm_rotary_embedding_module_forward_trampoline, "muillm rotary embedding module forward", py::arg("module"), py::arg("cache"), py::arg("q_in"), py::arg("k_in"), py::arg("v_in"), py::arg("position_ids"), py::arg("cos_sin"), py::arg("cache_positions"));

  // parallel attention
  pybind11::class_<muillm_parallel_attention_module_ptr_t> cl_parallel_attention_module(m, "muillm_parallel_attention_module_ptr");
  cl_parallel_attention_module.def(pybind11::init<>());

  m.def("muillm_parallel_attention_module_init", &muillm_parallel_attention_module_init_trampoline, "muillm parallel attention module init", py::arg("engine"), py::arg("comm"), py::arg("rotary"), py::arg("o_proj"), py::arg("num_tp_heads"), py::arg("num_tp_key_value_heads"), py::arg("head_dim"));
  m.def("muillm_parallel_attention_module_deinit", &muillm_parallel_attention_module_deinit_trampoline, "muillm parallel attention module deinit", py::arg("module"));
  m.def("muillm_parallel_attention_module_forward", &muillm_parallel_attention_module_forward_trampoline, "muillm parallel attention module forward", py::arg("module"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("m") = py::none(), py::arg("residual") = py::none());
  m.def("muillm_parallel_attention_module_rope_forward", &muillm_parallel_attention_module_rope_forward_trampoline, "muillm parallel attention module rope forward", py::arg("module"), py::arg("cache"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("m"), py::arg("residual"), py::arg("position_ids"), py::arg("cos_sin"), py::arg("cache_positions"));
}
