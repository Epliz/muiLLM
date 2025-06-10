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
#include "gateupmoe_kernels.cuh"


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

#include "l2norm_kernels.cuh"
#include "qkl2norm_kernels.cuh"
#include "rmsnorm_kernels.cuh"
#include "reduce_kernels.cuh"
#include "topk_kernels.cuh"
#include "rotary_kernels.h"
#include "kvcaches/static_kvcache_kernels.cuh"
#include "kvcaches/sliding_kvcache_kernels.cuh"
#include "temperature_tuning_kernels.cuh"

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

#include "modules/linear_module.h"

#include "parallel_linear_kernels.cuh"
#include "parallel_gateupmoe_kernels.cuh"

#include "modules/parallel_linear_module.h"
#include "modules/parallel_multilinear_module.h"
#include "modules/parallel_gateup_module.h"
#include "modules/parallel_gateupmoe_module.h"
#include "modules/parallel_attention_module.h"
#include "modules/parallel_llama4_attention_module.h"
#include "modules/parallel_decoder_module.h"
#include "modules/parallel_decoder_stack.h"

#include "parallel_gateup_kernels.cuh"

#include "modules/kvcache.h"
#include "modules/static_kvcache.h"
#include "modules/dynamic_kvcache.h"
#include "modules/hybrid_chunked_kvcache.h"

#include "modules/rotary_module.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("muillm_linear_forward", &muillm_linear_forward_trampoline, "muillm linear forward", py::arg("engine"), py::arg("x"), py::arg("weights"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none(), py::arg("residual") = py::none());
  m.def("muillm_parallel_linear_forward", &muillm_parallel_linear_forward_trampoline, "muillm parallel linear forward", py::arg("engine"), py::arg("comm"), py::arg("x"), py::arg("weights"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none(), py::arg("residual") = py::none(), py::arg("sharding_dim") = 1, py::arg("reduce") = false);
  m.def("muillm_int8_dequantize_forward", &muillm_int8_dequantize_forward, "muillm int8 dequantize forward");
  m.def("muillm_int8_linear_forward", &muillm_int8_linear_forward_trampoline, "muillm linear forward", py::arg("x"), py::arg("weights"), py::arg("scales_min_vals"), py::arg("group_size_shift"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none());
  m.def("muillm_gateupsilu_forward", &muillm_gateupsilu_forward_trampoline, "muillm gate up silu forward");
  m.def("muillm_gateupsilumoe_forward", &muillm_gateupsilumoe_forward_trampoline, "muillm gate up silu moe forward");
  m.def("muillm_parallel_gateupsilu_forward", &muillm_parallel_gateupsilu_forward_trampoline, "muillm parallel gate up silu forward",
    // args
    py::arg("engine"),
    py::arg("comm"),
    py::arg("norm_weights"),
    py::arg("epsilon"),
    py::arg("gate_weights"),
    py::arg("up_weights"),
    py::arg("down_weights"),
    py::arg("residual"),
    py::arg("x"),
    py::arg("reduce") = true
  );
  m.def("muillm_parallel_gateupsilumoe_forward", &muillm_parallel_gateupsilumoe_forward_trampoline, "muillm parallel gate up silu moe forward",
    // args
    py::arg("engine"),
    py::arg("comm"),
    py::arg("num_shared_experts"),
    py::arg("num_dynamic_experts"),
    py::arg("norm_weights"),
    py::arg("epsilon"),
    py::arg("gate_weights"),
    py::arg("up_weights"),
    py::arg("down_weights"),
    py::arg("residual"),
    py::arg("x"),
    py::arg("router_scores"),
    py::arg("router_indices"),
    py::arg("reduce") = true
  );

  m.def("muillm_gateupsilu_split_forward", &muillm_gateupsilu_split_forward_trampoline, "muillm gate up silu split K forward");
  m.def("muillm_parallel_gateupsilu_split_forward", &muillm_parallel_gateupsilu_split_forward_trampoline, "muillm parallel gate up silu split K forward", 
    // args
    py::arg("engine"),
    py::arg("comm"),
    py::arg("norm_weights"),
    py::arg("epsilon"),
    py::arg("gate_weights"),
    py::arg("up_weights"),
    py::arg("down_weights"),
    py::arg("residual"),
    py::arg("x"),
    py::arg("reduce") = true
  );
  m.def("muillm_parallel_gateupsilumoe_split_forward", &muillm_parallel_gateupsilumoe_split_forward_trampoline, "muillm parallel gate up silu moe split K forward",
    // args
    py::arg("engine"),
    py::arg("comm"),
    py::arg("num_shared_experts"),
    py::arg("num_dynamic_experts"),
    py::arg("norm_weights"),
    py::arg("epsilon"),
    py::arg("gate_weights"),
    py::arg("up_weights"),
    py::arg("down_weights"),
    py::arg("residual"),
    py::arg("x"),
    py::arg("router_scores"),
    py::arg("router_indices"),
    py::arg("reduce") = true
  );

  m.def("muillm_int8_gateupsilu_dequantize_forward", &muillm_int8_gateupsilu_dequantize_forward, "muillm int8 gate up dequantize");
  m.def("muillm_int8_gateupsilu_forward", &muillm_int8_gateupsilu_forward, "muillm int8 gate up silu forward");
  m.def("muillm_l2norm_forward", &muillm_l2norm_forward, "muillm l2norm forward");
  m.def("muillm_qkl2norm_forward", &muillm_qkl2norm_forward, "muillm qkl2norm forward");
  m.def("muillm_rmsnorm_forward", &muillm_rmsnorm_forward, "muillm rmsnorm forward");
  m.def("muillm_reduce_sum_forward", &muillm_reduce_sum_forward, "muillm reduce sum forward");
  m.def("muillm_topk_sigmoid_forward", &muillm_topk_sigmoid_forward, "muillm topk sigmoid forward");
  // rotary
  m.def("muillm_rope_forward_no_cache", &muillm_rope_forward_no_cache, "muillm rotary forward no cache");
  m.def("muillm_rope_forward_dynamic_cache", &muillm_rope_forward_dynamic_cache, "muillm rotary forward dynamic cache");
  m.def("muillm_rope_forward_static_cache", &muillm_rope_forward_static_cache, "muillm rotary forward static cache");
  m.def("muillm_complex_rope_forward_no_cache", &muillm_complex_rope_forward_no_cache, "muillm complex rotary forward no cache");

  // kv caches
  m.def("muillm_static_kvcache_update", &muillm_static_kvcache_update, "muillm static kvcache update");
  m.def("muillm_sliding_kvcache_update", &muillm_sliding_kvcache_update, "muillm sliding kvcache update");

  // causal transformer decoding
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

  // linear
  pybind11::class_<muillm_linear_module_ptr_t> cl_linear_module(m, "muillm_linear_module_ptr");
  cl_linear_module.def(pybind11::init<>());

  m.def("muillm_linear_module_init", &muillm_linear_module_init_trampoline, "muillm linear module init", py::arg("engine"), py::arg("weights"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none());
  m.def("muillm_linear_module_deinit", &muillm_linear_module_deinit_trampoline, "muillm linear module deinit", py::arg("module"));
  m.def("muillm_linear_module_forward", &muillm_linear_module_forward_trampoline, "muillm linear module forward", py::arg("module"), py::arg("inputs"), py::arg("residual") = py::none());


  // parallel linear
  pybind11::class_<muillm_parallel_linear_module_ptr_t> cl_parallel_linear_module(m, "muillm_parallel_linear_module_ptr");
  cl_parallel_linear_module.def(pybind11::init<>());

  m.def("muillm_parallel_linear_module_init", &muillm_parallel_linear_module_init_trampoline, "muillm parallel linear module init", py::arg("engine"), py::arg("comm"), py::arg("weights"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none(), py::arg("sharding_dim") = 1);
  m.def("muillm_parallel_linear_module_deinit", &muillm_parallel_linear_module_deinit_trampoline, "muillm parallel linear module deinit", py::arg("module"));
  m.def("muillm_parallel_linear_module_forward", &muillm_parallel_linear_module_forward_trampoline, "muillm parallel linear module forward", py::arg("module"), py::arg("inputs"), py::arg("residual") = py::none(), py::arg("reduce") = false);

  // parallel multilinear
  pybind11::class_<muillm_parallel_multilinear_module_ptr_t> cl_parallel_multilinear_module(m, "muillm_parallel_multilinear_module_ptr");
  cl_parallel_multilinear_module.def(pybind11::init<>());

  m.def("muillm_parallel_multilinear_module_init", &muillm_parallel_multilinear_module_init_trampoline, "muillm parallel multilinear module init", py::arg("engine"), py::arg("comm"), py::arg("linear"), py::arg("slices"), py::arg("sharding_dim"));
  m.def("muillm_parallel_multilinear_module_deinit", &muillm_parallel_multilinear_module_deinit_trampoline, "muillm parallel multilinear module deinit", py::arg("module"));
  m.def("muillm_parallel_multilinear_module_forward", &muillm_parallel_multilinear_module_forward_trampoline, "muillm parallel multilinear module forward", py::arg("module"), py::arg("input"), py::arg("collect_outputs"));

  // mlp interface
  pybind11::class_<muillm_parallel_igateupdownmlp_module_ptr_t> cl_parallel_igateupdownmlp_module(m, "muillm_parallel_igateupdownmlp_module_ptr");

  // parallel gateup/down mlp
  m.def("muillm_parallel_gateupdownmlp_module_init", &muillm_parallel_gateupdownmlp_module_init_trampoline, "muillm parallel gateupdown mlp module init", py::arg("engine"), py::arg("comm"), py::arg("method"), py::arg("norm_weights"), py::arg("gate_weights"), py::arg("up_weights"), py::arg("down_weights"), py::arg("variance_epsilon"));
  m.def("muillm_parallel_gateupdownmlp_module_deinit", &muillm_parallel_gateupdownmlp_module_deinit_trampoline, "muillm parallel gateupdown mlp module deinit", py::arg("module"));
  m.def("muillm_parallel_gateupdownmlp_module_forward", &muillm_parallel_gateupdownmlp_module_forward_trampoline, "muillm parallel gateupdown mlp module forward", py::arg("module"), py::arg("inputs"), py::arg("residual") = py::none(), py::arg("reduce") = true);

  // parallel gateup/down mlp moe
  m.def("muillm_parallel_gateupdownmlpmoe_module_init", &muillm_parallel_gateupdownmlpmoe_module_init_trampoline, "muillm parallel gateupdown mlp moe module init", py::arg("engine"), py::arg("comm"), py::arg("router"), py::arg("num_shared_experts"), py::arg("num_dynamic_experts"), py::arg("num_routed_experts"), py::arg("norm_weights"), py::arg("gate_weights"), py::arg("up_weights"), py::arg("down_weights"), py::arg("variance_epsilon"));
  m.def("muillm_parallel_gateupdownmlpmoe_module_deinit", &muillm_parallel_gateupdownmlpmoe_module_deinit_trampoline, "muillm parallel gateupdown mlp moe module deinit", py::arg("module"));
  m.def("muillm_parallel_gateupdownmlpmoe_module_forward", &muillm_parallel_gateupdownmlpmoe_module_forward_trampoline, "muillm parallel gateupdown mlp moe module forward", py::arg("module"), py::arg("inputs"), py::arg("residual") = py::none(), py::arg("reduce") = true);


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

  // hybrid chunked KV cache
  m.def("muillm_hybrid_chunked_kvcache_module_init", &muillm_hybrid_chunked_kvcache_module_init_trampoline, "muillm hybrid chunked kvcache module init", py::arg("engine"), py::arg("key_cache"), py::arg("value_cache"), py::arg("is_sliding"), py::arg("window_size"), py::arg("seen_tokens"));
  m.def("muillm_hybrid_chunked_kvcache_module_update", &muillm_hybrid_chunked_kvcache_module_update_trampoline, "muillm hybrid chunked kvcache module update", py::arg("module"), py::arg("key_states"), py::arg("value_states"), py::arg("cache_position"), py::arg("layer_index"));
  m.def("muillm_hybrid_chunked_kvcache_module_deinit", &muillm_hybrid_chunked_kvcache_module_deinit_trampoline, "muillm hybrid chunked kvcache module deinit", py::arg("module"));
  m.def("muillm_hybrid_chunked_kvcache_module_sync_back", &muillm_hybrid_chunked_kvcache_module_sync_back_trampoline, "muillm hybrid chunked kvcache module sync back", py::arg("module"));

  // rotary embedding
  pybind11::class_<muillm_rotary_embedding_module_ptr_t> cl_rotary_embedding_module(m, "muillm_rotary_embedding_module_ptr");
  cl_rotary_embedding_module.def(pybind11::init<>());

  m.def("muillm_rotary_embedding_module_init", &muillm_rotary_embedding_module_init_trampoline, "muillm rotary embedding module init", py::arg("engine"), py::arg("layer_idx"), py::arg("cos_cached"), py::arg("sin_cached"));
  m.def("muillm_rotary_embedding_module_deinit", &muillm_rotary_embedding_module_deinit_trampoline, "muillm rotary embedding module deinit", py::arg("module"));
  m.def("muillm_rotary_embedding_module_forward", &muillm_rotary_embedding_module_forward_trampoline, "muillm rotary embedding module forward", py::arg("module"), py::arg("cache"), py::arg("q_in"), py::arg("k_in"), py::arg("v_in"), py::arg("position_ids"), py::arg("cos_sin"), py::arg("cache_positions"));

  // temperature tuning
  m.def("muillm_apply_temperature_tuning", &muillm_apply_temperature_tuning, "muillm apply temperature tuning",
        py::arg("query_states"), py::arg("cache_position"), py::arg("attn_scale"), py::arg("floor_scale"));

  // parallel attention
  pybind11::class_<muillm_parallel_attention_module_ptr_t> cl_parallel_attention_module(m, "muillm_parallel_attention_module_ptr");
  cl_parallel_attention_module.def(pybind11::init<>());

  m.def("muillm_parallel_attention_module_init", &muillm_parallel_attention_module_init_trampoline, "muillm parallel attention module init", py::arg("engine"), py::arg("comm"), py::arg("rotary"), py::arg("o_proj"), py::arg("num_tp_heads"), py::arg("num_tp_key_value_heads"), py::arg("head_dim"));
  m.def("muillm_parallel_attention_module_deinit", &muillm_parallel_attention_module_deinit_trampoline, "muillm parallel attention module deinit", py::arg("module"));
  m.def("muillm_parallel_attention_module_forward", &muillm_parallel_attention_module_forward_trampoline, "muillm parallel attention module forward", py::arg("module"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("m") = py::none(), py::arg("residual") = py::none());
  m.def("muillm_parallel_attention_module_rope_forward", &muillm_parallel_attention_module_rope_forward_trampoline, "muillm parallel attention module rope forward", py::arg("module"), py::arg("cache"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("m"), py::arg("residual"), py::arg("position_ids"), py::arg("cos_sin"), py::arg("cache_positions"));
  
  // parallel attention
  pybind11::class_<muillm_parallel_llama4_attention_module_ptr_t> cl_parallel_llama4_attention_module(m, "muillm_parallel_llama4_attention_module_ptr");
  cl_parallel_llama4_attention_module.def(pybind11::init<>());

  m.def("muillm_parallel_llama4_attention_module_init", &muillm_parallel_llama4_attention_module_init_trampoline, "muillm parallel llama4 attention module init", py::arg("engine"), py::arg("comm"), py::arg("o_proj"), py::arg("num_tp_heads"), py::arg("num_tp_key_value_heads"), py::arg("head_dim"), py::arg("use_rope"), py::arg("use_qk_norm"), py::arg("norm_epsilon"), py::arg("use_temperature_tuning"), py::arg("attention_scale"), py::arg("floor_scale"), py::arg("layer_index"));
  m.def("muillm_parallel_llama4_attention_module_deinit", &muillm_parallel_llama4_attention_module_deinit_trampoline, "muillm parallel llama4 attention module deinit", py::arg("module"));
  m.def("muillm_parallel_llama4_attention_module_forward", &muillm_parallel_llama4_attention_module_forward_trampoline, "muillm parallel llama4 attention module forward", py::arg("module"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("m") = py::none(), py::arg("residual") = py::none());
  m.def("muillm_parallel_llama4_attention_module_rope_forward", &muillm_parallel_llama4_attention_module_rope_forward_trampoline, "muillm parallel llama4 attention module rope forward", py::arg("module"), py::arg("cache"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("m"), py::arg("residual"), py::arg("position_embeds"), py::arg("cache_positions"));

  // parallel decoder
  pybind11::class_<muillm_parallel_decoder_module_ptr_t> cl_parallel_decoder_module(m, "muillm_parallel_decoder_module_ptr");
  cl_parallel_decoder_module.def(pybind11::init<>());

  m.def("muillm_parallel_decoder_module_init", &muillm_parallel_decoder_module_init_trampoline, "muillm parallel decoder module init", py::arg("engine"), py::arg("comm"), py::arg("multilinear"), py::arg("attention"), py::arg("mlp"));
  m.def("muillm_parallel_decoder_module_deinit", &muillm_parallel_decoder_module_deinit_trampoline, "muillm parallel decoder module deinit", py::arg("module"));
  m.def("muillm_parallel_decoder_module_forward", &muillm_parallel_decoder_module_forward, "muillm parallel decoder module forward", py::arg("module"), py::arg("cache"), py::arg("h"), py::arg("m"), py::arg("position_ids"), py::arg("cos_sin"), py::arg("cache_positions"));

  // parallel decoder stack
  pybind11::class_<muillm_parallel_decoder_stack_ptr_t> cl_parallel_decoder_stack(m, "muillm_parallel_decoder_stack_ptr");
  cl_parallel_decoder_stack.def(pybind11::init<>());

  m.def("muillm_parallel_decoder_stack_init", &muillm_parallel_decoder_stack_init_trampoline, "muillm parallel decoder stack init", py::arg("engine"), py::arg("comm"), py::arg("decoders"));
  m.def("muillm_parallel_decoder_stack_deinit", &muillm_parallel_decoder_stack_deinit_trampoline, "muillm parallel decoder stack deinit", py::arg("module"));
  m.def("muillm_parallel_decoder_stack_forward", &muillm_parallel_decoder_stack_forward_trampoline, "muillm parallel decoder stack forward", py::arg("module"), py::arg("cache"), py::arg("h"), py::arg("m"), py::arg("position_ids"), py::arg("cos_sin"), py::arg("cache_positions"));
}
