#include <torch/extension.h>

#include <vector>
#include <tuple>

#include "linear_kernels.cuh"

at::Tensor muillm_linear_forward_trampoline(
    torch::Tensor x,
    torch::Tensor weights,
    std::optional<torch::Tensor> norm_weights_,
    float epsilon,
    std::optional<torch::Tensor> mul_bias_,
    std::optional<torch::Tensor> add_bias_,
    std::optional<torch::Tensor> residual_) {
    auto undef_tensor = torch::Tensor();

    torch::Tensor norm_weights = norm_weights_.has_value() ? norm_weights_.value() : undef_tensor;
    torch::Tensor mul_bias = mul_bias_.has_value() ? mul_bias_.value() : undef_tensor;
    torch::Tensor add_bias = add_bias_.has_value() ? add_bias_.value() : undef_tensor;
    torch::Tensor residual = residual_.has_value() ? residual_.value() : undef_tensor;
    return muillm_linear_activ_forward(
        norm_weights,
        epsilon,
        weights,
        mui_activation::Identity,
        mul_bias,
        add_bias,
        residual,
        x
    );
}

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

at::Tensor muillm_rmsnorm_forward(
    torch::Tensor weights,
    torch::Tensor inputs,
    float epsilon);


std::vector<at::Tensor> muillm_rope_forward_no_cache(
    torch::Tensor& position_ids,
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in
);

std::vector<at::Tensor> muillm_rope_forward_dynamic_cache(
    torch::Tensor& position_ids,
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& prev_k_cache,
    torch::Tensor& prev_v_cache
);

std::vector<at::Tensor> muillm_rope_forward_static_cache(
    torch::Tensor& position_ids,
    torch::Tensor& cos_cached,
    torch::Tensor& sin_cached,
    torch::Tensor& q_in,
    torch::Tensor& k_in,
    torch::Tensor& v_in,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& cache_position
);

#include "causal_transformer_decoding.cuh"

std::vector<at::Tensor> muillm_parallel_causal_transformer_decoding_no_mask(
    std::vector<torch::Tensor>& qs, // [B, num_q_heads, T, embed_dim]
    std::vector<torch::Tensor>& ks, // [B, num_k_heads, NEW_T, embed_dim]
    std::vector<torch::Tensor>& vs  // [B, num_v_heads, NEW_T, embed_dim]
);

std::vector<at::Tensor> muillm_parallel_causal_transformer_decoding_masked(
    std::vector<torch::Tensor>& qs, // [B, num_q_heads, T, embed_dim]
    std::vector<torch::Tensor>& ks, // [B, num_k_heads, NEW_T, embed_dim]
    std::vector<torch::Tensor>& vs,  // [B, num_v_heads, NEW_T, embed_dim]
    std::vector<torch::Tensor>& ms  // [B, 1, NEW_T, T]
);

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

#include "comm.h"

// needed because Pybind11 can't seem to be able to deal with opaque pointers
struct muillm_comm_ptr {
  muillm_comm_t* comm_ptr;
};

muillm_comm_ptr muillm_comm_init_trampoline(
  int local_size,
  bool allocate_streams
) {
  muillm_comm_t* ptr;
  if (muillm_comm_init(local_size, allocate_streams, &ptr) != MUILLM_COMM_SUCCESS) {
    TORCH_CHECK(false, "comm init failed");
  }

  muillm_comm_ptr ret;
  ret.comm_ptr = ptr;
  return ret;
}

#include "comm_torch.cuh"

void muillm_all_reduce_sum_trampoline(
    muillm_comm_ptr comm,
    std::vector<torch::Tensor>& tensors
) {
  muillm_all_reduce_sum(comm.comm_ptr, tensors);
}

std::vector<torch::Tensor> muillm_broadcast_trampoline(
    muillm_comm_ptr comm,
    torch::Tensor& tensor
) {
  return muillm_broadcast(comm.comm_ptr, tensor);
}

#include "parallel_linear_kernels.cuh"

std::vector<at::Tensor> muillm_parallel_linear_forward_trampoline(
    muillm_comm_ptr comm,
    std::vector<torch::Tensor> x,
    std::vector<torch::Tensor> weights,
    std::optional<std::vector<torch::Tensor>> norm_weights_,
    float epsilon,
    std::optional<std::vector<torch::Tensor>> mul_biases_,
    std::optional<std::vector<torch::Tensor>> add_biases_,
    std::optional<torch::Tensor> residual_,
    int sharding_dim,
    bool reduce) {

    auto undef_tensor = torch::Tensor();
    std::vector<torch::Tensor> empty_tensor_list;

    std::vector<torch::Tensor>& norm_weights = norm_weights_.has_value() ? norm_weights_.value() : empty_tensor_list;
    std::vector<torch::Tensor>& mul_biases = mul_biases_.has_value() ? mul_biases_.value() : empty_tensor_list;
    std::vector<torch::Tensor>& add_biases = add_biases_.has_value() ? add_biases_.value() : empty_tensor_list;
    torch::Tensor residual = residual_.has_value() ? residual_.value() : undef_tensor;

    return muillm_parallel_linear_activ_forward(
        comm.comm_ptr,
        norm_weights,
        epsilon,
        weights,
        mui_activation::Identity,
        mul_biases,
        add_biases,
        residual,
        sharding_dim,
        reduce,
        x
    );
}

std::vector<at::Tensor> muillm_parallel_gateupsilu_forward(
    muillm_comm_t* comm,
    std::vector<torch::Tensor>& norm_weights,
    float epsilon,
    std::vector<torch::Tensor>& gate_weights,
    std::vector<torch::Tensor>& up_weights,
    std::vector<torch::Tensor>& down_weights,
    torch::Tensor& residual,
    std::vector<torch::Tensor>& x);

std::vector<at::Tensor> muillm_parallel_gateupsilu_split_forward(
    muillm_comm_t* comm,
    std::vector<torch::Tensor>& norm_weights,
    float epsilon,
    std::vector<torch::Tensor>& gate_weights,
    std::vector<torch::Tensor>& up_weights,
    std::vector<torch::Tensor>& down_weights,
    torch::Tensor& residual,
    std::vector<torch::Tensor>& x);

std::vector<at::Tensor> muillm_parallel_gateupsilu_forward_trampoline(
    muillm_comm_ptr comm,
    std::vector<torch::Tensor> norm_weights,
    float epsilon,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> down_weights,
    torch::Tensor residual,
    std::vector<torch::Tensor> x) {
  return muillm_parallel_gateupsilu_forward(
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

std::vector<at::Tensor> muillm_parallel_gateupsilu_split_forward_trampoline(
    muillm_comm_ptr comm,
    std::vector<torch::Tensor> norm_weights,
    float epsilon,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> down_weights,
    torch::Tensor residual,
    std::vector<torch::Tensor> x) {
  return muillm_parallel_gateupsilu_split_forward(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("muillm_linear_forward", &muillm_linear_forward_trampoline, "muillm linear forward", py::arg("x"), py::arg("weights"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none(), py::arg("residual") = py::none());
  m.def("muillm_parallel_linear_forward", &muillm_parallel_linear_forward_trampoline, "muillm parallel linear forward", py::arg("comm"), py::arg("x"), py::arg("weights"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_biases") = py::none(), py::arg("add_biases") = py::none(), py::arg("residual") = py::none(), py::arg("sharding_dim") = 1, py::arg("reduce") = false);
  m.def("muillm_int8_dequantize_forward", &muillm_int8_dequantize_forward, "muillm int8 dequantize forward");
  m.def("muillm_int8_linear_forward", &muillm_int8_linear_forward_trampoline, "muillm linear forward", py::arg("x"), py::arg("weights"), py::arg("scales_min_vals"), py::arg("group_size_shift"), py::arg("norm_weights") = py::none(), py::arg("epsilon") = 0.f, py::arg("mul_bias") = py::none(), py::arg("add_bias") = py::none());
  m.def("muillm_gateupsilu_forward", &muillm_gateupsilu_forward, "muillm gate up silu forward");
  m.def("muillm_parallel_gateupsilu_forward", &muillm_parallel_gateupsilu_forward_trampoline, "muillm parallel gate up silu forward");
  m.def("muillm_gateupsilu_split_forward", &muillm_gateupsilu_split_forward, "muillm gate up silu split K forward");
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
  m.def("muillm_parallel_causal_transformer_decoding_no_mask", &muillm_parallel_causal_transformer_decoding_no_mask, "muillm parallel causal transformer decoding no mask");
  m.def("muillm_parallel_causal_transformer_decoding_masked", &muillm_parallel_causal_transformer_decoding_masked, "muillm parallel causal transformer decoding masked");

  // synchronization
  pybind11::class_<muillm_synchronizer_ptr> cl_sync(m, "muillm_synchronizer_ptr");
  cl_sync.def(pybind11::init<>());

  m.def("muillm_sync_init", &muillm_sync_init_trampoline, "muillm sync init");
  m.def("muillm_item_bool", &muillm_item_bool_trampoline, "muillm item bool");
  m.def("muillm_item_f16", &muillm_item_f16_trampoline, "muillm item f16");
  m.def("muillm_item_f32", &muillm_item_f32_trampoline, "muillm item f32");
  m.def("muillm_to_cpu", &muillm_to_cpu_trampoline, "muillm to cpu");

  // comms
  pybind11::class_<muillm_comm_ptr> cl_comm(m, "muillm_comm_ptr");
  cl_comm.def(pybind11::init<>());

  m.def("muillm_comm_init", &muillm_comm_init_trampoline, "muillm comm init", py::arg("local_size"), py::arg("allocate_streams"));
  m.def("muillm_all_reduce_sum", &muillm_all_reduce_sum_trampoline, "muillm all reduce sum");
  m.def("muillm_broadcast", &muillm_broadcast_trampoline, "muillm broadcast");
}
