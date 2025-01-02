#include "gateup_kernels.cuh"

std::vector<at::Tensor> muillm_parallel_gateupsilu_forward(
    std::vector<torch::Tensor> norm_weights,
    float epsilon,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> down_weights,
    torch::Tensor residual,
    std::vector<torch::Tensor> x) {
  size_t tp_level = gate_weights.size();

  std::vector<at::Tensor> outputs;

  auto undef_tensor = torch::Tensor();
  for (size_t t = 0; t < tp_level; t++) {
    outputs.push_back(
        muillm_gateupsilu_forward(
            norm_weights[t],
            epsilon,
            gate_weights[t],
            up_weights[t],
            down_weights[t],
            t == 0 ? residual : undef_tensor,
            x[t]
        )
    );
  }

  return outputs;
}

std::vector<at::Tensor> muillm_parallel_gateupsilu_split_forward(
    std::vector<torch::Tensor> norm_weights,
    float epsilon,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> down_weights,
    torch::Tensor residual,
    std::vector<torch::Tensor> x) {
  size_t tp_level = gate_weights.size();

  std::vector<at::Tensor> outputs;

  auto undef_tensor = torch::Tensor();
  for (size_t t = 0; t < tp_level; t++) {
    outputs.push_back(
        muillm_gateupsilu_split_forward(
            norm_weights[t],
            epsilon,
            gate_weights[t],
            up_weights[t],
            down_weights[t],
            t == 0 ? residual : undef_tensor,
            x[t]
        )
    );
  }

  return outputs;
}