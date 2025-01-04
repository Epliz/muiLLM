#include "gateup_kernels.cuh"
#include "comm_torch.cuh"

std::vector<at::Tensor> muillm_parallel_gateupsilu_forward(
    muillm_comm_t* comm,
    std::vector<torch::Tensor>& norm_weights,
    float epsilon,
    std::vector<torch::Tensor>& gate_weights,
    std::vector<torch::Tensor>& up_weights,
    std::vector<torch::Tensor>& down_weights,
    torch::Tensor& residual,
    std::vector<torch::Tensor>& x) {
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

  // we always have to reduce
  muillm_all_reduce_sum(comm, outputs);

  return outputs;
}

std::vector<at::Tensor> muillm_parallel_gateupsilu_split_forward(
    muillm_comm_t* comm,
    std::vector<torch::Tensor>& norm_weights,
    float epsilon,
    std::vector<torch::Tensor>& gate_weights,
    std::vector<torch::Tensor>& up_weights,
    std::vector<torch::Tensor>& down_weights,
    torch::Tensor& residual,
    std::vector<torch::Tensor>& x) {
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

  // we always have to reduce
  muillm_all_reduce_sum(comm, outputs);

  return outputs;
}