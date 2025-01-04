#include "parallel_linear_kernels.cuh"
#include "comm_torch.cuh"

std::vector<at::Tensor> muillm_parallel_linear_activ_forward(
    muillm_comm_t* comm,
    std::vector<torch::Tensor>& norm_weights,
    float epsilon,
    std::vector<torch::Tensor>& weights,
    mui_activation activ,
    std::vector<torch::Tensor>& mul_bias,
    std::vector<torch::Tensor>& add_bias,
    torch::Tensor& residual,
    bool reduce,
    std::vector<torch::Tensor>& x
) {
  size_t tp_level = weights.size();

  std::vector<at::Tensor> outputs;

  size_t norm_weights_size = norm_weights.size();
  size_t mul_bias_size = mul_bias.size();
  size_t add_bias_size = add_bias.size();

  auto undef_tensor = torch::Tensor();

  for (size_t t = 0; t < tp_level; t++) {
    outputs.push_back(
        muillm_linear_activ_forward(
            t < norm_weights_size ? norm_weights[t] : undef_tensor,
            epsilon,
            weights[t],
            activ,
            t < mul_bias_size ?  mul_bias[t] : undef_tensor,
            t < add_bias_size ? add_bias[t] : undef_tensor,
            // we apply the residual only on device 0
            t == 0 ? residual : undef_tensor,
            x[t]
        )
    );
  }

  if (reduce) {
    muillm_all_reduce_sum(comm, outputs);
  }

  return outputs;
}