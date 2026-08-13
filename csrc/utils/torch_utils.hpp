#ifndef __MUILLM_TORCH_UTILS_HPP__
#define __MUILLM_TORCH_UTILS_HPP__

#include <torch/extension.h>

// return true if the last dimension of the tensor has stride 1,
// the second to last has a stride potentially different from the size of the last
// dimension, and all other dimensions have a stride equal to the size of the next dimension
bool muillm_torch_tensor_is_mostly_contiguous(torch::Tensor& tensor);

#endif // __MUILLM_TORCH_UTILS_HPP__