#include "torch_utils.hpp"


// return true if the last dimension of the tensor has stride 1,
// the second to last has a stride potentially different from the size of the last
// dimension, and all other dimensions have a stride equal to the size of the next dimension
bool muillm_torch_tensor_is_mostly_contiguous(torch::Tensor& tensor) {
  auto ndim = tensor.dim();
    
  auto sizes = tensor.sizes();
  auto strides = tensor.strides();
    
  if (strides[ndim - 1] != 1) {
    // last dimension must have stride 1
    return false;
  }

  unsigned total_stride = 1;
  for (int d = ndim - 2; d >= 0; d--) {
    if (d == ndim - 2) {
      // second to last dimension can have a stride different from the size of the last dimension
      total_stride = strides[d];
      continue;
    }
    total_stride *= sizes[d + 1];
    if (strides[d] != total_stride) {
      return false;
    }
  }
    
  return true;
}