import math
from typing import Tuple
import torch

def _ispow2(x: int):
    return (x & (x-1) == 0) and x != 0

def _int_log2(x: int):
    l = math.log2(x)
    return int(l)

class RTNQuantizer:
    def __init__(self, n_bit: int = 8, groupsize: int = 128, min_bias = 0, max_bias = 0, f=1.0):
        self.n_bit = n_bit

        if not _ispow2(groupsize):
            raise ValueError(f"RTNQuantizer only supports power of two group sizes, but the provided groupsize {groupsize} is not")

        self.group_size_shift = _int_log2(groupsize)

        self.min_bias = min_bias
        self.max_bias = max_bias
        self.f = f

    def _check_group_size(self, groupsize: int):
        if groupsize <= 0:
            raise ValueError(f"groupsize must be greater than 0 but was {groupsize}")
        
    def _check_group_size_weight_size_compatible(self, groupsize: int, w: torch.Tensor):
        if w.shape[-1] % groupsize != 0:
            raise ValueError(f"groupsize must divide the weight dimension but {groupsize} doesn't divide {w.shape[-1]}")
        
    def _check_2d_weights(self, w: torch.Tensor):
        if w.dim() != 2:
            raise ValueError(f"Only 2D tensors can be quantized, but w has {w.dim()} dimensions")
        
    def _check_no_nans(self, w: torch.Tensor):
        if torch.isnan(w).sum() > 0:
            raise ValueError("Cannot quantize tensors with NaN values")

    def _pack_scales_mins(self, scales: torch.Tensor, min_vals: torch.Tensor):
        return torch.concat([scales, min_vals], dim=-1) # shape [N*G, 2]

    def _unpack_scales_mins(self, scales_mins: torch.Tensor):
        scales = scales_mins[:,None,0] # shape [N*G, 1]
        mins = scales_mins[:,None,1] # shape [N*G, 1]
        return scales, mins

    def group_quantize_tensor(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        groupsize = 1 << self.group_size_shift
        n_bit = self.n_bit
        f = self.f
    
        self._check_group_size(groupsize)
        self._check_group_size_weight_size_compatible(groupsize, w)
        self._check_2d_weights(w)

        wdtype = w.dtype

        # use f32 to avoid accuracy loss during the quantization process
        to_quant = w.reshape(-1, groupsize).to(dtype=torch.float32)
        self._check_no_nans(to_quant)

        max_val = to_quant.amax(dim=1, keepdim=True) + self.max_bias
        min_val = to_quant.amin(dim=1, keepdim=True) + self.min_bias
        uint_range = 2**n_bit - 1

        # compute the value range
        val_range = (f * (max_val - min_val)).clamp(min=1e-6)
        val_center = (min_val + max_val) * 0.5

        # recomput min_val and max_val after potentially modifying the value range
        min_val: torch.Tensor = val_center - val_range * 0.5 # shape [N*G, 1]
        max_val: torch.Tensor = val_center + val_range * 0.5

        # compute scales and min_val
        scales: torch.Tensor = val_range / uint_range # shape [N*G, 1]

        w_int8 = (
            to_quant.sub(min_val) # in range [0, max_val - min_val]
            .div(scales) # in range [0, uint_range]
            .clamp(0, uint_range)
            .round()
            .to(torch.uint8)
            .reshape_as(w)
        )

        return w_int8, self._pack_scales_mins(scales, min_val).to(dtype=wdtype)


    def group_dequantize_tensor(self, w_uint8: torch.Tensor, scales_min_vals: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        groupsize = 1 << self.group_size_shift

        self._check_group_size(groupsize)
        self._check_group_size_weight_size_compatible(groupsize, w_uint8)
        self._check_2d_weights(w_uint8)

        w_uint8_grouped = w_uint8.reshape(-1, groupsize)
        scales, min_vals = self._unpack_scales_mins(scales_min_vals)

        w_dq = w_uint8_grouped.mul(scales).add(min_vals).reshape_as(w_uint8)
        return w_dq.to(dtype=dtype)