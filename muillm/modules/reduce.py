import torch

import muillm_ext


class _MuiReducetorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, dim, keepdim=False):
        output = muillm_ext.muillm_reduce_sum_forward(
            inputs,
            dim,
            keepdim,
        )

        ctx.save_for_backward(inputs)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Reduce backward is not implemented")


def reduce_sum(inputs: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    Reduce the input tensor by summing over the specified dimension.
    Args:
        inputs (torch.Tensor): The input tensor to reduce.
        dim (int): The dimension to reduce.
        keepdim (bool, optional): Whether to keep the reduced dimension. Defaults to False.
    Returns:
        torch.Tensor: The reduced tensor.
    """
    if inputs.is_cuda and (
        (inputs.dtype == torch.float16) or (inputs.dtype == torch.bfloat16)
    ):
        return _MuiReducetorch.apply(inputs, dim, keepdim)

    return torch.sum(inputs, dim=dim, keepdim=keepdim)
