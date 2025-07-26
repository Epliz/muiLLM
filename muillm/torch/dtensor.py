import torch

import torch.nn as nn


def to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a potentially distributed tensor to a local tensor.

    Args:
        tensor (torch.Tensor): The potentially distributed tensor to convert.

    Returns:
        torch.Tensor: The local tensor.
    """
    try:
        # DTensor is available only in recent versions of PyTorch
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, nn.Parameter):
            tensor = tensor.data

        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()

        return tensor.detach()  # normal tensor
    except ImportError:
        # If DTensor is not available, return the tensor as is
        return tensor
