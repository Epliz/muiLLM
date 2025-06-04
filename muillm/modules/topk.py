from typing import Tuple

import muillm_ext

import torch


class _MuiTopKSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        router_logits: torch.Tensor,
        top_k: int,
    ):

        output = muillm_ext.muillm_topk_sigmoid_forward(
            router_logits,
            top_k,
        )

        ctx.save_for_backward(router_logits)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("top K sigmoid backward is not implemented")


def topk_sigmoid(
    router_logits: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if router_logits.dtype == torch.float16:
        # dispatchablable to custom kernel
        return _MuiTopKSigmoid.apply(
            router_logits,
            top_k,
        )
    else:
        router_top_values, router_indices = torch.topk(router_logits, top_k, dim=-1)
        router_top_values = torch.sigmoid(router_top_values.float()).to(
            router_top_values.dtype
        )
        return router_top_values, router_indices
