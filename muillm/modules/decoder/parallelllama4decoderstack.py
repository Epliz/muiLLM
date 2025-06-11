import muillm_ext

import torch
import torch.nn as nn


class _MuiParallelLlama4DecoderStack(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module,
        cache_module,
        h,
        attention_mask,
        chunked_mask,
        position_embeddings,
        cache_positions,
    ):
        output = muillm_ext.muillm_parallel_llama4_decoder_stack_forward(
            module,
            cache_module,
            h,
            attention_mask,
            chunked_mask,
            position_embeddings,
            cache_positions,
        )

        ctx.save_for_backward(h, attention_mask, chunked_mask)

        return output
