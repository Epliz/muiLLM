import muillm_ext

import torch
import torch.nn as nn


class _MuiParallelDecoderStack(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module,
        cache_module,
        input_ids,
        input_embeds,
        m,
        position_ids,
        cache_positions,
    ):
        output = muillm_ext.muillm_parallel_decoder_stack_forward(
            module,
            cache_module,
            input_ids,
            input_embeds,
            m,
            position_ids,
            cache_positions,
        )

        return output
