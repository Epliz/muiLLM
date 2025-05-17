import torch
import torch.nn as nn

import muillm_ext


class _MuiTemperatureTuning(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_states, cache_position, attn_scale, floor_scale):
        output = muillm_ext.muillm_apply_temperature_tuning(
            query_states, cache_position, attn_scale, floor_scale
        )

        ctx.save_for_backward(query_states, cache_position)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("temperature tuning backward not implemented")
