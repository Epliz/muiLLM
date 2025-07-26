from typing import Optional, Tuple, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.hftensorparallelism.hftensorparallelism import _to_local_module
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.linear import MuiLinear
from muillm.modules.module import MuiModule

import muillm_ext

import torch
import torch.nn as nn

from transformers.models.llama4.modeling_llama4 import (
    Llama4TextMoe,
    Llama4TextRMSNorm,
)

from muillm.modules.norm.rmsnorm import _MuiRMSNorm, MuiRMSNorm
from muillm.modules.topk import topk_sigmoid
from muillm.replacement.replacementcontext import MuiReplacementContext


class _MuiGateUpDownMoe(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        engine: MuiEngineConfig,
        num_shared_experts: int,
        num_dynamic_experts: int,
        inputs,
        router_scores,
        router_indices,
        norm_weights,
        gate_weights,
        up_weights,
        down_weights,
        residual,
        epsilon,
    ):

        output = muillm_ext.muillm_gateupsilumoe_forward(
            engine.cpp_engine,
            num_shared_experts,
            num_dynamic_experts,
            norm_weights,
            epsilon,
            gate_weights,
            up_weights,
            down_weights,
            residual,
            inputs,
            router_scores,
            router_indices,
        )

        ctx.save_for_backward(inputs)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Gate/Up Down MoE MLP backward is not implemented")


class MuiGateUpDownMLPMoe(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        num_dynamic_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: nn.Module,
        normalize: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(engine_config=engine_config)

        self.top_k = top_k
        self.hidden_dim = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function

        self.num_shared_experts = 1
        self.num_dynamic_experts = num_dynamic_experts
        self.num_experts = self.num_shared_experts + self.num_dynamic_experts

        # we fuse the layernorm into the router
        self.router = MuiLinear(
            engine_config=engine_config,
            in_features=hidden_size,
            out_features=self.num_dynamic_experts,
            bias=False,
            normalize=normalize,
            device=device,
            dtype=dtype,
        )
        # we don't want the router to be sharded, so mark as not a target for further
        # replacements
        self.router._muillm_no_further_replacement = True

        # Inline MuiExperts logic here
        self.expert_dim = intermediate_size

        # Shared and dynamic experts are packed together
        self.gate_projs = MuiLinear(
            engine_config=engine_config,
            in_features=hidden_size,
            out_features=self.num_experts * intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.up_projs = MuiLinear(
            engine_config=engine_config,
            in_features=hidden_size,
            out_features=self.num_experts * intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.down_projs = MuiLinear(
            engine_config=engine_config,
            in_features=intermediate_size,
            out_features=self.num_experts * hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        wdtype = self.gate_projs.weight.dtype
        dispatchable_type = (wdtype == torch.float16) or (wdtype == torch.bfloat16)
        dispatchable_device = self.gate_projs.weight.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def finalize_init(self):
        self._check_dispatchable()

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_module: Llama4TextMoe,
        prev_layernorm_module: Union[MuiRMSNorm, Llama4TextRMSNorm] = None,
    ) -> "MuiGateUpDownMLPMoe":
        engine_config = replacement_context.engine_config
        device = replacement_context.device

        if device is None:
            raise ValueError("device was None")

        prev_module = replacement_context.to_local_module(prev_module)

        if (prev_layernorm_module is not None) and (
            not isinstance(prev_layernorm_module, MuiRMSNorm)
        ):
            prev_layernorm_module = replacement_context.to_local_module(
                prev_layernorm_module
            )

        device = prev_module.router.weight.device if device is None else device
        dtype = prev_module.router.weight.dtype

        num_dynamic_experts = prev_module.num_experts
        top_k = prev_module.top_k
        hidden_size = prev_module.shared_expert.gate_proj.in_features
        intermediate_size = prev_module.shared_expert.gate_proj.out_features
        activation_function = prev_module.experts.act_fn

        new_module = MuiGateUpDownMLPMoe(
            engine_config=engine_config,
            num_dynamic_experts=num_dynamic_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            normalize=prev_layernorm_module is not None,
            device=device,
            dtype=dtype,
        )

        new_module.copy_module(
            prev_module=prev_module,
            prev_layernorm_module=prev_layernorm_module,
            device=device,
        )

        # delete the previous modules to free memory
        del prev_module.shared_expert
        del prev_module.experts
        del prev_module.router

        # trigger garbage collection to free memory
        trigger_gc()

        return new_module

    def copy_module(
        self,
        prev_module: Llama4TextMoe,
        prev_layernorm_module: Union[MuiRMSNorm, Llama4TextRMSNorm] = None,
        device=None,
    ):
        if device is None:
            raise ValueError("device was None")

        variance_epsilon = 0.0
        norm_weights = None
        if prev_layernorm_module is not None:
            variance_epsilon = MuiRMSNorm._extract_eps(prev_layernorm_module)
            norm_weights = prev_layernorm_module.weight

        self.router.copy_module(
            prev_module=prev_module.router,
            norm_weights=norm_weights,
            variance_epsilon=variance_epsilon,
            device=device,
        )

        ## Copy the experts
        prev_experts = prev_module.experts

        # ensure we have local tensors
        prev_experts_gate_up_proj = prev_experts.gate_up_proj
        prev_experts_down_proj = prev_experts.down_proj

        # reshape the dynamic expert weights to match the expected shape
        gate_chunk, up_chunk = prev_experts_gate_up_proj.chunk(2, dim=2)
        gate_weights = (
            gate_chunk.transpose(1, 2)
            .reshape(self.num_dynamic_experts * self.intermediate_size, self.hidden_dim)
            .contiguous()
        )
        up_weights = (
            up_chunk.transpose(1, 2)
            .reshape(self.num_dynamic_experts * self.intermediate_size, self.hidden_dim)
            .contiguous()
        )
        down_weights = (
            prev_experts_down_proj.transpose(1, 2)
            .reshape(self.num_dynamic_experts * self.hidden_dim, self.intermediate_size)
            .contiguous()
        )

        ## Shared expert weights

        # ensure we have local tensors
        shared_gate_weight = prev_module.shared_expert.gate_proj.weight
        shared_up_weight = prev_module.shared_expert.up_proj.weight
        shared_down_weight = prev_module.shared_expert.down_proj.weight

        # Pack shared expert and dynamic expert weights together
        gate_weights = torch.cat([shared_gate_weight, gate_weights], dim=0)
        up_weights = torch.cat([shared_up_weight, up_weights], dim=0)
        down_weights = torch.cat([shared_down_weight, down_weights], dim=0)

        self.gate_projs._set_weights(gate_weights)
        self.up_projs._set_weights(up_weights)
        self.down_projs._set_weights(down_weights)

    def forward(
        self,
        hidden_states,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute the router logits
        router_logits = self.router(hidden_states)
        router_top_values, router_indices = topk_sigmoid(router_logits, self.top_k)

        # compute the output of the MoE
        batch, seq_len, hidden_dim = hidden_states.shape
        tokens_per_expert = batch * seq_len

        if self.dispatchable and (batch * seq_len) == 1:
            moe_out = _MuiGateUpDownMoe.apply(
                self.engine_config,
                self.num_shared_experts,
                self.num_dynamic_experts,
                hidden_states,
                router_top_values,
                router_indices,
                self.router.norm_weights,  # norm_weights
                self.gate_projs.weight,
                self.up_projs.weight,
                self.down_projs.weight,
                residual,  # residual
                self.router.variance_epsilon,  # epsilon
            )

            return moe_out, router_top_values
        else:
            # normalize the hidden states if needed
            if self.router.normalize:
                hidden_states = _MuiRMSNorm.apply(
                    hidden_states,
                    self.router.norm_weights,
                    self.router.variance_epsilon,
                )

            out_shape = hidden_states.shape

            router_scores = (
                torch.full(
                    size=(batch, seq_len, self.num_dynamic_experts),
                    fill_value=0.0,
                    dtype=router_top_values.dtype,
                    device=router_top_values.device,
                )
                .scatter_(2, router_indices, router_top_values)
                .view(-1, self.num_dynamic_experts)
                .transpose(0, 1)
            )

            router_indices_flat = (
                torch.arange(tokens_per_expert, device=hidden_states.device)
                .view(1, -1)
                .expand(router_scores.size(0), -1)
            )
            router_indices_flat = router_indices_flat.reshape(-1, 1).expand(
                -1, hidden_dim
            )

            routed_in = torch.gather(
                input=hidden_states.view(-1, self.hidden_dim),
                dim=0,
                index=router_indices_flat,
            ).to(hidden_states.device)
            routed_in = routed_in * router_scores.reshape(-1, 1)

            hidden_states_expert = routed_in.view(
                self.num_dynamic_experts, -1, self.hidden_dim
            )

            # concatenate the hidden states for the shared expert
            hidden_states_expert = torch.cat(
                [
                    hidden_states.view(1, -1, self.hidden_dim).expand(
                        self.num_shared_experts, -1, -1
                    ),
                    hidden_states_expert,
                ],
                dim=0,
            )

            gate_proj_weights = self.gate_projs.weight.view(
                self.num_experts, self.expert_dim, self.hidden_dim
            ).transpose(1, 2)
            up_proj_weights = self.up_projs.weight.view(
                self.num_experts, self.expert_dim, self.hidden_dim
            ).transpose(1, 2)
            down_proj_weights = self.down_projs.weight.view(
                self.num_experts, self.hidden_dim, self.expert_dim
            ).transpose(1, 2)

            g = torch.bmm(
                hidden_states_expert,
                gate_proj_weights,
            )
            u = torch.bmm(
                hidden_states_expert,
                up_proj_weights,
            )
            next_states = torch.bmm(
                (self.activation_function(g) * u),
                down_proj_weights,
            )

            # extract the shared expert output
            shared_expert_output = next_states[: self.num_shared_experts, :, :]

            shared_expert_output = shared_expert_output.view(-1, hidden_dim)

            # extract the dynamic expert output
            next_states = next_states[self.num_shared_experts :, :, :]

            next_states = next_states.view(-1, self.hidden_dim)
            shared_expert_output.scatter_add_(
                dim=0, index=router_indices_flat, src=next_states.view(-1, hidden_dim)
            )
            out = shared_expert_output.view(out_shape)

            # apply the residual if provided
            if residual is not None:
                out = out + residual

            return out, router_scores
