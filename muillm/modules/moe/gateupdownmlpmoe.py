from typing import Optional, Tuple, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
from muillm.modules.linear import MuiLinear
from muillm.modules.module import MuiModule

import muillm_ext

import torch
import torch.nn as nn

from transformers.models.llama4.modeling_llama4 import (
    Llama4TextExperts,
    Llama4TextMoe,
    Llama4TextRMSNorm,
)

from muillm.modules.rmsnorm import MuiRMSNorm


class _MuiGateUpDownMoe(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        engine: MuiEngineConfig,
        shared_expert_output: torch.Tensor,
        num_experts: int,
        inputs: torch.Tensor,
        router_scores: torch.Tensor,
        router_indices: torch.Tensor,
        norm_weights: Optional[torch.Tensor],
        gate_weights: torch.Tensor,
        up_weights: torch.Tensor,
        down_weights: torch.Tensor,
        residual: Optional[torch.Tensor],
        epsilon,
    ):

        output = muillm_ext.muillm_gateupsilumoe_forward(
            engine.cpp_engine,
            shared_expert_output,
            num_experts,
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


class MuiExperts(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: nn.Module,
        device=None,
        dtype=None,
    ):
        super().__init__(engine_config=engine_config)
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.expert_dim = intermediate_size

        # use the storage (num_experts, out_features, in_features)
        # which is better for the custom kernels

        self.gate_projs = MuiLinear(
            engine_config=self.engine_config,
            in_features=hidden_size,
            out_features=num_experts * self.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.up_projs = MuiLinear(
            engine_config=self.engine_config,
            in_features=hidden_size,
            out_features=num_experts * self.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.down_projs = MuiLinear(
            engine_config=self.engine_config,
            in_features=intermediate_size,
            out_features=num_experts * hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.activation_function = activation_function

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        wdtype = self.gate_projs.weight.dtype
        dispatchable_type = wdtype == torch.float16
        dispatchable_device = self.gate_projs.weight.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    @staticmethod
    def replace(
        prev_module: Llama4TextExperts,
        engine_config: MuiEngineConfig,
        device=None,
    ) -> "MuiExperts":

        if device is None:
            raise ValueError("device was None")

        device = prev_module.gate_up_proj.device if device is None else device
        dtype = prev_module.gate_up_proj.dtype

        # put on the end device to accelerate things
        # (ok as we are replacing the module entirely so we can change its device)
        if device is not None:
            prev_module = prev_module.to(device)

        num_experts = prev_module.num_experts
        hidden_size = prev_module.hidden_size
        intermediate_size = prev_module.intermediate_size
        activation_function = prev_module.act_fn

        new_module = MuiExperts(
            engine_config=engine_config,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            device=device,
            dtype=dtype,
        )

        new_module.copy_module(prev_module=prev_module, device=device)

        # delete the previous module to save memory
        del prev_module
        # trigger GC to save memory
        trigger_gc()

        return new_module

    def copy_module(self, prev_module: Llama4TextExperts, device=None):
        if device is None:
            raise ValueError("device was None")

        # copy the shared expert

        # the original storage layyout is:
        # gate_up_proj  shape (num_experts, hidden_size, 2 * expert_dim)
        # down_proj shape (num_experts, expert_dim, hidden_size)

        gate_chunk, up_chunk = prev_module.gate_up_proj.chunk(2, dim=2)

        # we need to transpose the weights to match the new layout
        gate_weights = (
            gate_chunk.transpose(1, 2)
            .reshape(self.num_experts * self.intermediate_size, self.hidden_size)
            .contiguous()
        )
        up_weights = (
            up_chunk.transpose(1, 2)
            .reshape(self.num_experts * self.intermediate_size, self.hidden_size)
            .contiguous()
        )
        down_weights = (
            prev_module.down_proj.transpose(1, 2)
            .reshape(self.num_experts * self.hidden_size, self.intermediate_size)
            .contiguous()
        )

        self.gate_projs._set_weights(gate_weights)
        self.up_projs._set_weights(up_weights)
        self.down_projs._set_weights(down_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_top_values: torch.Tensor,
        router_indices: torch.Tensor,
        shared_expert_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This should really not be run on a single machine, as we are reaching compute bound:
        - the inputs are expected to be "sorted" per expert already.
        - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """

        batch, seq_len, hidden_dim = hidden_states.shape
        tokens_per_expert = batch * seq_len

        if self.dispatchable and (batch * seq_len) == 1:
            # we are running on a single token, so we can use the custom kernel

            out = _MuiGateUpDownMoe.apply(
                self.engine_config,
                shared_expert_output,
                self.num_experts,
                hidden_states,
                router_top_values,
                router_indices,
                None,  # norm_weights
                self.gate_projs.weight,
                self.up_projs.weight,
                self.down_projs.weight,
                None,  # residual
                0,  # epsilon
            )

            return out, None
        else:

            out_shape = hidden_states.shape

            shared_expert_output = shared_expert_output.view(-1, hidden_dim)

            router_scores = (
                torch.full(
                    size=(batch, seq_len, self.num_experts),
                    fill_value=0.0,
                    dtype=router_top_values.dtype,
                    device=router_top_values.device,
                )
                .scatter_(2, router_indices, router_top_values)
                .view(-1, self.num_experts)
                .transpose(0, 1)
            )
            # We do this to make sure we have -inf for non topK tokens before going through the !
            # Here we are just creating a tensor to index each and every single one of the hidden states. Let s maybe register a buffer for this!
            router_indices = (
                torch.arange(tokens_per_expert, device=hidden_states.device)
                .view(1, -1)
                .expand(router_scores.size(0), -1)
            )

            router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)

            routed_in = torch.gather(
                input=hidden_states.view(-1, self.hidden_size),
                dim=0,
                index=router_indices,
            ).to(hidden_states.device)
            # we gather inputs corresponding to each expert based on the router indices
            routed_in = routed_in * router_scores.reshape(-1, 1)

            hidden_states = routed_in.view(self.num_experts, -1, self.hidden_size)

            # the bmm operation requires the weights to be in the shape of (num_experts, expert_dim, hidden_size)
            # so we need to transpose them
            gate_proj_weights = self.gate_projs.weight.view(
                self.num_experts, self.expert_dim, self.hidden_size
            ).transpose(1, 2)

            up_proj_weights = self.up_projs.weight.view(
                self.num_experts, self.expert_dim, self.hidden_size
            ).transpose(1, 2)

            down_proj_weights = self.down_projs.weight.view(
                self.num_experts, self.hidden_size, self.expert_dim
            ).transpose(1, 2)

            g = torch.bmm(
                hidden_states,
                gate_proj_weights,
            )

            u = torch.bmm(
                hidden_states,
                up_proj_weights,
            )

            next_states = torch.bmm(
                (
                    self.activation_function(g) * u
                ),  # shape (num_experts, -1, expert_dim)
                down_proj_weights,
            )

            next_states = next_states.view(-1, self.hidden_size)

            # now that we finished expert computation -> we just aggregate all the expert contributions
            # with a sum reduction (works because we scaled inputs with 0 where not selected)
            shared_expert_output = shared_expert_output + next_states.view(
                -1, hidden_dim
            ).sum(dim=0)

            shared_expert_output = shared_expert_output.view(out_shape)

            return shared_expert_output, router_scores


class MuiGateUpDownMLPMoe(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: nn.Module,
        layernorm: Optional[MuiRMSNorm] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(engine_config=engine_config)

        self.top_k = top_k
        self.hidden_dim = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function

        self.num_experts = num_experts

        self.layernorm = layernorm

        self.router = MuiLinear(
            engine_config=engine_config,
            in_features=hidden_size,
            out_features=num_experts,
            bias=False,
            device=device,
            dtype=dtype,
        )
        # we don't want the router to be sharded, so mark as not a target for further
        # replacements
        self.router._muillm_no_further_replacement = True

        self.shared_expert = MuiGateUpDownMLP(
            engine_config=engine_config,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            device=device,
            dtype=dtype,
        )

        self.experts = MuiExperts(
            engine_config=engine_config,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            device=device,
            dtype=dtype,
        )

    def finalize_init(self):
        self.experts.finalize_init()

    @staticmethod
    def replace(
        prev_module: Llama4TextMoe,
        engine_config: MuiEngineConfig,
        prev_layernorm_module: Union[MuiRMSNorm, Llama4TextRMSNorm] = None,
        device=None,
    ) -> "MuiGateUpDownMLPMoe":

        if device is None:
            raise ValueError("device was None")

        device = prev_module.router.weight.device if device is None else device
        dtype = prev_module.router.weight.dtype

        num_experts = prev_module.num_experts
        top_k = prev_module.top_k

        # the shared expert and the MoE experts have the same shapes
        hidden_size = prev_module.shared_expert.gate_proj.in_features
        intermediate_size = prev_module.shared_expert.gate_proj.out_features

        activation_function = prev_module.experts.act_fn

        layernorm = None
        if prev_layernorm_module is not None:
            layernorm = MuiRMSNorm.replace(
                prev_module=prev_layernorm_module,
                engine_config=engine_config,
                device=device,
            )

        new_module = MuiGateUpDownMLPMoe(
            engine_config=engine_config,
            layernorm=layernorm,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            device=device,
            dtype=dtype,
        )

        new_module.copy_module(prev_module=prev_module, device=device)

        # delete the previous module to save memory
        del prev_module

        # trigger GC to save memory
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

        if prev_layernorm_module is not None:
            if self.layernorm is not None:
                self.layernorm.copy_module(
                    prev_module=prev_layernorm_module, device=device
                )
            else:
                raise ValueError(
                    "prev_layernorm_module was not None, but self.layernorm is None"
                )

        # copy the router

        self.router.copy_module(prev_module=prev_module.router, device=device)

        # copy the shared expert
        self.shared_expert.copy_module(
            prev_module=prev_module.shared_expert, device=device
        )

        # copy the experts
        self.experts.copy_module(prev_module=prev_module.experts, device=device)

    def forward(
        self,
        hidden_states,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.layernorm is not None:
            hidden_states = self.layernorm(hidden_states)

        router_logits = self.router(hidden_states)
        # TODO: custom kernel for topk+sigmoid
        router_top_values, router_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        router_top_values = torch.sigmoid(router_top_values.float()).to(
            router_top_values.dtype
        )

        shared_expert_out = self.shared_expert(
            hidden_states,
            residual=residual,
        )

        out, router_scores = self.experts(
            hidden_states,
            router_top_values=router_top_values,
            router_indices=router_indices,
            shared_expert_output=shared_expert_out,
        )

        return out, router_scores
