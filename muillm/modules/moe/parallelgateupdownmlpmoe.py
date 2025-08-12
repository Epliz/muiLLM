from typing import List, Optional, Tuple, Union
from muillm.comms.communicator import MuiCommunicator
from muillm.engineconfig import MuiEngineConfig
from muillm.hftensorparallelism.hftensorparallelism import _to_local_module
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.linear import MuiLinear
from muillm.modules.module import MuiModule

from muillm.modules.parallellinear import MuiParallelLinear
from muillm.modules.parallelmultilinear import MuiParallelMultiLinear
import torch
import torch.nn as nn

from transformers.models.llama4.modeling_llama4 import (
    Llama4TextMoe,
    Llama4TextRMSNorm,
)

from muillm.modules.norm.rmsnorm import _MuiRMSNorm, MuiRMSNorm

import muillm_ext

from muillm.modules.topk import topk_sigmoid
from muillm.replacement.replacementcontext import MuiReplacementContext


class _MuiParallelGateUpDownMoe(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        module,
        inputs,
        residual,
        reduce=True,
    ):

        output = muillm_ext.muillm_parallel_gateupdownmlpmoe_module_forward(
            module, inputs, residual, reduce
        )

        ctx.save_for_backward(inputs)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Gate/Up Down MoE MLP backward is not implemented")


class MuiParallelGateUpDownMLPMoe(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        num_dynamic_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: nn.Module,
        norm: Optional[MuiRMSNorm] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(engine_config=engine_config)

        self.cpp_engine = engine_config.cpp_engine
        # the cpp module will be created at the end of all layer replacements
        # (set the field here before potential OOM errors so that it can still be manipulated in
        # the destructor)
        self.cpp_module = None
        self.comms = engine_config.comms
        self.tensor_parallelism = engine_config.tensor_parallelism

        self.top_k = top_k
        self.hidden_dim = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function

        self.num_shared_experts = 1
        self.num_dynamic_experts = num_dynamic_experts
        self.num_experts = self.num_shared_experts + self.num_dynamic_experts

        self.norm = norm

        # We fuse the layernorm into the router
        # We do not shard the router, as it is small
        self.router = MuiLinear(
            engine_config=engine_config,
            in_features=hidden_size,
            out_features=self.num_dynamic_experts,
            bias=False,
            norm=norm,
            device=device,
            dtype=dtype,
        )
        # we don't want the router to be sharded, so mark as not a target for further
        # replacements
        self.router._muillm_no_further_replacement = True

        self.expert_dim = intermediate_size
        self.tp_expert_dim = self.expert_dim // self.tensor_parallelism

        self.gate_projs = MuiParallelMultiLinear(
            engine_config=engine_config,
            in_features=hidden_size,
            out_features=self.num_experts * [self.expert_dim],
            bias=False,
            sharding_dim=0,
            device=device,
            dtype=dtype,
        )

        self.up_projs = MuiParallelMultiLinear(
            engine_config=engine_config,
            in_features=hidden_size,
            out_features=self.num_experts * [self.expert_dim],
            bias=False,
            sharding_dim=0,
            device=device,
            dtype=dtype,
        )

        self.down_projs = MuiParallelMultiLinear(
            engine_config=engine_config,
            in_features=self.expert_dim,
            out_features=self.num_experts * [hidden_size],
            bias=False,
            sharding_dim=1,
            device=device,
            dtype=dtype,
        )

        self._check_dispatchable()

    def _check_dispatchable(self):
        wdtype = self.gate_projs.linear.weights[0].dtype
        dispatchable_type = (wdtype == torch.float16) or (wdtype == torch.bfloat16)
        dispatchable_device = self.gate_projs.linear.weights[0].is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def finalize_init(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_gateupdownmlpmoe_module_deinit(self.cpp_module)

        normalize = self.norm is not None

        # make sure the router is initialized
        self.router.finalize_init()

        self.cpp_module = muillm_ext.muillm_parallel_gateupdownmlpmoe_module_init(
            self.cpp_engine,
            self.comms.comms,
            self.router.cpp_module,
            self.num_shared_experts,
            self.num_dynamic_experts,
            self.top_k,
            self.norm.weight if normalize else None,
            self.gate_projs.linear.weights[0],
            self.up_projs.linear.weights[0],
            self.down_projs.linear.weights[0],
            self.norm.variance_epsilon if normalize else 0.0,
            self.norm.weight_offset if normalize else 0.0,
        )

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def finalize_deinit(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_gateupdownmlpmoe_module_deinit(self.cpp_module)
            self.cpp_module = None

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_module: Llama4TextMoe,
        prev_layernorm_module: Union[MuiRMSNorm, Llama4TextRMSNorm] = None,
    ) -> "MuiParallelGateUpDownMLPMoe":
        engine_config = replacement_context.engine_config
        device = replacement_context.device
        if device is None:
            raise ValueError("device was None")

        # Make sure we convert the previous module to a local module
        prev_module.experts = replacement_context.to_local_module(prev_module.experts)
        prev_module.shared_expert = replacement_context.to_local_module(
            prev_module.shared_expert
        )
        prev_module.router = replacement_context.to_local_module(prev_module.router)

        if (prev_layernorm_module is not None) and (
            not isinstance(prev_layernorm_module, MuiRMSNorm)
        ):
            # Make sure we convert the previous layernorm module to a local module
            # so that we can safely copy its parameters
            prev_layernorm_module = replacement_context.to_local_module(
                prev_layernorm_module
            )

        device = prev_module.router.weight.device if device is None else device
        dtype = prev_module.router.weight.dtype

        num_dynamic_experts = prev_module.num_experts
        top_k = prev_module.top_k

        # the shared expert and the MoE experts have the same shapes
        hidden_size = prev_module.shared_expert.gate_proj.in_features
        intermediate_size = prev_module.shared_expert.gate_proj.out_features

        activation_function = prev_module.experts.act_fn

        norm = (
            MuiRMSNorm.replace(
                replacement_context,
                prev_layernorm_module,
            )
            if prev_layernorm_module is not None
            else None
        )

        new_module = MuiParallelGateUpDownMLPMoe(
            engine_config=engine_config,
            num_dynamic_experts=num_dynamic_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            norm=norm,
            device=device,
            dtype=dtype,
        )

        new_module.copy_module(
            prev_module=prev_module,
            device=device,
        )

        # delete the previous modules to free memory
        del prev_module.shared_expert
        del prev_module.experts
        del prev_module.router

        # trigger garbage collection to free memory
        trigger_gc()

        return new_module

    def _extract_expert_linears(
        self, tensor: torch.Tensor, device, dtype, requires_grad: Optional[bool] = None
    ) -> List[nn.Linear]:
        num_experts, out_features, in_features = tensor.shape

        tensors = tensor.chunk(num_experts, dim=0)

        if len(tensors) != num_experts:
            raise ValueError(
                f"Expected {num_experts} tensors, got {len(tensors)} instead"
            )

        all_linears = []

        for i in range(num_experts):
            linear = MuiLinear(
                engine_config=self.engine_config,
                in_features=in_features,
                out_features=out_features,
                bias=False,
                device=device,
                dtype=dtype,
            )

            linear.weight = nn.Parameter(
                tensors[i].reshape(out_features, in_features).clone().detach()
            )
            # TODO: check if requires_grad is set correctly
            linear.weight.requires_grad = tensor.requires_grad

            all_linears.append(linear)

        return all_linears

    def copy_module(
        self,
        prev_module: Llama4TextMoe,
        device=None,
    ):
        if device is None:
            raise ValueError("device was None")

        # copy the router

        self.router.copy_module(
            prev_module=prev_module.router,
            device=device,
        )

        prev_experts = prev_module.experts

        device = prev_experts.gate_up_proj.device if device is None else device
        dtype = prev_experts.gate_up_proj.dtype

        ## Shared expert weights

        # reshape the shared expert weights to match the expected shape

        shared_gate_weight = prev_module.shared_expert.gate_proj.weight.view(
            1, self.expert_dim, self.hidden_dim
        )
        shared_up_weight = prev_module.shared_expert.up_proj.weight.view(
            1, self.expert_dim, self.hidden_dim
        )
        shared_down_weight = prev_module.shared_expert.down_proj.weight.view(
            1, self.hidden_dim, self.expert_dim
        )

        ## dynamic expert weights

        # reshape the dynamic expert weights to match the expected shape
        gate_chunk, up_chunk = prev_experts.gate_up_proj.chunk(2, dim=2)
        down_proj = prev_experts.down_proj

        gate_weights = gate_chunk.reshape(
            self.num_dynamic_experts, self.hidden_dim, self.expert_dim
        ).transpose(1, 2)
        up_weights = up_chunk.reshape(
            self.num_dynamic_experts, self.hidden_dim, self.expert_dim
        ).transpose(1, 2)
        down_weights = down_proj.reshape(
            self.num_dynamic_experts, self.expert_dim, self.hidden_dim
        ).transpose(1, 2)

        # pack all the weights together
        all_gate_weights = torch.cat(
            [
                shared_gate_weight,
                gate_weights,
            ],
            dim=0,
        )
        all_up_weights = torch.cat(
            [
                shared_up_weight,
                up_weights,
            ],
            dim=0,
        )
        all_down_weights = torch.cat(
            [
                shared_down_weight,
                down_weights,
            ],
            dim=0,
        )

        gate_linears = self._extract_expert_linears(
            tensor=all_gate_weights,
            device=device,
            dtype=dtype,
        )

        up_linears = self._extract_expert_linears(
            tensor=all_up_weights,
            device=device,
            dtype=dtype,
        )

        down_linears = self._extract_expert_linears(
            tensor=all_down_weights,
            device=device,
            dtype=dtype,
        )

        self.gate_projs.copy_modules(prev_modules=gate_linears, device=device)
        self.up_projs.copy_modules(prev_modules=up_linears, device=device)
        self.down_projs.copy_modules(prev_modules=down_linears, device=device)

        MuiParallelLinear._sync_all(engine_config=self.engine_config)

    def parallel_forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        sharded_inputs = isinstance(hidden_states, list)
        if sharded_inputs:
            hidden_states = hidden_states[0]
        else:
            raise ValueError("not implemented")

        batch, seq_len, hidden_dim = hidden_states.shape
        num_tokens = batch * seq_len

        if self.dispatchable and num_tokens == 1:
            moe_out = _MuiParallelGateUpDownMoe.apply(
                self.cpp_module,
                hidden_states,
                residual,
                True,  # reduce
            )
            return [moe_out], [None]
        else:
            router_logits = self.router(hidden_states)
            router_top_values, router_indices = topk_sigmoid(router_logits, self.top_k)

            # normalize the hidden states if needed
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)

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
            router_indices_tensor = (
                torch.arange(num_tokens, device=hidden_states.device)
                .view(1, -1)
                .expand(router_scores.size(0), -1)
            )
            router_indices_tensor = router_indices_tensor.reshape(-1, 1).expand(
                -1, hidden_dim
            )

            routed_in = torch.gather(
                input=hidden_states.view(-1, self.hidden_dim),
                dim=0,
                index=router_indices_tensor,
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

            gate_proj_weights = (
                self.gate_projs.linear.weights[0]
                .view(self.num_experts, self.tp_expert_dim, self.hidden_dim)
                .transpose(1, 2)
            )

            up_proj_weights = (
                self.up_projs.linear.weights[0]
                .view(self.num_experts, self.tp_expert_dim, self.hidden_dim)
                .transpose(1, 2)
            )

            down_proj_weights = (
                self.down_projs.linear.weights[0]
                .view(self.num_experts, self.hidden_dim, self.tp_expert_dim)
                .transpose(1, 2)
            )

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
                dim=0, index=router_indices_tensor, src=next_states.view(-1, hidden_dim)
            )

            out = shared_expert_output.view(out_shape)

            # apply the residual if provided
            if (residual is not None) and self.comms.rank == 0:
                out = out + residual

            out = self.comms.all_reduce_sum(out)

            return [out], [router_scores]

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.tensor_parallelism > 1:
            outs, scores = self.parallel_forward(
                [hidden_states],
                residual=residual,
            )
            return outs[0], scores[0]

        raise ValueError("Only parallel inference is supported")
