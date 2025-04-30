from typing import List, Tuple, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.linear import MuiLinear
from muillm.modules.module import MuiModule

from muillm.modules.moe.gateupdownmlpmoe import _MuiGateUpDownMoe
from muillm.modules.parallelgateupdownmlp import MuiParallelGateUpDownMLP
from muillm.modules.parallellinear import MuiParallelLinear
from muillm.modules.parallelmultilinear import MuiParallelMultiLinear
import torch
import torch.nn as nn

from transformers.models.llama4.modeling_llama4 import Llama4TextExperts, Llama4TextMoe


class MuiParallelExperts(MuiModule):
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

        self.comms = engine_config.comms
        self.tensor_parallelism = engine_config.tensor_parallelism

        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.expert_dim = intermediate_size

        self.tp_expert_dim = self.expert_dim // self.tensor_parallelism

        # use the storage (num_experts, out_features, in_features)
        # which is better for the custom kernels

        # We shard the gate and up projections by row so that we don't have to
        # do an all reduce before the down projection

        self.gate_projs = MuiParallelMultiLinear(
            engine_config=self.engine_config,
            in_features=self.hidden_size,
            out_features=self.num_experts * [self.expert_dim],
            bias=False,
            sharding_dim=0,
            device=device,
            dtype=dtype,
        )

        self.up_projs = MuiParallelMultiLinear(
            engine_config=self.engine_config,
            in_features=self.hidden_size,
            out_features=self.num_experts * [self.expert_dim],
            bias=False,
            sharding_dim=0,
            device=device,
            dtype=dtype,
        )

        self.down_projs = MuiParallelMultiLinear(
            engine_config=self.engine_config,
            in_features=self.expert_dim,
            out_features=self.num_experts * [self.hidden_size],
            bias=False,
            sharding_dim=1,
            device=device,
            dtype=dtype,
        )

        self.activation_function = activation_function

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    def _check_dispatchable(self):
        wdtype = self.gate_projs.linear.weights[0].dtype
        dispatchable_type = wdtype == torch.float16
        dispatchable_device = self.gate_projs.linear.weights[0].is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

    @staticmethod
    def replace(
        prev_module: Llama4TextExperts,
        engine_config: MuiEngineConfig,
        device=None,
    ) -> "MuiParallelExperts":

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

        new_module = MuiParallelExperts(
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

    def _extract_expert_linears(
        self, tensor: torch.Tensor, device, dtype
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
            linear.weight.requires_grad = tensor.requires_grad

            all_linears.append(linear)

        return all_linears

    def copy_module(self, prev_module: Llama4TextExperts, device=None):
        if device is None:
            raise ValueError("device was None")

        # TODO: the sharding of linear actually makes us have 2 experts per rank rather
        # than all ranks having all experts but only some rows

        device = prev_module.gate_up_proj.device if device is None else device
        dtype = prev_module.gate_up_proj.dtype

        # copy the shared expert

        # the original storage layyout is:
        # gate_up_proj  shape (num_experts, hidden_size, 2 * expert_dim)
        # down_proj shape (num_experts, expert_dim, hidden_size)

        gate_proj, up_proj = prev_module.gate_up_proj.chunk(2, dim=2)
        down_proj = prev_module.down_proj

        # we need to transpose the weights to match the new layout
        # which is (num_experts, expert_dim, hidden_size) for gate/up
        # and (num_experts, hidden_size, expert_dim) for down
        gate_linears = self._extract_expert_linears(
            tensor=gate_proj.reshape(
                self.num_experts, self.hidden_size, self.expert_dim
            )
            .transpose(1, 2)
            .contiguous(),
            device=device,
            dtype=dtype,
        )

        up_linears = self._extract_expert_linears(
            tensor=up_proj.reshape(self.num_experts, self.hidden_size, self.expert_dim)
            .transpose(1, 2)
            .contiguous(),
            device=device,
            dtype=dtype,
        )

        down_linears = self._extract_expert_linears(
            tensor=down_proj.reshape(
                self.num_experts, self.expert_dim, self.hidden_size
            )
            .transpose(1, 2)
            .contiguous(),
            device=device,
            dtype=dtype,
        )

        # Copy the gate_proj weights
        self.gate_projs.copy_modules(prev_modules=gate_linears, device=device)

        # delete the previous module to save memory
        for gate_linear in gate_linears:
            del gate_linear
        # trigger GC to save memory
        trigger_gc()

        # Copy the up_proj weights
        self.up_projs.copy_modules(prev_modules=up_linears, device=device)

        # delete the previous module to save memory
        for up_linear in up_linears:
            del up_linear
        # trigger GC to save memory
        trigger_gc()

        # Copy the down_proj weights
        self.down_projs.copy_modules(prev_modules=down_linears, device=device)

        # delete the previous module to save memory
        for down_linear in down_linears:
            del down_linear
        # trigger GC to save memory
        trigger_gc()

        # Need to synchronize after copying the tensors to make sure the transfers
        # completed
        MuiParallelLinear._sync_all(engine_config=self.engine_config)

    def parallel_forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        router_top_values: torch.Tensor,
        router_indices: torch.Tensor,
        shared_expert_output: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
        sharded_inputs = isinstance(hidden_states, list)
        if sharded_inputs:
            hidden_states = hidden_states[0]
        else:
            raise ValueError("not implemented")

        batch, seq_len, hidden_dim = hidden_states.shape
        tokens_per_expert = batch * seq_len

        if self.dispatchable and (batch * seq_len) == 1:

            # we are running on a single token, so we can use the custom kernel
            router_scores = torch.sigmoid(router_top_values.float()).to(
                hidden_states.dtype
            )

            moe_out = _MuiGateUpDownMoe.apply(
                self.engine_config,
                self.num_experts,
                hidden_states,
                router_scores,
                router_indices,
                None,  # norm_weights
                self.gate_projs.linear.weights[0],
                self.up_projs.linear.weights[0],
                self.down_projs.linear.weights[0],
                None,  # residual
                0,  # epsilon
            )

            # reduce across all the ranks the moe output
            moe_out = self.comms.all_reduce_sum(moe_out)

            # TODO: fuse the shared expert and the MoE computations
            out = shared_expert_output + moe_out

            return out, router_scores
        else:
            shared_expert_output = shared_expert_output.view(-1, hidden_dim)

            router_scores = (
                torch.full(
                    size=(batch, seq_len, self.num_experts),
                    fill_value=float("-inf"),
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
            router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

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
            gate_proj_weights = (
                self.gate_projs.linear.weights[0]
                .view(self.num_experts, self.tp_expert_dim, self.hidden_size)
                .transpose(1, 2)
            )

            up_proj_weights = (
                self.up_projs.linear.weights[0]
                .view(self.num_experts, self.tp_expert_dim, self.hidden_size)
                .transpose(1, 2)
            )

            down_proj_weights = (
                self.down_projs.linear.weights[0]
                .view(self.num_experts, self.hidden_size, self.tp_expert_dim)
                .transpose(1, 2)
            )

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
                ),  # shape (num_experts, -1, tp_expert_dim)
                down_proj_weights,
            )

            # reduce across all the ranks
            next_states = self.comms.all_reduce_sum(next_states)

            next_states = next_states.view(-1, self.hidden_size)

            # now that we finished expert computation -> we scatter add because we gathered previously
            # we have to do this because we used all experts on all tokens. This is faster than the for loop, tho you are compute bound
            # this scales a lot better if you do EP!
            shared_expert_output.scatter_add_(
                dim=0, index=router_indices, src=next_states.view(-1, hidden_dim)
            )

            return [shared_expert_output], [router_scores]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallelism > 1:
            out, score = self.parallel_forward([hidden_states])
            return out[0], score[0]

        raise ValueError("Only parallel inference is supported")


class MuiParallelGateUpDownMLPMoe(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: nn.Module,
        device=None,
        dtype=None,
    ):
        super().__init__(engine_config=engine_config)

        self.comms = engine_config.comms
        self.tensor_parallelism = engine_config.tensor_parallelism

        self.top_k = top_k
        self.hidden_dim = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function

        self.num_experts = num_experts

        # We do not shard the router, as it is small
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

        self.shared_expert = MuiParallelGateUpDownMLP(
            engine_config=engine_config,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            device=device,
            dtype=dtype,
        )

        self.experts = MuiParallelExperts(
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
        device=None,
    ) -> "MuiParallelGateUpDownMLPMoe":

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

        new_module = MuiParallelGateUpDownMLPMoe(
            engine_config=engine_config,
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

    def copy_module(self, prev_module: Llama4TextMoe, device=None):
        if device is None:
            raise ValueError("device was None")

        # copy the router

        self.router.copy_module(prev_module=prev_module.router, device=device)

        # copy the shared expert
        self.shared_expert.copy_module(
            prev_module=prev_module.shared_expert, device=device
        )

        # copy the experts
        self.experts.copy_module(prev_module=prev_module.experts, device=device)

    def parallel_forward(
        self, hidden_states: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        sharded_inputs = isinstance(hidden_states, list)
        if sharded_inputs:
            hidden_states = hidden_states[0]
        else:
            raise ValueError("not implemented")

        router_logits = self.router(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)

        shared_expert_out = self.shared_expert.parallel_forward([hidden_states])[0]

        out, scores_out = self.experts.parallel_forward(
            [hidden_states],
            router_top_values=router_top_value,
            router_indices=router_indices,
            shared_expert_output=shared_expert_out,
        )

        return out, scores_out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallelism > 1:
            outs, scores = self.parallel_forward([hidden_states])
            return outs[0], scores[0]

        raise ValueError("Only parallel inference is supported")
