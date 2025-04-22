from typing import List
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
from muillm.modules.moe.gateupdownmlpmoe import MuiGateUpDownMLPMoe
import torch
import torch.nn as nn

from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

from .test_utils import tensors_equal


def random_llama4_moe_mlp(hidden_size: int, intermediate_size: int) -> Llama4TextMoe:
    config = Llama4TextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    mlp = Llama4TextMoe(config)

    # We seed to have reproducible results
    torch.manual_seed(0)

    # initialize weights

    # router
    torch.nn.init.xavier_uniform_(mlp.router.weight)

    # shared expert
    torch.nn.init.xavier_uniform_(mlp.shared_expert.gate_proj.weight)
    torch.nn.init.xavier_uniform_(mlp.shared_expert.up_proj.weight)
    torch.nn.init.xavier_uniform_(mlp.shared_expert.down_proj.weight)

    # experts
    torch.nn.init.xavier_uniform_(mlp.experts.gate_up_proj)
    torch.nn.init.xavier_uniform_(mlp.experts.down_proj)

    return mlp


def copy_llama4_moe_mlp(mlp: Llama4TextMoe) -> Llama4TextMoe:

    num_experts = mlp.num_experts
    top_k = mlp.top_k
    hidden_size = mlp.hidden_dim
    intermediate_size = mlp.experts.intermediate_size

    config = Llama4TextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    new_mlp = Llama4TextMoe(config=config)

    # router
    new_mlp.router.weight = nn.Parameter(mlp.router.weight.clone().detach())

    # shared expert
    new_mlp.shared_expert.gate_proj.weight = nn.Parameter(
        mlp.shared_expert.gate_proj.weight.clone().detach()
    )
    new_mlp.shared_expert.up_proj.weight = nn.Parameter(
        mlp.shared_expert.up_proj.weight.clone().detach()
    )
    new_mlp.shared_expert.down_proj.weight = nn.Parameter(
        mlp.shared_expert.down_proj.weight.clone().detach()
    )

    # experts
    new_mlp.experts.gate_up_proj = nn.Parameter(
        mlp.experts.gate_up_proj.clone().detach()
    )
    new_mlp.experts.down_proj = nn.Parameter(mlp.experts.down_proj.clone().detach())

    return new_mlp


def test_basic_llama4_moe_mlp():
    hidden_size = 128
    intermediate_size = 256
    moe_mlp = random_llama4_moe_mlp(
        hidden_size=hidden_size, intermediate_size=intermediate_size
    )

    # replace destroys the passed linear module so we need to copy it
    moe_mlp_copy = copy_llama4_moe_mlp(moe_mlp)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    muimlp = MuiGateUpDownMLPMoe.replace(
        prev_module=moe_mlp_copy,
        engine_config=engine_config,
        device="cpu",
    )

    # check that outputs seem fine
    input_tensor = torch.rand(size=(1, 3, hidden_size))

    y, scores = moe_mlp(input_tensor)

    y_m, scores_m = muimlp(input_tensor)

    tensors_equal(y, y_m)
    tensors_equal(scores, scores_m)


# TODO tests with bias and no bias
# TODO tests with input norm
# TODO tests with other data types
