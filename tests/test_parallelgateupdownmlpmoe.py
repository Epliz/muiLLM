from muillm.engineconfig import MuiEngineConfig
import torch
import torch.nn as nn

from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

from muillm.modules.moe.parallelgateupdownmlpmoe import MuiParallelGateUpDownMLPMoe
from muillm.replacement.replacementcontext import MuiReplacementContext

from .test_utils import execute_distributed, tensors_equal


def random_llama4_moe_mlp(
    hidden_size: int, intermediate_size: int, device: str, dtype=torch.float32
) -> Llama4TextMoe:
    config = Llama4TextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=16,
        hidden_act="silu",
        initializer_factor=0.02,
        layer_norm_eps=1e-5,
    )

    mlp = Llama4TextMoe(config).to(device=device, dtype=dtype)

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
    device = mlp.router.weight.device

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

    new_mlp = Llama4TextMoe(config=config).to(device)

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


def _test_basic_llama4_moe_mlp(
    size,
    hidden_size: int,
    intermediate_size: int,
    device: str,
    dtype=torch.float32,
):
    moe_mlp = random_llama4_moe_mlp(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
        dtype=dtype,
    )

    # replace destroys the passed linear module so we need to copy it
    moe_mlp_copy = copy_llama4_moe_mlp(moe_mlp)

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )
    muimlp = MuiParallelGateUpDownMLPMoe.replace(
        replacement_context=replacement_context,
        prev_module=moe_mlp_copy,
    )
    muimlp.finalize_init()

    # check that outputs seem fine
    input_tensor = torch.rand(size=size, device=device, dtype=dtype)

    # We don't really care about the scores, just the main output
    y, _ = moe_mlp(input_tensor)

    y_m, _ = muimlp(input_tensor)

    # the mui layer returns tensors in the same shape as the input
    # but the original layer flattens the B and T dimensions
    # so make a view so that the equality check doesn't fail as we don't
    # really care about the exact shape
    y_m = y_m.view(y.shape)

    tensors_equal(y, y_m, rtol=5 * 1e-4)


def test_basic_llama4_moe_mlp_fp16():
    device = "cuda"
    dtype = torch.float16
    hidden_size = 128
    intermediate_size = 2 * hidden_size
    size = (1, 1, hidden_size)
    execute_distributed(
        _test_basic_llama4_moe_mlp,
        size=size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
        dtype=dtype,
    )


def test_basic_llama4_moe_mlp_bf16():
    device = "cuda"
    dtype = torch.bfloat16
    hidden_size = 128
    intermediate_size = 2 * hidden_size
    size = (1, 1, hidden_size)
    execute_distributed(
        _test_basic_llama4_moe_mlp,
        size=size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
        dtype=dtype,
    )


def test_basic_llama4_moe_mlp_fp32():
    device = "cuda"
    dtype = torch.float32
    hidden_size = 128
    intermediate_size = 2 * hidden_size
    size = (1, 1, hidden_size)
    execute_distributed(
        _test_basic_llama4_moe_mlp,
        size=size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
        dtype=dtype,
    )


def test_basic_llama4_moe_mlp_batched_fp16():
    device = "cuda"
    dtype = torch.float16
    hidden_size = 128
    intermediate_size = 2 * hidden_size
    size = (3, 1, hidden_size)
    execute_distributed(
        _test_basic_llama4_moe_mlp,
        size=size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
        dtype=dtype,
    )


def test_basic_llama4_moe_mlp_batched_bf16():
    device = "cuda"
    dtype = torch.bfloat16
    hidden_size = 128
    intermediate_size = 2 * hidden_size
    size = (3, 1, hidden_size)
    execute_distributed(
        _test_basic_llama4_moe_mlp,
        size=size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
        dtype=dtype,
    )


def test_basic_llama4_moe_mlp_batched_fp32():
    device = "cuda"
    dtype = torch.float32
    hidden_size = 128
    intermediate_size = 2 * hidden_size
    size = (3, 1, hidden_size)
    execute_distributed(
        _test_basic_llama4_moe_mlp,
        size=size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
        dtype=dtype,
    )


# TODO tests with bias and no bias
# TODO tests with input norm
# TODO tests with other data types
