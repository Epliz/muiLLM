# test that we get the same output with MuiGemma3DecoderLayer as with Gemma3DecoderLayer
import os
import torch
from muillm.modules.decoder.gemma3decoder import (
    MuiGemma3DecoderLayer,
    Gemma3DecoderLayer,
)
from muillm.modules.module import MuiModule
from muillm.replacement.replacementcontext import MuiReplacementContext
from muillm.engineconfig import MuiEngineConfig

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from tests.test_utils import tensors_equal


def create_random_gemma3_decoder_layer(
    hidden_size: int = 1024,
    intermediate_size: int = 4096,
    device: str = "cpu",
    dtype=torch.float16,
) -> Gemma3DecoderLayer:
    config = Gemma3TextConfig(
        hidden_size=hidden_size,
        num_attention_heads=16,
        intermediate_size=intermediate_size,
        sliding_window=128,
    )

    decoder = Gemma3DecoderLayer(config=config, layer_idx=0).to(
        device=device, dtype=dtype
    )

    # Initialize all the parameters randomly
    for param in decoder.parameters(recurse=True):
        if param.dim() == 2:
            # use Xavier uniform for 2d parameters
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.normal_(param)

    return decoder


def copy_gemma3_decoder_layer(gemma3_layer: Gemma3DecoderLayer) -> Gemma3DecoderLayer:
    config = gemma3_layer.config
    layer_idx = gemma3_layer.layer_idx

    device = gemma3_layer.mlp.gate_proj.weight.device
    dtype = gemma3_layer.mlp.gate_proj.weight.dtype

    new_layer = Gemma3DecoderLayer(config=config, layer_idx=layer_idx).to(
        device=device, dtype=dtype
    )

    # Copy parameters from the original layer
    orig_params_dict = {name: param for name, param in gemma3_layer.named_parameters()}
    new_params_dict = {name: param for name, param in new_layer.named_parameters()}

    for name, param in orig_params_dict.items():
        if name in new_params_dict:
            new_params_dict[name].data.copy_(param.data)

    return new_layer


def decode_step_by_step(
    decoder,
    hidden_states,
    position_embeddings_global,
    position_embeddings_local,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    sliding_window_mask=None,
    **kwargs,
):

    residual = hidden_states

    input_layernorm_output = decoder.input_layernorm(hidden_states)
    hidden_states = input_layernorm_output

    # apply global RoPE to non-sliding layer only
    if decoder.self_attn.is_sliding:
        position_embeddings = position_embeddings_local
    else:
        position_embeddings = position_embeddings_global

    attn_output, self_attn_weights = decoder.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = attn_output

    post_attention_layernorm_output = decoder.post_attention_layernorm(hidden_states)
    hidden_states = post_attention_layernorm_output

    hidden_states = residual + hidden_states

    residual = hidden_states

    if hasattr(decoder, "pre_feedforward_layernorm"):
        pre_feedforward_layernorm_output = decoder.pre_feedforward_layernorm(
            hidden_states
        )
        hidden_states = pre_feedforward_layernorm_output
    else:
        pre_feedforward_layernorm_output = None

    mlp_output = decoder.mlp(hidden_states)
    hidden_states = mlp_output

    post_feedforward_layernorm_output = decoder.post_feedforward_layernorm(
        hidden_states
    )
    hidden_states = post_feedforward_layernorm_output

    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    return (
        input_layernorm_output,
        attn_output,
        post_attention_layernorm_output,
        pre_feedforward_layernorm_output,
        mlp_output,
        post_feedforward_layernorm_output,
        hidden_states,
    )


def _test_gemma3_decoder_layer(
    batch_size: int = 1,
    hidden_size: int = 128,
    intermediate_size: int = 4096,
    device: str = "cpu",
    dtype=torch.float16,
):
    seq_length = 5

    # Create a Gemma3DecoderLayer instance
    decoder = create_random_gemma3_decoder_layer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
        dtype=dtype,
    )

    head_dim = getattr(
        decoder.config,
        "head_dim",
        decoder.config.hidden_size // decoder.config.num_attention_heads,
    )

    # replace destroys the passed linear module so we need to copy it
    decoder_copy = copy_gemma3_decoder_layer(decoder)

    # Create a replacement context
    engine_config = MuiEngineConfig(tensor_parallelism=1)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )

    # Replace with MuiGemma3DecoderLayer
    mui_decoder = MuiGemma3DecoderLayer.replace(
        replacement_context=replacement_context,
        prev_module=decoder_copy,
    )

    # finalize all mui modules
    for module in mui_decoder.modules():
        if isinstance(module, MuiModule):
            module.finalize_init()

    # Test the output of both layers with a random input
    input_tensor = torch.rand(
        size=(batch_size, seq_length, hidden_size), device=device, dtype=dtype
    )
    position_embeddings_global_cos = torch.rand(
        size=(batch_size, seq_length, head_dim), device=device, dtype=dtype
    )
    position_embeddings_global_sin = torch.rand(
        size=(batch_size, seq_length, head_dim), device=device, dtype=dtype
    )
    position_embeddings_global = (
        position_embeddings_global_cos,
        position_embeddings_global_sin,
    )

    position_embeddings_local_cos = torch.rand(
        size=(batch_size, seq_length, head_dim), device=device, dtype=dtype
    )
    position_embeddings_local_sin = torch.rand(
        size=(batch_size, seq_length, head_dim), device=device, dtype=dtype
    )
    position_embeddings_local = (
        position_embeddings_local_cos,
        position_embeddings_local_sin,
    )

    # (output_gemma3,) = decoder(
    #     hidden_states=input_tensor,
    #     position_embeddings_global=position_embeddings_global,
    #     position_embeddings_local=position_embeddings_local,
    # )
    # (output_mui_gemma3,) = mui_decoder(
    #     hidden_states=input_tensor,
    #     position_embeddings_global=position_embeddings_global,
    #     position_embeddings_local=position_embeddings_local,
    # )

    # # Check if the outputs are equal
    # tensors_equal(output_gemma3, output_mui_gemma3)

    # run step by step and compare all tensors
    orig_outputs = decode_step_by_step(
        decoder=decoder,
        hidden_states=input_tensor,
        position_embeddings_global=position_embeddings_global,
        position_embeddings_local=position_embeddings_local,
    )

    mui_outputs = decode_step_by_step(
        decoder=mui_decoder,
        hidden_states=input_tensor,
        position_embeddings_global=position_embeddings_global,
        position_embeddings_local=position_embeddings_local,
    )

    # Check if the outputs are equal
    for i, (orig, mui) in enumerate(zip(orig_outputs, mui_outputs)):
        print(f"Comparing output {i}:")
        if orig is None or mui is None:
            continue
        tensors_equal(orig, mui)


def test_gemma3_decoder_layer_bs1_cpu_fp32():
    _test_gemma3_decoder_layer(
        batch_size=1,
        hidden_size=1024,
        intermediate_size=4096,
        device="cpu",
        dtype=torch.float32,
    )


def test_gemma3_decoder_layer_bs1_gpu_fp32():
    _test_gemma3_decoder_layer(
        batch_size=1,
        hidden_size=2048,
        intermediate_size=8192,
        device="cuda",
        dtype=torch.float32,
    )


def test_gemma3_decoder_layer_bs1_gpu_fp16():
    _test_gemma3_decoder_layer(
        batch_size=1,
        hidden_size=2048,
        intermediate_size=8192,
        device="cuda",
        dtype=torch.float16,
    )


def test_gemma3_decoder_layer_bs1_gpu_bf16():
    _test_gemma3_decoder_layer(
        batch_size=1,
        hidden_size=512,
        intermediate_size=2048,
        device="cuda",
        dtype=torch.bfloat16,
    )


def test_gemma3_decoder_layer_bs4_cpu_fp32():
    _test_gemma3_decoder_layer(
        batch_size=4,
        hidden_size=1024,
        intermediate_size=4096,
        device="cpu",
        dtype=torch.float32,
    )


def test_gemma3_decoder_layer_bs4_gpu_fp32():
    _test_gemma3_decoder_layer(
        batch_size=4,
        hidden_size=2048,
        intermediate_size=8192,
        device="cuda",
        dtype=torch.float32,
    )


def test_gemma3_decoder_layer_bs4_gpu_fp16():
    _test_gemma3_decoder_layer(
        batch_size=4,
        hidden_size=2048,
        intermediate_size=8192,
        device="cuda",
        dtype=torch.float16,
    )


def test_gemma3_decoder_layer_bs4_gpu_bf16():
    _test_gemma3_decoder_layer(
        batch_size=4,
        hidden_size=512,
        intermediate_size=2048,
        device="cuda",
        dtype=torch.bfloat16,
    )
