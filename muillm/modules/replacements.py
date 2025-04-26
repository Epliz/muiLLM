from muillm.modules.attention.llama4attention import MuiLlama4TextAttention
from muillm.modules.attention.rotaryembedding import MuiRotaryEmbedding
from muillm.modules.decoder.paralleldecoder import MuiParallelDecoderLayer
from muillm.modules.models.llama.model import MuiLlamaForCausalLM, MuiLlamaModel
from muillm.modules.moe.gateupdownmlpmoe import MuiGateUpDownMLPMoe
from muillm.modules.moe.parallelgateupdownmlpmoe import MuiParallelGateUpDownMLPMoe
from muillm.modules.multilinear import MuiMultiLinear
from muillm.modules.parallelgateupdownmlp import MuiParallelGateUpDownMLP
from muillm.modules.parallellinear import MuiParallelLinear
from muillm.modules.parallelmultilinear import MuiParallelMultiLinear
import torch
import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.linear import MuiLinear
from muillm.modules.rmsnorm import MuiRMSNorm
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
from muillm.modules.models.mistral.model import MuiMistralModel, MuiMistralForCausalLM
from muillm.memorymanagement.gc import trigger_gc

from transformers.models.mistral.modeling_mistral import (
    MistralRotaryEmbedding,
    MistralRMSNorm,
    MistralMLP,
    MistralDecoderLayer,
    MistralModel,
    MistralForCausalLM,
)
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.models.llama4.modeling_llama4 import (
    Llama4TextRMSNorm,
    Llama4TextAttention,
    Llama4TextMLP,
    Llama4TextMoe,
)

from muillm.modules.decoder.decoder import MuiDecoderLayer


_LAYER_REPLACEMENTS = {
    # Linear
    nn.Linear: MuiLinear,
    # MLPs
    MistralMLP: MuiGateUpDownMLP,
    LlamaMLP: MuiGateUpDownMLP,
    Llama4TextMLP: MuiGateUpDownMLP,
    # MoE MLPS
    # Llama4TextMoe: MuiGateUpDownMLPMoe,
    # RMS Norm layers
    MistralRMSNorm: MuiRMSNorm,
    LlamaRMSNorm: MuiRMSNorm,
    # Llama4TextRMSNorm: MuiRMSNorm,
    # Rotary embeddings
    MistralRotaryEmbedding: MuiRotaryEmbedding,
    LlamaRotaryEmbedding: MuiRotaryEmbedding,
    # attentions
    # Llama4TextAttention: MuiLlama4TextAttention,
    # Decoders
    # We replace the full decoder all at once to avoid issues due to replacement order
    # (e.g. if replacing the MLP not as part of the decoder, we don't get the norm layer)
    MistralDecoderLayer: MuiDecoderLayer,
    LlamaDecoderLayer: MuiDecoderLayer,
    # replacements for full models
    MistralModel: MuiMistralModel,
    LlamaModel: MuiLlamaModel,
    MistralForCausalLM: MuiMistralForCausalLM,
    LlamaForCausalLM: MuiLlamaForCausalLM,
}

_TP_LAYER_REPLACEMENTS = {
    # Linear
    MuiMultiLinear: MuiParallelMultiLinear,
    nn.Linear: MuiParallelLinear,
    MuiLinear: MuiParallelLinear,
    # MLPs
    MistralMLP: MuiParallelGateUpDownMLP,
    LlamaMLP: MuiParallelGateUpDownMLP,
    # Llama4TextMLP: MuiParallelGateUpDownMLP,
    MuiGateUpDownMLP: MuiParallelGateUpDownMLP,
    # MoE MLPS
    Llama4TextMoe: MuiParallelGateUpDownMLPMoe,
    # RMS Norm layers
    MistralRMSNorm: MuiRMSNorm,
    LlamaRMSNorm: MuiRMSNorm,
    # Llama4TextRMSNorm: MuiRMSNorm,
    # Rotrary embeddings
    MistralRotaryEmbedding: MuiRotaryEmbedding,
    LlamaRotaryEmbedding: MuiRotaryEmbedding,
    # attentions
    # Llama4TextAttention: MuiLlama4TextAttention,
    # We replace the full decoder all at once to avoid issues due to replacement order
    # (e.g. if replacing the MLP not as part of the decoder, we don't get the norm layer)
    MistralDecoderLayer: MuiParallelDecoderLayer,
    LlamaDecoderLayer: MuiParallelDecoderLayer,
    # replacements for full models
    MistralModel: MuiMistralModel,
    MistralForCausalLM: MuiMistralForCausalLM,
    LlamaModel: MuiLlamaModel,
    LlamaForCausalLM: MuiLlamaForCausalLM,
}


def _recursive_setattr(model: nn.Module, module_name: str, new_module: nn.Module):
    split_list = module_name.split(".")
    current_module = model
    for name in split_list[:-1]:
        current_module = getattr(current_module, name)
    current_module.__setattr__(split_list[-1], new_module)


def replace_layers(
    module: nn.Module,
    engine_config: MuiEngineConfig,
    device="cuda",
    name_prefix="",
) -> nn.Module:
    module_type = type(module)

    replacements = _LAYER_REPLACEMENTS

    if engine_config.tensor_parallelism > 1:
        # we want to use tensor parallelism
        # use the correct replacements
        replacements = _TP_LAYER_REPLACEMENTS

    if engine_config.is_rank0():
        print(f"Replace {name_prefix} ({module_type})?")

    if module_type in replacements:
        new_module_type = replacements[module_type]

        if engine_config.is_rank0():
            print(f"Replacing {name_prefix} ({module_type} to {new_module_type}) ...")

        new_module = new_module_type.replace(
            module, engine_config=engine_config, device=device
        )

        # delete the previous module to save memory
        if new_module != module:
            del module

        # trigger GC to save memory
        trigger_gc()

        # point to the new module so that we recurse on it
        module = new_module

    # only replace the immediate children and not all children
    # as we might update the structure quite a lot through our replacements
    # and original children might not have counterparts anymore after replacements
    # (e.g. q, k, v projs will not exist anymore in the attention after replacement to our attention classes)
    for sub_module_name, sub_module in module.named_children():
        full_module_name = (
            name_prefix + "." + sub_module_name
            if name_prefix != ""
            else sub_module_name
        )

        # replace modules in this module (updated or not) recursively
        new_sub_module = replace_layers(
            sub_module,
            engine_config=engine_config,
            device=device,
            name_prefix=full_module_name,
        )

        _recursive_setattr(module, sub_module_name, new_sub_module)

    return module
