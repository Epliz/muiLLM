
from muillm.layers.multilinear import MuiMultiLinear
from muillm.layers.parallellinear import MuiParallelLinear
from muillm.layers.parallelmultilinear import MuiParallelMultiLinear
from muillm.layers.transformer.paralleldecoder import MuiParallelDecoderLayer
import torch
import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig
from muillm.layers.linear import MuiLinear
from muillm.layers.rmsnorm import MuiRMSNorm
from muillm.layers.gateupdownmlp import MuiGateUpDownMLP
from muillm.layers.parallelgateupdownmlp import MuiParallelGateUpDownMLP
from muillm.layers.attention.mistral.sdpaattention import MuiMistralSdpaAttention
from muillm.layers.models.mistral.model import MuiMistralModel, MuiMistralForCausalLM
from muillm.memorymanagement.gc import trigger_gc

from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralSdpaAttention, MistralMLP, MistralDecoderLayer, MistralModel, MistralForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from muillm.layers.transformer.decoder import MuiDecoderLayer




_LAYER_REPLACEMENTS = {
    nn.Linear: MuiLinear,

    # We replace the full decoder all at once to avoid issues due to replacement order
    # (e.g. replacing the MLP then the decoder)
    MistralDecoderLayer : MuiDecoderLayer,

    # replacements for full layers
    MistralModel : MuiMistralModel,
    MistralForCausalLM : MuiMistralForCausalLM,
}

_TP_LAYER_REPLACEMENTS = {
    nn.Linear: MuiParallelLinear,
    MuiLinear: MuiParallelLinear,

    MuiMultiLinear: MuiParallelMultiLinear,

    MistralMLP: MuiParallelGateUpDownMLP,
    MuiGateUpDownMLP: MuiParallelGateUpDownMLP,

    # We replace the full decoder all at once to avoid issues due to replacement order
    # (e.g. replacing the MLP then the decoder)
    MistralDecoderLayer : MuiParallelDecoderLayer,
    MuiDecoderLayer: MuiParallelDecoderLayer,

    # replacements for full layers
    MistralModel : MuiMistralModel,
    MistralForCausalLM : MuiMistralForCausalLM,
}

def _recursive_setattr(model: nn.Module, module_name: str, new_module: nn.Module):
    split_list = module_name.split('.')
    current_module = model
    for name in split_list[:-1]:
        current_module = getattr(current_module, name)
    current_module.__setattr__(split_list[-1], new_module)

def replace_layers(module: nn.Module, engine_config: MuiEngineConfig, name_prefix = "") -> nn.Module:

    module_type = type(module)

    replacements = _LAYER_REPLACEMENTS

    if engine_config.tensor_parallelism > 1:
        # we want to use tensor parallelism
        # use the correct replacements
        replacements = _TP_LAYER_REPLACEMENTS

    print(f"Replace {name_prefix} ({module_type})?")

    if module_type in replacements:
        new_module_type = replacements[module_type]
        print(f"Replacing {name_prefix} ({module_type} to {new_module_type}) ...")

        new_module = new_module_type.replace(module, engine_config=engine_config)

        # delete the previous module to save memory
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
        full_module_name = name_prefix + "." + sub_module_name if name_prefix != "" else sub_module_name

        # replace modules in this module (updated or not) recursively
        new_sub_module = replace_layers(sub_module, engine_config=engine_config, name_prefix=full_module_name)

        _recursive_setattr(module, sub_module_name, new_sub_module)

    return module
