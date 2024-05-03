import torch
import torch.nn as nn

from muillm.layers.linear import MuiLinear
from muillm.memorymanagement.gc import trigger_gc


_LAYER_REPLACEMENTS = {
    nn.Linear: MuiLinear,
}

def _recursive_setattr(model: nn.Module, module_name: str, new_module: nn.Module):
    split_list = module_name.split('.')
    current_module = model
    for name in split_list[:-1]:
        current_module = getattr(current_module, name)
    current_module.__setattr__(split_list[-1], new_module)

def replace_layers(model: nn.Module, name_prefix = ""):

    # only replace the immediate children and not all children
    # as we might update the structure quite a lot through our replacements
    # and original children might not have counterparts anymore after replacements
    # (e.g. q, k, v projs will not exist anymore in the attention after replacement to our attention classes)
    for module_name, module in model.named_children():
        module_type = type(module)

        full_module_name = name_prefix + "." + module_name if name_prefix != "" else module_name

        if module_type in _LAYER_REPLACEMENTS:
            print(f"Replacing {full_module_name} ({module_type}) ...")
            new_module_type = _LAYER_REPLACEMENTS[module_type]

            new_module = new_module_type.replace(module)

            _recursive_setattr(model, module_name, new_module)

            # delete the previous module to save memory
            del module

            # trigger GC to save memory
            trigger_gc()

            # point to the new module so that we recurse on it
            module = new_module

        # replace modules in this module (updated or not) recursively
        replace_layers(module, name_prefix=full_module_name)