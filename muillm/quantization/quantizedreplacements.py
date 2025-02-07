from typing import List
import torch
import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.quantized.int8linear import MuiInt8Linear
from muillm.modules.quantized.int8gateupdownmlp import MuiInt8GateUpDownMLP
from muillm.modules.gateupdownmlp import MuiGateUpDownMLP
from muillm.quantization.quantizationmethod import Int8WeightOnlyQuantizationMethod, QuantizationMethod

from muillm.modules.linear import MuiLinear

_INT8_LAYER_REPLACEMENTS = {
    nn.Linear: MuiInt8Linear,
    MuiLinear: MuiInt8Linear,
    MuiGateUpDownMLP: MuiInt8GateUpDownMLP,
}

def _recursive_setattr(model: nn.Module, module_name: str, new_module: nn.Module):
    split_list = module_name.split('.')
    current_module = model
    for name in split_list[:-1]:
        current_module = getattr(current_module, name)
    current_module.__setattr__(split_list[-1], new_module)

def _module_name_matches(module_name: str, modules: List[str]):
    for m in modules:
        if m in module_name:
            return True
    return False

def quantize_layers(model: nn.Module, engine_config: MuiEngineConfig):
    quantization_method = engine_config.quantization_method
    if not isinstance(quantization_method, Int8WeightOnlyQuantizationMethod):
        raise ValueError(f"The quantization method {quantization_method} is not supported")
    
    replacements = _INT8_LAYER_REPLACEMENTS

    for module_name, module in model.named_modules():
        module_type = type(module)
        if module_type in _INT8_LAYER_REPLACEMENTS:

            # only replace the modules that match one of the desired patterns
            if quantization_method.modules is not None:
                if not _module_name_matches(module_name, quantization_method.modules):
                    continue
    
            if engine_config.is_rank0():
                print(f"Quantizing {module_name}...")

            new_module_type = replacements[module_type]

            new_module = new_module_type.replace(module, engine_config=engine_config)

            _recursive_setattr(model, module_name, new_module)

            # delete the previous module to save memory
            del module

            # trigger GC to save memory
            trigger_gc()