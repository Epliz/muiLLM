import torch
import torch.nn as nn

from muillm.layers.replacements import replace_layers
from muillm.modules.wrapping import wrap_model
from muillm.quantization.quantizationmethod import QuantizationMethod
from muillm.quantization.quantizedreplacements import quantize_layers

def init_engine(model: nn.Module, quantization_method: QuantizationMethod = None) -> nn.Module :

    # replace full modules/layers first, then quantize
    replace_layers(model=model)

    if quantization_method is not None:
        quantize_layers(model=model, quantization_method=quantization_method)

    # wrap model e.g. to replace generation function for transformer models
    return wrap_model(model)