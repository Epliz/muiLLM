from typing import Optional
import torch
import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig
from muillm.layers.replacements import replace_layers
from muillm.modules.wrapping import wrap_model
from muillm.quantization.quantizationmethod import QuantizationMethod
from muillm.quantization.quantizedreplacements import quantize_layers

def init_engine(model: nn.Module, quantization_method: QuantizationMethod = None, tensor_parallelism: Optional[int] = 1) -> nn.Module :

    engine_config = MuiEngineConfig(quantization_method, tensor_parallelism=tensor_parallelism)

    # replace full modules/layers first, then quantize
    model = replace_layers(module=model, engine_config=engine_config)

    if quantization_method is not None:
        quantize_layers(model=model, engine_config=engine_config)

    # wrap model e.g. to replace generation function for transformer models
    model = wrap_model(model, engine_config=engine_config)

    # store the config in the model
    setattr(model, "muillm_config", engine_config)

    return model
