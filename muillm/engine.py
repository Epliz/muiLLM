import torch
import torch.nn as nn

from muillm.layers.replacements import replace_layers
from muillm.modules.wrapping import wrap_model

def init_engine(model: nn.Module) -> nn.Module :

    replace_layers(model=model)

    # wrap model e.g. to replace generation function for transformer models
    return wrap_model(model)