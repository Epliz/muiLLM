import torch
import torch.nn as nn

from muillm.layers.replacements import replace_layers

def init_engine(model: nn.Module) -> nn.Module :

    replace_layers(model=model)

    return model