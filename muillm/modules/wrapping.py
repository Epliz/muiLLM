import torch
import torch.nn as nn

# TODO: dynamic import
from transformers.generation.utils import GenerationMixin

from muillm.modules.wrappedtransformers import _wrap_transformers_model

def wrap_model(model: nn.Module):

    if isinstance(model, GenerationMixin):
        return _wrap_transformers_model(model)

    return model