from typing import Optional
import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig
from muillm.replacement.replacementcontext import MuiReplacementContext
from muillm.replacement.replacements import replace_layers
from muillm.quantization.quantizationmethod import QuantizationMethod
from muillm.quantization.quantizedreplacements import quantize_layers


def _finalize_module(module: nn.Module):
    from muillm.modules.module import MuiModule

    # finalize the sub modules first
    for sub_module in module.children():
        _finalize_module(sub_module)

    # then finalize the current one
    if isinstance(module, MuiModule):
        module.finalize_init()


def init_engine(
    model: nn.Module,
    device="cuda",
    quantization_method: QuantizationMethod = None,
    tensor_parallelism: Optional[int] = 1,
    engine_config: MuiEngineConfig = None,
) -> nn.Module:

    if engine_config is None:
        engine_config = MuiEngineConfig(
            quantization_method, tensor_parallelism=tensor_parallelism
        )

    quantization_method = engine_config.quantization_method
    tensor_parallelism = engine_config.tensor_parallelism

    # replace full modules/layers first, then quantize
    replacement_context = MuiReplacementContext(
        engine_config, model=model, device=device
    )
    model = replace_layers(module=model, replacement_context=replacement_context)

    if quantization_method is not None:
        quantize_layers(model=model, engine_config=engine_config, device=device)

    # make sure everything is on the right device
    if device is not None:
        model = model.to(device=device)

    # store the config in the model
    setattr(model, "muillm_config", engine_config)

    # finalize model
    _finalize_module(model)

    return model


def load_model(
    model_id: str,
    device="cuda",
    model_dtype="auto",
    quantization_method: QuantizationMethod = None,
    tensor_parallelism: Optional[int] = 1,
    **hfkwargs,
) -> nn.Module:

    # create an engine config to know the tensor parallelism if needs to be detected
    engine_config = MuiEngineConfig(
        quantization_method, tensor_parallelism=tensor_parallelism
    )

    if engine_config.tensor_parallelism == 1:
        # we can simplify the model loading as everything has to go on the GPU
        # by loading everything on the GPU first, then replacing layers
        from transformers import AutoModelForCausalLM

        hfkwargs["torch_dtype"] = model_dtype
        hfkwargs["low_cpu_mem_usage"] = True
        model = AutoModelForCausalLM.from_pretrained(model_id, **hfkwargs).to(
            device=device
        )

        model = init_engine(
            model,
            device=device,
            engine_config=engine_config,
        )

        return model

    elif engine_config.tensor_parallelism > 1:
        hfkwargs["torch_dtype"] = model_dtype

        # If the model is big, we can load it on CPU first and then move it to the GPU
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_id, **hfkwargs)

        model = init_engine(
            model,
            device=device,
            engine_config=engine_config,
        )

        # ensure the model is moved to the right device
        model = model.to(device=device)

        # refinalize the model
        _finalize_module(model)

        return model

    else:
        raise ValueError(
            f"Invalid tensor parallelism {engine_config.tensor_parallelism}. Must be > 0."
        )
