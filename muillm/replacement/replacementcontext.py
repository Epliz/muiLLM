# dataclass containing the MuiEngineConfig, current model being replaced etc
from muillm.engineconfig import MuiEngineConfig

import torch.nn as nn


class MuiReplacementContext:
    """
    Context for the replacement process.
    Contains the engine configuration and the current model being replaced.
    """

    def __init__(
        self, engine_config: MuiEngineConfig, model: nn.Module, device: str = "cuda"
    ):
        self.engine_config = engine_config
        self.model = model
        self.device = device

    def _get_module_name(self, module: nn.Module) -> str:
        """
        Get the name of the module in the context of the model.
        """
        for name, mod in self.model.named_modules():
            if mod is module:
                return name

        raise ValueError(f"Module {module} not found in the model.")

    def to_local_module(self, module: nn.Module) -> nn.Module:
        """
        Convert a module to a local module if it is a distributed module.
        """
        if self.model is None:
            # if the model is None, we probably don't need to convert
            return module

        from muillm.hftensorparallelism.hftensorparallelism import _to_local_module

        module_name = self._get_module_name(module)

        return _to_local_module(self.model, module, name=module_name)
