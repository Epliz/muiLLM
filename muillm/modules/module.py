import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig


class MuiModule(nn.Module):
    def __init__(self, engine_config: MuiEngineConfig = None, **kargs) -> None:
        if engine_config is None:
            # some classes don't support inheritance properly
            # that's our way to detect when they call super().__init__()
            # in which case we do nothing and wait for the right init call
            return None

        nn.Module.__init__(self, **kargs)

        self.engine_config = engine_config

    def __del__(self):
        self.finalize_deinit()

        if hasattr(nn.Module, "__del__"):
            # call the parent destructor if it exists
            nn.Module.__del__(self)

    def finalize_init(self) -> None:
        # Method called at the end of replacements of all layers
        # can for example build the C++ module counterparts
        pass

    def finalize_deinit(self):
        # Method called during destruction of the module
        pass
