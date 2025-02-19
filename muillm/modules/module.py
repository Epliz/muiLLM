import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig

class MuiModule(nn.Module):
    def __init__(self, engine_config: MuiEngineConfig, **kargs) -> None:
        nn.Module.__init__(self, **kargs)

        self.engine_config = engine_config

    def finalize_init(self) -> None:
        # Method called at the end of replacements of all layers
        # can for example build the C++ module counterparts
        pass
