import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig

class MuiModule(nn.Module):
    def __init__(self, engine_config: MuiEngineConfig, **kargs) -> None:
        nn.Module.__init__(self, **kargs)

        self.engine_config = engine_config
