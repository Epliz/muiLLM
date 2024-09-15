from muillm.engineconfig import MuiEngineConfig
import torch.nn as nn

class MuiModule(nn.Module):
    def __init__(
            self,
            engine_config: MuiEngineConfig):
        nn.Module.__init__(self)

        self.engine_config = engine_config
        