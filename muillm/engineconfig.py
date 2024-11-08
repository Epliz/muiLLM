from muillm.quantization.quantizationmethod import QuantizationMethod
from muillm.synchronization.synchronizer import Synchronizer

class MuiEngineConfig:
    def __init__(
            self,
            quantization_method: QuantizationMethod
            ):
        self.synchronizer = Synchronizer()
        self.quantization_method = quantization_method
