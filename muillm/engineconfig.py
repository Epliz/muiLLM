from typing import Optional
from muillm.quantization.quantizationmethod import QuantizationMethod
from muillm.synchronization.synchronizer import Synchronizer

import torch

class MuiEngineConfig:
    def __init__(
            self,
            quantization_method: Optional[QuantizationMethod] = None,
            tensor_parallelism: Optional[int] = 1,
            ):
        self.synchronizer = Synchronizer()
        self.quantization_method = quantization_method

        if tensor_parallelism is None:
            # None means use all GPUs
            tensor_parallelism = torch.cuda.device_count()

        self.tensor_parallelism = tensor_parallelism
