from typing import Optional
from muillm.comms.communicator import Communicator, TorchCommunicator
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

        device_count = torch.cuda.device_count()
        if tensor_parallelism is None:
            # None means use all GPUs
            tensor_parallelism = device_count


        if tensor_parallelism > device_count:
            raise ValueError(f"tensor_parallelism {tensor_parallelism} is bigger than number of available devices: {device_count}")

        self.tensor_parallelism = tensor_parallelism

        # only creates comms if necessary because we want to use tensor parallelism
        self.comms = None
        if self.tensor_parallelism > 1:
            self.comms = TorchCommunicator()

            if self.tensor_parallelism != self.comms.world_size:
                raise ValueError(f"tensor_parallelism should match world_size but got {self.tensor_parallelism} and {self.comms.world_size}")
