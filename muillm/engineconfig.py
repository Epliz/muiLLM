from typing import Optional
from muillm.comms.communicator import Communicator
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

        self.devices = [torch.device(f"cuda:{d}") for d in range(self.tensor_parallelism)]
        self.streams = [torch.cuda.Stream(self.devices[i]) for i in range(self.tensor_parallelism)]

        for s in self.streams:
            torch.cuda.set_stream(s)

        self.comms = Communicator(tensor_parallelism=tensor_parallelism, devices=self.devices)

        # set default device correctly (comm init might have changed it)
        torch.cuda.set_device(self.devices[0])
