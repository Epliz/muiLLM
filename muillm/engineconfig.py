from typing import Optional
from muillm.comms.communicator import MuiCommunicator
from muillm.quantization.quantizationmethod import QuantizationMethod
from muillm.synchronization.synchronizer import Synchronizer

import torch

import muillm_ext


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
            raise ValueError(
                f"tensor_parallelism {tensor_parallelism} is bigger than number of available devices: {device_count}"
            )

        self.tensor_parallelism = tensor_parallelism

        # initialize engine in C++ side
        self.cpp_engine = muillm_ext.muillm_engine_init()

        # only creates comms if necessary because we want to use tensor parallelism
        self.comms = None
        if self.tensor_parallelism > 1:
            self.comms = MuiCommunicator(cpp_engine=self.cpp_engine)

            if self.tensor_parallelism != self.comms.world_size:
                raise ValueError(
                    f"tensor_parallelism should match world_size but got {self.tensor_parallelism} and {self.comms.world_size}"
                )

            self.devices = [
                torch.device(f"cuda:{d}") for d in range(self.tensor_parallelism)
            ]
            self.streams = [
                torch.cuda.Stream(self.devices[i])
                for i in range(self.tensor_parallelism)
            ]

            # set default device correctly (comm init might have changed it)
            for s in self.streams:
                torch.cuda.set_stream(s)

            torch.cuda.set_device(self.devices[self.comms.local_rank])

    def rank(self) -> int:
        if self.comms is None:
            return 0
        return self.comms.rank

    def is_rank0(self) -> bool:
        return self.rank() == 0
