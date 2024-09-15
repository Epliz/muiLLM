
from typing import Optional
from muillm.commmunication.communicator import MuiCommunicator, TorchCommunicator

from muillm.quantization.quantizationmethod import QuantizationMethod
from muillm.synchronization.synchronizer import Synchronizer

class MuiEngineConfig:
    def __init__(
            self,
            quantization_method: QuantizationMethod,
            tensor_parallelism: Optional[int] = None
            ):
        self.synchronizer = Synchronizer()
        self.quantization_method = quantization_method

        try:
            torch_communicator = TorchCommunicator()

            if torch_communicator.is_multi_node():
                # only the torch communicator supports multi-node at the moment
                self.communicator = torch_communicator
            else:
                # but we can use the mui communicator for single-machine
                self.communicator = MuiCommunicator()
        except:
            # if torch dist is not available, we will get an exception
            self.communicator = None

        if tensor_parallelism is None:
            world_size = self.communicator.world_size if self.communicator is not None else 1
            # for now the world size if the tensor parallelism level, but in the future,
            # maybe we would support mixes between pipelining and tensor parallelism
            self.tensor_parallelism = world_size
