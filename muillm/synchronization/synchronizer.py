from typing import Union
import torch

import muillm_ext

class Synchronizer:
    """
    Provides functionalities to do fast CPU/GPU synchronization and copies
    """
    def __init__(self):

        self.sync = muillm_ext.muillm_sync_init()


    def item(self, tensor: torch.Tensor) -> Union[int, float, bool]:
        """
        Replacement for torch.item() that avoids some HIP bugs
        """
        if tensor.dtype == torch.bool:
            return muillm_ext.muillm_item_bool(self.sync, tensor)
        elif tensor.dtype == torch.float16:
            return muillm_ext.muillm_item_f16(self.sync, tensor)
        elif tensor.dtype == torch.float32:
            return muillm_ext.muillm_item_f32(self.sync, tensor)
        else:
            raise ValueError(f"Unnsupport dtype {tensor.dtype}")
        
    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == torch.device("cpu"):
            return tensor

        return muillm_ext.muillm_to_cpu(self.sync, tensor)