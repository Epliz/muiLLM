from typing import List, Optional


class QuantizationMethod:
    """
    Base class for all supported quantization methods
    """
    def __init__(self, modules: Optional[List[str]] = None):
        self.modules = list(modules) if modules is not None else None

class Int8WeightOnlyQuantizationMethod(QuantizationMethod):
    """
    Quantization method quantizing floating point number model weights
    to 8 bit integers.
    """
    def __init__(
            self,
            group_size: int = 128,
            f: float = 1.0,
            modules: Optional[List[str]] = None,
    ):
        super().__init__(modules)
        self.group_size = group_size
        self.f = f
