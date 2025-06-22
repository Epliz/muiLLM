from muillm.engineconfig import MuiEngineConfig
import torch
import torch.nn as nn


from transformers.models.llama4.modeling_llama4 import Llama4TextL2Norm

from muillm.modules.norm.l2norm import MuiL2Norm
from .test_utils import tensors_equal


def random_llama4_l2norm(
    eps=1e-5, dtype=torch.float16, device="cpu"
) -> Llama4TextL2Norm:
    norm = Llama4TextL2Norm(eps=eps)

    norm = norm.to(dtype=dtype, device=device)

    return norm


def copy_llama4_l2norm(norm: Llama4TextL2Norm) -> Llama4TextL2Norm:
    new_norm = Llama4TextL2Norm(eps=norm.eps)

    return new_norm


def _test_basic_llama4_l2norm(dtype: torch.dtype, device: str):
    hidden_size = 256
    norm = random_llama4_l2norm(dtype=dtype, device=device)

    # replace destroys the passed linear module so we need to copy it
    norm_copy = copy_llama4_l2norm(norm)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    muinorm = MuiL2Norm.replace(
        prev_module=norm_copy,
        engine_config=engine_config,
        device=device,
    )

    input_tensor = torch.rand(size=(4, hidden_size), dtype=dtype, device=device)

    y = norm(input_tensor)

    y_m = muinorm(input_tensor)

    tensors_equal(y, y_m)


def test_basic_llama4_l2norm_fp32_cpu():
    _test_basic_llama4_l2norm(dtype=torch.float32, device="cpu")


def test_basic_llama4_l2norm_fp32_gpu():
    _test_basic_llama4_l2norm(dtype=torch.float32, device="cuda")


def test_basic_llama4_l2norm_fp16_gpu():
    _test_basic_llama4_l2norm(dtype=torch.float16, device="cuda")


def test_basic_llama4_l2norm_bf16_gpu():
    _test_basic_llama4_l2norm(dtype=torch.bfloat16, device="cuda")
