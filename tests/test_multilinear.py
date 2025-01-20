from typing import List
from muillm.engineconfig import MuiEngineConfig
import torch
import torch.nn as nn

from muillm.modules.multilinear import MuiMultiLinear

def tensors_equal(t1, t2):
    assert t1.shape == t2.shape
    assert torch.allclose(t1, t2, rtol=1e-04)

def random_linear(in_features: int, out_features: int) -> nn.Linear:
    return nn.Linear(in_features= in_features, out_features = out_features)

def random_linears(in_features: int, out_features: List[int]) -> List[nn.Linear]:
    return [random_linear(in_features=in_features, out_features=out_feat) for out_feat in out_features]

def test_basic_linears():
    in_features = 4096
    linears = random_linears(in_features=in_features, out_features=[1024, 2048, 4096])

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    multilinear = MuiMultiLinear.replace(prev_modules=linears, engine_config=engine_config)

    input_tensor = torch.rand(size=(4, in_features))

    q = linears[0](input_tensor)
    k = linears[1](input_tensor)
    v = linears[2](input_tensor)

    q_m, k_m, v_m = multilinear(input_tensor)

    tensors_equal(q, q_m)
    tensors_equal(k, k_m)
    tensors_equal(v, v_m)

# TODO tests with bias and no bias
# TODO tests with input norm
# TODO tests with other data types

def test_replace_back():
    in_features = 4096
    linears = random_linears(in_features=in_features, out_features=[1024, 2048, 4096])

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    multilinear = MuiMultiLinear.replace(prev_modules=linears, engine_config=engine_config)

    replaced_back_linears, _ = multilinear.replace_back()

    input_tensor = torch.rand(size=(4, in_features))

    q = linears[0](input_tensor)
    k = linears[1](input_tensor)
    v = linears[2](input_tensor)

    q_r = replaced_back_linears[0](input_tensor)
    k_r = replaced_back_linears[1](input_tensor)
    v_r = replaced_back_linears[2](input_tensor)

    tensors_equal(q, q_r)
    tensors_equal(k, k_r)
    tensors_equal(v, v_r)