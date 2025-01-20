from typing import List
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.parallelmultilinear import MuiParallelMultiLinear
import torch
import torch.nn as nn

from muillm.modules.multilinear import MuiMultiLinear

def tensors_equal(t1: torch.Tensor, t2: torch.Tensor):
    device = t1.device
    t2 = t2.to(device=device)

    rtol=1e-04
    atol=1e-06
    assert t1.shape == t2.shape
    if torch.allclose(t1, t2, atol=atol, rtol=rtol) == False:

        max_diff = torch.max(torch.abs(t2 - t1))
        print("max_diff ", max_diff.item())

        assert torch.allclose(t1, t2, atol=atol, rtol=rtol)



def random_linear(in_features: int, out_features: int) -> nn.Linear:
    return nn.Linear(in_features= in_features, out_features = out_features)

def random_linears(in_features: int, out_features: List[int]) -> List[nn.Linear]:
    return [random_linear(in_features=in_features, out_features=out_feat) for out_feat in out_features]


def test_replace_linears_row_sharding():
    in_features = 16
    linears = random_linears(in_features=in_features, out_features=[4, 8, 16])

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    parallel_multilinear = MuiParallelMultiLinear.replace(prev_modules=linears, engine_config=engine_config, sharding_dim=0)

    input_tensor = torch.rand(size=(2, in_features))

    q = linears[0](input_tensor)
    k = linears[1](input_tensor)
    v = linears[2](input_tensor)

    q_m, k_m, v_m = parallel_multilinear(input_tensor)

    tensors_equal(q, q_m)
    tensors_equal(k, k_m)
    tensors_equal(v, v_m)

    # second inference to make sure it works several time in a row
    q_m, k_m, v_m = parallel_multilinear(input_tensor)

    tensors_equal(q, q_m)
    tensors_equal(k, k_m)
    tensors_equal(v, v_m)

# TODO tests with bias and no bias
# TODO tests with input norm
# TODO tests with other data types

def test_replace_multilinear_row_sharding():
    in_features = 16
    linears = random_linears(in_features=in_features, out_features=[4, 8, 16])

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    multilinear = MuiMultiLinear.replace(prev_modules=linears, engine_config=engine_config)
    parallel_multilinear = MuiParallelMultiLinear.replace(prev_modules=multilinear, engine_config=engine_config, sharding_dim=0)

    input_tensor = torch.rand(size=(2, in_features))

    q = linears[0](input_tensor)
    k = linears[1](input_tensor)
    v = linears[2](input_tensor)

    q_m, k_m, v_m = parallel_multilinear(input_tensor)

    tensors_equal(q, q_m)
    tensors_equal(k, k_m)
    tensors_equal(v, v_m)

def test_replace_linears_col_sharding():
    in_features = 16
    linears = random_linears(in_features=in_features, out_features=[4, 8, 16])

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    parallel_multilinear = MuiParallelMultiLinear.replace(prev_modules=linears, engine_config=engine_config, sharding_dim=1)

    input_tensor = torch.rand(size=(2, in_features))

    q = linears[0](input_tensor)
    k = linears[1](input_tensor)
    v = linears[2](input_tensor)

    q_m, k_m, v_m = parallel_multilinear(input_tensor)

    tensors_equal(q, q_m)
    tensors_equal(k, k_m)
    tensors_equal(v, v_m)

# TODO tests with input norm
# TODO tests with other data types

def test_replace_multilinear_col_sharding():
    in_features = 16
    linears = random_linears(in_features=in_features, out_features=[4, 8, 16])

    engine_config = MuiEngineConfig(tensor_parallelism=None)
    multilinear = MuiMultiLinear.replace(prev_modules=linears, engine_config=engine_config)
    parallel_multilinear = MuiParallelMultiLinear.replace(prev_modules=multilinear, engine_config=engine_config, sharding_dim=1)

    input_tensor = torch.rand(size=(2, in_features))

    q = linears[0](input_tensor)
    k = linears[1](input_tensor)
    v = linears[2](input_tensor)

    q_m, k_m, v_m = parallel_multilinear(input_tensor)

    tensors_equal(q, q_m)
    tensors_equal(k, k_m)
    tensors_equal(v, v_m)