from typing import List
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.embedding import MuiEmbedding
import torch
import torch.nn as nn

from muillm.replacement.replacementcontext import MuiReplacementContext

from .test_utils import tensors_equal


def random_embedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int = 0,
    device="cuda",
    dtype=torch.float16,
) -> nn.Embedding:
    embedding = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        device=device,
        dtype=dtype,
    )

    # We seed to have reproducible results
    torch.manual_seed(0)
    embedding.weight = nn.Parameter(torch.randn_like(embedding.weight))

    return embedding


def copy_embedding(embedding: nn.Embedding):
    device = embedding.weight.device
    dtype = embedding.weight.dtype
    new_embedding = nn.Embedding(
        num_embeddings=embedding.num_embeddings,
        embedding_dim=embedding.embedding_dim,
        padding_idx=embedding.padding_idx,
        device=device,
        dtype=dtype,
    )
    new_embedding.weight = nn.Parameter(embedding.weight.clone().detach())

    return new_embedding


def _test_basic_embedding(
    input_size=(4, 3),
    num_embeddings: int = 4,
    embedding_dim: int = 1024,
    device: str = "cpu",
    dtype=torch.float16,
):
    embedding = random_embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        padding_idx=0,
        device=device,
        dtype=dtype,
    )

    # replace destroys the passed embedding module so we need to copy it
    embedding_copy = copy_embedding(embedding)

    engine_config = MuiEngineConfig(tensor_parallelism=1)
    replacement_context = MuiReplacementContext(
        engine_config=engine_config,
        model=None,  # No model context needed for this test
        device=device,
    )
    muiembedding = MuiEmbedding.replace(
        replacement_context=replacement_context,
        prev_module=embedding_copy,
    )
    muiembedding.finalize_init()

    input_tensor = torch.randint(
        low=0, high=num_embeddings, size=input_size, device=device, dtype=torch.int64
    )

    y = embedding(input_tensor)

    y_m = muiembedding(input_tensor)

    tensors_equal(y, y_m)


def test_basic_embedding_fp32_cpu():
    device = "cpu"
    input_size = (1,)
    num_embeddings = 3
    embedding_dim = 128
    _test_basic_embedding(
        input_size=input_size,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=device,
        dtype=torch.float32,
    )


def test_basic_embedding_fp32_gpu():
    device = "cuda"
    input_size = (1,)
    num_embeddings = 11
    embedding_dim = 128
    _test_basic_embedding(
        input_size=input_size,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=device,
        dtype=torch.float32,
    )


def test_basic_embedding_fp16_gpu():
    device = "cuda"
    input_size = (1,)
    num_embeddings = 126
    embedding_dim = 128
    _test_basic_embedding(
        input_size=input_size,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=device,
        dtype=torch.float16,
    )


def test_basic_embedding_bf16_gpu():
    device = "cuda"
    input_size = (1,)
    num_embeddings = 48
    embedding_dim = 256
    _test_basic_embedding(
        input_size=input_size,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=device,
        dtype=torch.bfloat16,
    )


def test_basic_embedding_batched_fp32_cpu():
    device = "cpu"
    input_size = (4, 3)
    num_embeddings = 3
    embedding_dim = 128
    _test_basic_embedding(
        input_size=input_size,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=device,
        dtype=torch.float32,
    )


def test_basic_embedding_batched_fp32_gpu():
    device = "cuda"
    input_size = (4, 3)
    num_embeddings = 3
    embedding_dim = 128
    _test_basic_embedding(
        input_size=input_size,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=device,
        dtype=torch.float32,
    )


def test_basic_embedding_batched_fp16_gpu():
    device = "cuda"
    input_size = (4, 3)
    num_embeddings = 3
    embedding_dim = 128
    _test_basic_embedding(
        input_size=input_size,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=device,
        dtype=torch.float16,
    )


def test_basic_embedding_batched_bf16_gpu():
    device = "cuda"
    input_size = (4, 3)
    num_embeddings = 3
    embedding_dim = 128
    _test_basic_embedding(
        input_size=input_size,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=device,
        dtype=torch.bfloat16,
    )
