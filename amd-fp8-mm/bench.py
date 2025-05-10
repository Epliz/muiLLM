from typing import Tuple
import torch

# a, b, a_scale, b_scale, c
input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# c
output_t = torch.Tensor

block_shape = (128, 128)

# Tested shapes for correctness:
# array with all the tested shapes as tuples (without the seed)
# (executed when doing /leaderboard submit test)
tested_shapes = [
    # M, N, K
    (128, 64, 64),
    (7168, 64, 1536),
    (1536, 64, 3072),
    (7168, 64, 576),
    (256, 96, 7168),
    (2048, 96, 7168),
    (7168, 96, 4608),
    (2304, 128, 7168),
    (7168, 128, 512),
    (512, 512, 4096),
    (7168, 512, 1536),
]

# shapes for benchmarking
benchmarking_shapes = [
    # M, N, K
    (1024, 1536, 7168),  # probably requires split K
    (1024, 4608, 7168),
    (6144, 1536, 7168),
    (6144, 4608, 7168),
    (1024, 7168, 256),
    (6144, 7168, 256),
]


def generate_input(m: int, n: int, k: int, seed: int) -> input_t:
    """
    Generate random input and weights for Blockwise W8A8 Matmul scaled to FP32.

    Returns:
        Tuple of (
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k],
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k],
            a_scale: torch.Tensor[float32] of shape [m, k // 128],
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128],
            c: torch.Tensor[bfloat16] of shape [m, n]
        )
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    # Generate random inputs with FP8 quantization
    a = (torch.randn((k, m), dtype=torch.bfloat16, device="cuda", generator=gen)).to(
        torch.float8_e4m3fnuz
    )
    b = (torch.randn((k, n), dtype=torch.bfloat16, device="cuda", generator=gen)).to(
        torch.float8_e4m3fnuz
    )

    # Generate scaling factors with FP32
    a_scale = torch.randn(
        [scale_k, m], dtype=torch.float32, device="cuda", generator=gen
    )
    b_scale = torch.randn(
        [scale_k, scale_n], dtype=torch.float32, device="cuda", generator=gen
    )

    c = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    return (a.T, b.T, a_scale.T, b_scale.T, c)


def ref_kernel(data: input_t) -> output_t:
    """
    Highly inefficient torch reference implementation of FP8 GEMM.
    You can use this as a reference / starting template for your implementation.
    """
    # c: [m, n] is pre-allocated memory to help remove allocation overhead.
    a, b, a_scale, b_scale, c = data

    # a is M x K in column-major order, we convert here for simplicity.
    a = a.contiguous()
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()

    # constants
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]
    block_shape_n = 128
    block_shape_k = 128
    scale_n = b_scale.shape[0]
    scale_k = b_scale.shape[1]

    # Apply blockwise scaling to input 'a'
    a_scale = a_scale.unsqueeze(-1).repeat(
        1, 1, block_shape_k
    )  # Shape: [m, scale_k, block_shape_k]
    a_scale = a_scale.reshape(m, scale_k * block_shape_k)
    a_scale = a_scale[:, :k]

    # Dequantize 'a', in your implementation you should do this at the end.
    a = a.to(a_scale.dtype) * a_scale

    # Apply blockwise scaling to input 'b'
    b_scale = (
        b_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k)
        .permute(0, 2, 1, 3)  # Reorder dimensions: [scale_n, blk_n, scale_k, blk_k]
        .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    )
    b_scale = b_scale[:n, :k]

    # Dequantize 'b', in your implementation you should do this at the end.
    b = b.to(b_scale.dtype) * b_scale

    # Compute FP8 GEMM and write to 'c'.
    c[...] = (a @ b.T).to(torch.bfloat16)
    return c


def custom_kernel(data: input_t) -> output_t:
    """
    Highly inefficient torch reference implementation of FP8 GEMM.
    You can use this as a reference / starting template for your implementation.
    """
    # c: [m, n] is pre-allocated memory to help remove allocation overhead.
    a, b, a_scale, b_scale, c = data

    # a is M x K in column-major order, we convert here for simplicity.
    a = a.contiguous()
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()

    # constants
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]
    block_shape_n = 128
    block_shape_k = 128
    scale_n = b_scale.shape[0]
    scale_k = b_scale.shape[1]

    # Apply blockwise scaling to input 'a'
    a_scale = a_scale.unsqueeze(-1).repeat(
        1, 1, block_shape_k
    )  # Shape: [m, scale_k, block_shape_k]
    a_scale = a_scale.reshape(m, scale_k * block_shape_k)
    a_scale = a_scale[:, :k]

    # Dequantize 'a', in your implementation you should do this at the end.
    a = a.to(a_scale.dtype) * a_scale

    # Apply blockwise scaling to input 'b'
    b_scale = (
        b_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k)
        .permute(0, 2, 1, 3)  # Reorder dimensions: [scale_n, blk_n, scale_k, blk_k]
        .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    )
    b_scale = b_scale[:n, :k]

    # Dequantize 'b', in your implementation you should do this at the end.
    b = b.to(b_scale.dtype) * b_scale

    # Compute FP8 GEMM and write to 'c'.
    c[...] = (a @ b.T).to(torch.bfloat16)
    return c


def make_match_reference(custom_kernel, rtol=1e-02, atol=1e-03):
    """
    Compare the output of the custom kernel with the reference kernel.
    """

    def check_implementation(data: input_t) -> bool:
        ref_output = ref_kernel(data)
        custom_output = custom_kernel(data)
        return torch.allclose(ref_output, custom_output, rtol=rtol, atol=atol)

    for shape in tested_shapes:
        m, n, k = shape
        seed = 0
        data = generate_input(m, n, k, seed)

        if not check_implementation(data):
            print(f"Mismatch for shape {shape} with seed {seed}")
            return False


def benchmark_kernel(
    custom_kernel,
    shape: Tuple[int, int, int],
    num_warmups: int = 10,
    num_runs: int = 100,
):
    """
    Benchmark the custom kernel for a given shape.
    """
    m, n, k = shape
    seed = 0
    data = generate_input(m, n, k, seed)

    # Warm up
    for _ in range(num_warmups):
        custom_kernel(data)

    torch.cuda.synchronize()

    # Benchmark
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(num_runs):
        custom_kernel(data)
    end_time.record()

    torch.cuda.synchronize()
    elapsed_time = 1000.0 * start_time.elapsed_time(end_time) / num_runs
    print(f"Elapsed time for shape {shape}: {elapsed_time:.3f} us")


if __name__ == "__main__":
    # Testing for correctness
    make_match_reference(custom_kernel)
    print("Custom kernel matches reference kernel for all tested shapes.")

    # Benchmarking for speed
    for shape in benchmarking_shapes:
        benchmark_kernel(custom_kernel, shape)
        print(f"Benchmarking completed for shape {shape}.")
