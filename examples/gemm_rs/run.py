from enum import IntEnum
from reference import generate_input, check_implementation
from submission import custom_kernel

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

test_shapes = [
    {
        "world_size": 8,
        "m": 64,
        "n": 2880,
        "k": 2880,
        "has_bias": True,
        "seed": 2035,
    },
    {
        "world_size": 8,
        "m": 64,
        "n": 3584,
        "k": 14336,
        "has_bias": True,
        "seed": 13,
    },
    {
        "world_size": 8,
        "m": 512,
        "n": 3584,
        "k": 14336,
        "has_bias": True,
        "seed": 4297,
    },
    {
        "world_size": 8,
        "m": 512,
        "n": 4608,
        "k": 36864,
        "has_bias": False,
        "seed": 1597,
    },
    {
        "world_size": 8,
        "m": 2048,
        "n": 4096,
        "k": 7168,
        "has_bias": False,
        "seed": 716,
    },
    {
        "world_size": 8,
        "m": 2048,
        "n": 8192,
        "k": 30720,
        "has_bias": False,
        "seed": 20201,
    },
    {
        "world_size": 8,
        "m": 4096,
        "n": 2880,
        "k": 2880,
        "has_bias": True,
        "seed": 136,
    },
    {
        "world_size": 8,
        "m": 4096,
        "n": 8192,
        "k": 2048,
        "has_bias": True,
        "seed": 138,
    },
    {
        "world_size": 8,
        "m": 8192,
        "n": 3584,
        "k": 14336,
        "has_bias": True,
        "seed": 748,
    },
    {
        "world_size": 8,
        "m": 8192,
        "n": 4608,
        "k": 36864,
        "has_bias": True,
        "seed": 4422,
    },
    {
        "world_size": 8,
        "m": 8192,
        "n": 8192,
        "k": 28672,
        "has_bias": False,
        "seed": 1536,
    },
]

benchmark_shapes = [
    {
        "world_size": 8,
        "m": 64,
        "n": 7168,
        "k": 18432,
        "has_bias": False,
        "seed": 1234,
    },
    {
        "world_size": 8,
        "m": 512,
        "n": 4096,
        "k": 12288,
        "has_bias": True,
        "seed": 663,
    },
    {
        "world_size": 8,
        "m": 2048,
        "n": 2880,
        "k": 2880,
        "has_bias": True,
        "seed": 166,
    },
    {
        "world_size": 8,
        "m": 4096,
        "n": 4096,
        "k": 4096,
        "has_bias": False,
        "seed": 1371,
    },
    {
        "world_size": 8,
        "m": 8192,
        "n": 4096,
        "k": 14336,
        "has_bias": True,
        "seed": 7168,
    },
    {
        "world_size": 8,
        "m": 8192,
        "n": 8192,
        "k": 29568,
        "has_bias": False,
        "seed": 42,
    },
]


def time_func(f):
    import time

    start_time = time.time()
    ret = f()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return ret, elapsed_time


# enumeration class for the type of run: testing or benchmarking
class RunType(IntEnum):
    TEST = 0
    BENCHMARK = 1
    PROFILE = 2


def run(rank, world_size, shape, run_type):
    data = generate_input(rank=rank, **shape)

    if run_type == RunType.TEST:
        output = custom_kernel(data)

        result, message = check_implementation(data, output)

        if not result:
            print(f"(rank {rank}) Test failed for shape: {shape}, message: {message}")
            raise ValueError("Test failed")
        else:
            print(f"(rank {rank}) Test passed for shape: {shape}")

    elif run_type == RunType.BENCHMARK:
        # warmup
        num_warmups = 10
        num_runs = 1

        def benchmark(f, data, num_runs=10):
            for _ in range(num_runs):
                ret = f(data)
            torch.cuda.synchronize()
            return ret

        for _ in range(num_warmups):
            _ = custom_kernel(data)

        # benchmark
        _, elapsed_time = time_func(
            lambda: benchmark(custom_kernel, data, num_runs=num_runs)
        )

        elapsed_time_usec = (elapsed_time * 1e6) / num_runs

        if rank == 0:
            print(f"shape: {shape}, avg time: {elapsed_time_usec:.6f} usec")
    else:
        raise ValueError("Invalid run type")


def run_profile(rank, world_size, shapes, run_type):
    for shape in shapes:
        data = generate_input(rank=rank, **shape)

        if run_type == RunType.PROFILE:
            import torch.autograd.profiler as profiler

            num_warmups = 10
            num_runs = 10

            # warmup
            for _ in range(num_warmups):
                _ = custom_kernel(data)

            from torch.profiler import profile, ProfilerActivity

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                for _ in range(num_runs):
                    output = custom_kernel(data)
                torch.cuda.synchronize()

            prof.export_chrome_trace(
                f"trace_gemmrs_rank{rank}_m{shape['m']}_n{shape['n']}_k{shape['k']}_bias{shape['has_bias']}_world_size{shape['world_size']}.json"
            )

            torch.distributed.barrier()

        else:
            raise ValueError("Invalid run type")


def init_process(rank, size, shape, run_type, fn, backend="nccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["LOCAL_SIZE"] = str(size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # if rank == 0:
    #     os.environ["HSA_TOOLS_LIB"] = "/opt/rocm/lib/librocm-debug-agent.so.2"
    #     os.environ["HSA_ENABLE_DEBUG"] = "1"

    local_size = torch.cuda.device_count()
    print(f"(rank {rank}) local_size = {local_size}")

    # set the current device to the GPU we need
    torch.cuda.set_device(rank)

    try:
        dist.init_process_group(backend, rank=rank, world_size=size, device_id=rank)
        fn(rank, size, shape, run_type)
        torch.cuda.synchronize()
        print("completed", flush=True)
    except Exception as e:
        print(f"(rank {rank}) Caught exception: {e}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":

    import sys

    # Default run type
    run_type = RunType.PROFILE
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "test":
            run_type = RunType.TEST
        elif arg == "benchmark":
            run_type = RunType.BENCHMARK
        elif arg == "profile":
            run_type = RunType.PROFILE
        else:
            print(f"Unknown run type '{sys.argv[1]}', defaulting to PROFILE.")

    shapes = test_shapes if (run_type == RunType.TEST) else benchmark_shapes

    # get the number of GPUs
    size = torch.cuda.device_count()

    # patch the world size into each shape
    for shape in shapes:
        shape["world_size"] = size

    print(f"{size} GPUs available.")

    if run_type == RunType.PROFILE:
        # Spawn one subprocess per GPU
        processes = []
        mp.set_start_method("spawn")
        for rank in range(size):
            p = mp.Process(
                target=init_process, args=(rank, size, shapes, run_type, run_profile)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        mp_context = mp.get_context("spawn")
        with mp_context.Pool(size) as pool:
            for shape in shapes:
                print(f"shape: {shape}")

                # Spawn one subprocess per GPU
                rets = []
                for rank in range(size):
                    p = pool.apply_async(
                        func=init_process, args=(rank, size, shape, run_type, run)
                    )
                    rets.append(p)

                rets = [el.get(60) for el in rets]
