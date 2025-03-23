"""run.py:"""
#!/usr/bin/env python

# Example showing how to use muiLLM's tensor parallelism support on the Llama 3.1 8b model

import os

from muillm.engineconfig import MuiEngineConfig
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
import torch
import torch.nn as nn

from typing import List, Union

def generate(model, tokenizer, prompt:Union[str, List[str]], max_new_tokens=20) -> Union[str, List[str]]:
    single_prompt = isinstance(prompt, str)
    if single_prompt:
        prompts = [prompt]
    else:
        prompts = prompt

    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding="longest").to(device="cuda")

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    texts = [text[len(prompts[i]):] for i, text in enumerate(texts)]
    if single_prompt:
        return texts[0]
    else:
        return texts

def time_func(f):
    import time
    start_time = time.time()
    ret = f()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return ret, elapsed_time

def profile_func(f, trace_path= "trace.json"):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        ret = f()
    prof.export_chrome_trace(trace_path)
    return ret

def repeat(n, func):
    for _ in range(n):
        func()

def run(rank, world_size):

    sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]

    r = 4
    # Pytorch all-reduce
    for s in sizes:
        tensor = torch.zeros(s, dtype=torch.float16, device="cuda")

        # Warmup
        repeat(r, lambda: dist.all_reduce(tensor))

        # profile
        profile_func(lambda: repeat(r, lambda: dist.all_reduce(tensor)), trace_path=f"trace_rccl_all_reduce_tp{world_size}_size{s}_rank{rank}.json")

        torch.cuda.synchronize()
    

    # tensor_parallelism=None indicates to use all GPUs
    engine_config = MuiEngineConfig(tensor_parallelism=None)
    comms = engine_config.comms

    # muillm all-reduce
    for s in sizes:
        tensor = torch.zeros(s, dtype=torch.float16, device="cuda")

        # Warmup
        repeat(r, lambda: comms.all_reduce_sum(tensor))
    
        # profile
        profile_func(lambda: repeat(r, lambda: comms.all_reduce_sum(tensor)), trace_path=f"trace_mui_all_reduce_tp{world_size}_size{s}_rank{rank}.json")

        torch.cuda.synchronize()

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['LOCAL_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

    local_size = torch.cuda.device_count()
    print(f"(rank {rank}) local_size = {local_size}")

    # set the current device to the GPU we need
    torch.cuda.set_device(rank)

    dist.init_process_group(backend, rank=rank, world_size=size)

    fn(rank, size)


if __name__ == "__main__":
    # get the number of GPUs
    size = torch.cuda.device_count()

    print(f"{size} GPUs available.")

    # Spawn one subprocess per GPU
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
