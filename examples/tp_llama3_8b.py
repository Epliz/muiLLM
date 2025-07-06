"""run.py:"""

#!/usr/bin/env python

# Example showing how to use muiLLM's tensor parallelism support on the Llama 3.1 8b model

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
import torch
import torch.nn as nn

from typing import List, Union


def generate(
    model, tokenizer, prompt: Union[str, List[str]], max_new_tokens=20
) -> Union[str, List[str]]:
    single_prompt = isinstance(prompt, str)
    if single_prompt:
        prompts = [prompt]
    else:
        prompts = prompt

    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding="longest").to(
            device="cuda"
        )

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    texts = [text[len(prompts[i]) :] for i, text in enumerate(texts)]
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


def profile_func(f, trace_path="trace.json"):
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        ret = f()
    prof.export_chrome_trace(trace_path)
    return ret


def run(rank, size):

    # this example requires the LLama 3.1 8B Instruct model
    # Provided that you have a HF token to access the Llama models, you can download it with
    # huggingface-cli download --token <your_token> meta-llama/Llama-3.1-8B-Instruct --local-dir Llama-3.1-8B-Instruct

    # either set this environment variable before running the example, or adapt the path
    model_id = os.getenv("LLAMA3_8B_PATH", "/storage/models/Llama-3.1-8B-Instruct/")

    ## Load the original model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    # we load the original model in fp16 precision
    model: nn.Module = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, tp_plan="auto"
    ).to(device="cuda", dtype=torch.bfloat16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    if rank == 0:
        print("Model : ", model)

    # 5 tokens prompt
    prompt = "Hello my name is"

    tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest")
    if rank == 0:
        print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

    num_input_tokens = tokenized_prompts["input_ids"].shape[1]
    batch_size = tokenized_prompts["input_ids"].shape[0]
    num_output_tokens = 256
    num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

    # Use the muiLLM replacements layers
    from muillm.engine import init_engine

    # use auto-detected tensor parallelism level by setting to None
    model = init_engine(model, tensor_parallelism=None)

    if rank == 0:
        print("Optimized models: ", model)

    tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest")

    if rank == 0:
        print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

    num_input_tokens = tokenized_prompts["input_ids"].shape[1]
    batch_size = tokenized_prompts["input_ids"].shape[0]
    num_output_tokens = 256
    num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

    # Have a look at the speed
    text, time = time_func(lambda: generate(model, tokenizer, prompt, 10))
    text, time = time_func(
        lambda: generate(model, tokenizer, prompt, num_output_tokens)
    )

    # check how many tokens were actually generated
    tokenized_outputs = tokenizer(text, return_tensors="pt", padding="longest")
    num_total_tokens = (
        num_input_tokens + tokenized_outputs["input_ids"].shape[1]
    ) * batch_size

    if rank == 0:
        print("[Optimized] Completion: ", text)
        print("[Optimized] Time: ", time)
        print(
            f"tot toks/s:  {num_total_tokens / time} (batch size {batch_size}, prompt len {num_input_tokens})"
        )

    # Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
    text, time = profile_func(
        lambda: time_func(
            lambda: generate(model, tokenizer, prompt, num_output_tokens)
        ),
        trace_path=f"trace_llama_muillm_tp{size}_unbatched_rank{rank}.json",
    )


def init_process(rank, size, fn, backend="nccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["LOCAL_SIZE"] = str(size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

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
