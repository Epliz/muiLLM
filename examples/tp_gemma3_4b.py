"""run.py:"""

#!/usr/bin/env python

# Example showing how to use muiLLM's tensor parallelism support on the Gemma 3 4B model

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import (
    AutoModelForCausalLM,
    GenerationMixin,
    AutoProcessor,
    AutoTokenizer,
)
import torch
import torch.nn as nn

from typing import List, Union


def generate(
    model: GenerationMixin,
    processor,
    prompt: Union[str, List[str]],
    max_new_tokens=20,
    streamer=None,
    **kwargs,
) -> Union[str, List[str]]:
    if streamer is not None:
        print("--- start streaming ---")

    single_prompt = isinstance(prompt, str)
    if single_prompt:
        prompts = [prompt]
    else:
        prompts = prompt

    with torch.no_grad():
        inputs = processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device="cuda")
        input_len = inputs["input_ids"].shape[-1]

        outputs = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            **kwargs,
        )
        outputs = [outputs[i][input_len:] for i in range(len(outputs))]

    if streamer is not None:
        print("--- end streaming ---")

    texts = [
        processor.decode(outputs[i], skip_special_tokens=False)
        for i in range(len(outputs))
    ]

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

    stream_output = os.getenv("STREAM_OUTPUT", "1") == "1"

    # this example requires the Gemma 3 4B model
    # Provided that you have a HF token to access the Gemma models, you can download it with
    # huggingface-cli download --token <your_token> google/gemma-3-4b-it --local-dir google-gemma-3-4b-it

    # either set this environment variable before running the example, or adapt the path
    model_id = os.getenv("GEMMA3_4B_PATH", "/storage/models/google-gemma-3-4b-it/")

    ## Load the original model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    processor = AutoProcessor.from_pretrained(model_id)

    model: nn.Module = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16
    )

    from transformers import TextStreamer

    streamer = (
        TextStreamer(tokenizer, skip_prompt=True)
        if (stream_output and rank == 0)
        else None
    )

    if rank == 0:
        print("Model : ", model)

    # 5 tokens prompt
    prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is your name?"},
            ],
        },
    ]

    tokenized_prompts = processor.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    if rank == 0:
        print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

    num_input_tokens = tokenized_prompts["input_ids"].shape[1]
    batch_size = tokenized_prompts["input_ids"].shape[0]
    num_output_tokens = 64
    num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

    # Have a look at the original speed
    if num_total_tokens < 100:
        text, time = time_func(
            lambda: generate(model, processor, prompt, 10, disable_compile=True)
        )
        text, time = time_func(
            lambda: generate(
                model, processor, prompt, num_output_tokens, disable_compile=True
            )
        )

        # check how many tokens were actually generated
        tokenized_outputs = tokenizer(text, return_tensors="pt", padding="longest")
        num_output_tokens = tokenized_outputs["input_ids"].shape[1]
        num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

        print("[Original] Completion: ", text)
        print("[Original] Time: ", time)
        print(
            f"tot toks/s:  {num_total_tokens / time} (batch size {batch_size}, prompt len {num_input_tokens}, output len {num_output_tokens})"
        )

    # Use the muiLLM replacements layers
    from muillm.engine import init_engine

    # use auto-detected tensor parallelism level by setting to None
    model = init_engine(model, tensor_parallelism=None)

    if rank == 0:
        print("Optimized models: ", model)

    tokenized_prompts = processor.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    if rank == 0:
        print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

    num_input_tokens = tokenized_prompts["input_ids"].shape[1]
    batch_size = tokenized_prompts["input_ids"].shape[0]
    num_output_tokens = 64
    num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

    # Have a look at the speed
    text, time = time_func(
        lambda: generate(model, processor, prompt, 10, disable_compile=True)
    )
    text, time = time_func(
        lambda: generate(
            model,
            processor,
            prompt,
            num_output_tokens,
            streamer=streamer,
            disable_compile=True,
        )
    )

    # check how many tokens were actually generated
    tokenized_outputs = tokenizer(text, return_tensors="pt", padding="longest")
    num_output_tokens = tokenized_outputs["input_ids"].shape[1]
    num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

    if rank == 0:
        if streamer is None:
            print("[Optimized] Completion: ", text)
        print("[Optimized] Time: ", time)
        print(
            f"tot toks/s:  {num_total_tokens / time} (batch size {batch_size}, prompt len {num_input_tokens}, output len {num_output_tokens})"
        )

    # Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
    text, time = profile_func(
        lambda: time_func(
            lambda: generate(
                model, processor, prompt, num_output_tokens, disable_compile=True
            )
        ),
        trace_path=f"trace_gemma3_muillm_tp{size}_unbatched_rank{rank}.json",
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
