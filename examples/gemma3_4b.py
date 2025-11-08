# Example showing how to use muiLLM's fp16 support on the Google Gemma 3 4b model

import os

# Run this example on a single GPU
os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["ROCM_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

from transformers import (
    GenerationMixin,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
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


stream_output = os.getenv("STREAM_OUTPUT", "1") == "1"

# this example requires the Google Gemma 3 4b Instruct model
# Provided that you have a HF token to access the Gemma models, you can download it with
# huggingface-cli download --token <your_token> google/gemma-3-4b-it --local-dir google-gemma-3-4b-it

# either set this environment variable before running the example, or adapt the path
model_id = os.getenv("GEMMA3_4B_PATH", "/storage/models/google-gemma-3-4b-it/")

## Load the original model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

processor = AutoProcessor.from_pretrained(model_id)

# we load the original model in fp16 precision
model: nn.Module = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cuda", torch_dtype=torch.bfloat16
)

from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_prompt=True) if stream_output else None

print("Model : ", model)

#
prompt = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Write a blogpost about how Croatia is the dream destination for summer vacations.",
            },
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

print("tokenized prompts shape: ", tokenized_prompts["input_ids"].shape)

num_input_tokens = tokenized_prompts["input_ids"].shape[1]
batch_size = tokenized_prompts["input_ids"].shape[0]
num_output_tokens = 256
num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

# Have a look at the original speed
if num_total_tokens < 1024:
    text, time = time_func(
        lambda: generate(model, processor, prompt, 10, disable_compile=True)
    )
    text, time = time_func(
        lambda: generate(
            model, processor, prompt, num_output_tokens, disable_compile=True
        )
    )

    print("[Original] Completion: ", text)
    print("[Original] Time: ", time)

    # check how many tokens were actually generated
    tokenized_outputs = tokenizer(text, return_tensors="pt", padding="longest")
    num_total_tokens = (
        num_input_tokens + tokenized_outputs["input_ids"].shape[1]
    ) * batch_size

    print(
        f"tot toks/s:  {num_total_tokens / time} (batch size {batch_size}, prompt len {num_input_tokens}, output len {num_output_tokens})"
    )


# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(
    lambda: time_func(
        lambda: generate(model, processor, prompt, 50, disable_compile=True)
    ),
    trace_path="trace_gemma3_orig_unbatched.json",
)

# Use the muiLLM replacements layers
from muillm.engine import init_engine

model = init_engine(model, tensor_parallelism=1)

print("Optimized models: ", model)

tokenized_prompts = processor.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

num_input_tokens = tokenized_prompts["input_ids"].shape[1]
batch_size = tokenized_prompts["input_ids"].shape[0]

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

if streamer is None:
    print("[Optimized] Completion: ", text)
print("[Optimized] Time: ", time)

# check how many tokens were actually generated
tokenized_outputs = tokenizer(text, return_tensors="pt", padding="longest")
num_output_tokens = tokenized_outputs["input_ids"].shape[1]
num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

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
    trace_path="trace_gemma3_muillm_unbatched.json",
)
