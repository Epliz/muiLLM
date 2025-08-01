# Example showing how to use muiLLM's fp16 support on the Meta LLama 4 model

import os

# Run this example on a single GPU
os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["ROCM_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import torch.nn as nn

# either set this environment variable before running the example, or adapt the path
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# TODO: modify to use the processor and chat template
## Load the original model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

# load the original configuration
config = AutoConfig.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# make the model smaller
config.text_config.num_hidden_layers = 4
config.vision_config.num_hidden_layers = 4

print("Config : ", config)

model: nn.Module = AutoModel.from_config(config, torch_dtype=torch.bfloat16)

print("Model : ", model)

model = model.to(device="cuda", dtype=torch.bfloat16)

# print all the parameters and buffer shapes, and dtypes
for name, param in model.named_parameters():
    print(f"param {name}: {param.shape} {param.dtype}")
for name, buffer in model.named_buffers():
    print(f"buffer {name}: {buffer.shape} {buffer.dtype}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))


from typing import List, Union


def generate(
    model, prompt: Union[str, List[str]], max_new_tokens=20
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
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True
        )

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


# 5 tokens prompt
prompt = "Hello my name is"

tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest")
print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

num_input_tokens = tokenized_prompts["input_ids"].shape[1]
batch_size = tokenized_prompts["input_ids"].shape[0]
num_output_tokens = 4
num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

# Have a look at the original speed
if num_total_tokens < 100:
    text, time = time_func(lambda: generate(model, prompt, 10))
    text, time = time_func(lambda: generate(model, prompt, num_output_tokens))

    # check how many tokens were actually generated
    tokenized_outputs = tokenizer(text, return_tensors="pt", padding="longest")
    num_total_tokens = (
        num_input_tokens + tokenized_outputs["input_ids"].shape[1]
    ) * batch_size

    print("[Original] Completion: ", text)
    print("[Original] Time: ", time)
    print(
        f"tot toks/s:  {num_total_tokens / time} (batch size {batch_size}, prompt len {num_input_tokens})"
    )


# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(
    lambda: time_func(lambda: generate(model, prompt, 50)),
    trace_path="trace_llama4_sandbox_orig_unbatched.json",
)

# Use the muiLLM replacements layers
from muillm.engine import init_engine

model = init_engine(model)

print("Optimized models: ", model)

tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest")
print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

num_input_tokens = tokenized_prompts["input_ids"].shape[1]
batch_size = tokenized_prompts["input_ids"].shape[0]
num_output_tokens = 4
num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

# Have a look at the speed
text, time = time_func(lambda: generate(model, prompt, 10))
text, time = time_func(lambda: generate(model, prompt, num_output_tokens))


# check how many tokens were actually generated
tokenized_outputs = tokenizer(text, return_tensors="pt", padding="longest")
num_total_tokens = (
    num_input_tokens + tokenized_outputs["input_ids"].shape[1]
) * batch_size

print("[Optimized] Completion: ", text)
print("[Optimized] Time: ", time)
print(
    f"tot toks/s:  {num_total_tokens / time} (batch size {batch_size}, prompt len {num_input_tokens})"
)

# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(
    lambda: time_func(lambda: generate(model, prompt, num_output_tokens)),
    trace_path="trace_llama4_sandbox_muillm_unbatched.json",
)
