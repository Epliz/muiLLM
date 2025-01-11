# Example showing how to use muiLLM's tensor parallelism and fp16 support on the Meta LLama 3.1 8b model

import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

# this example requires the LLama 3.1 8B Instruct model
# Provided that you have a HF token to access the Mistral models, you can download it with 
# huggingface-cli download --token <your_token> meta-llama/Llama-3.1-8B-Instruct --local-dir Llama-3.1-8B-Instruct

# either set this environment variable before running the example, or adapt the path
model_id = os.getenv("LLAMA3_8B_PATH", "/storage/models/Llama-3.1-8B-Instruct/")

## Load the original model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id,padding_side="left")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# we load the original model in fp16 precision
model: nn.Module = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device="cuda", dtype=torch.float16)

print("Model : ", model)

from typing import List, Union


def generate(model, prompt:Union[str, List[str]], max_new_tokens=20) -> Union[str, List[str]]:
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

num_output_tokens = 50

# Have a look at the original speed (~50 tokens/s generation on MI300x)
text, time = time_func(lambda: generate(model, "Hello my name is", num_output_tokens))
text, time = time_func(lambda: generate(model, "Hello my name is", num_output_tokens))
text, time = time_func(lambda: generate(model, "Hello my name is", num_output_tokens))
print("[Original] Completion: ", text)
print("[Original] Time: ", time)

# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(lambda: time_func(lambda: generate(model, "Hello my name is", num_output_tokens)), trace_path="trace_orig.json")

# Use the muiLLM replacements layers
from muillm.engine import init_engine
# tensor_parallelism=None indicates to use all GPUs
model = init_engine(model, tensor_parallelism=None)

print("Optimized models: ", model)

# Have a look at the speed (~140 token/s generation on MI300x)
text, time = time_func(lambda: generate(model, "Hello my name is", num_output_tokens))
text, time = time_func(lambda: generate(model, "Hello my name is", num_output_tokens))
text, time = time_func(lambda: generate(model, "Hello my name is", num_output_tokens))
print("[Optimized] Completion: ", text)
print("[Optimized] Time: ", time)

# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(lambda: time_func(lambda: generate(model, "Hello my name is", num_output_tokens)), trace_path="trace_muillm.json")