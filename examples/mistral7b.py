# Example showing how to use muiLLM's fp16 support on the Mistral 7b model

import os

# Run this example on a single GPU
os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["ROCM_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

# this example requires the Mistral 7b Instruct v0.2 model
# Provided that you have a HF token to access the Mistral models, you can download it with 
# huggingface-cli download --token <your_hf_token> mistralai/Mistral-7B-Instruct-v0.2 --local-dir Mistral-7B-Instruct-v0.2 --revision 41b61a33a2483885c981aa79e0df6b32407ed873
# (the specific revision is required as Mistral changed the repo to use their own tokenizer past that revision)

# either set this environment variable before running the example, or adapt the path
model_id = os.getenv("MISTRAL_7B_PATH", "/storage/models/Mistral-7B-Instruct-v0.2/")

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

# Have a look at the original speed (~50 tokens/s generation on MI300x)
text, time = time_func(lambda: generate(model, "Hello my name is", 50))
text, time = time_func(lambda: generate(model, "Hello my name is", 50))
text, time = time_func(lambda: generate(model, "Hello my name is", 50))
print("[Original] Completion: ", text)
print("[Original] Time: ", time)

# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(lambda: time_func(lambda: generate(model, "Hello my name is", 50)), trace_path="trace_orig.json")

# Use the muiLLM replacements layers
from muillm.engine import init_engine
model = init_engine(model)

print("Optimized models: ", model)

# Have a look at the speed (~140 token/s generation on MI300x)
text, time = time_func(lambda: generate(model, "Hello my name is", 50))
text, time = time_func(lambda: generate(model, "Hello my name is", 50))
text, time = time_func(lambda: generate(model, "Hello my name is", 50))
print("[Optimized] Completion: ", text)
print("[Optimized] Time: ", time)

# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(lambda: time_func(lambda: generate(model, "Hello my name is", 50)), trace_path="trace_muillm.json")