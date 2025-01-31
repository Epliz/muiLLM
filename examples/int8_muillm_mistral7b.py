# Example showing how to use muiLLM's int8 RTN support on the Mistral 7b model

import os

from muillm.quantization.quantizationmethod import Int8WeightOnlyQuantizationMethod

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

# 5 tokens prompt
prompt = "Hello my name is"

tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest")
print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

num_input_tokens = tokenized_prompts["input_ids"].shape[1]
batch_size = tokenized_prompts["input_ids"].shape[0]
num_output_tokens = 256
num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

# Have a look at the original speed
if num_total_tokens < 100:
    text, time = time_func(lambda: generate(model, prompt, 10))
    text, time = time_func(lambda: generate(model, prompt, num_output_tokens))
    print("[Original] Completion: ", text)
    print("[Original] Time: ", time)
    print(f"tot toks/s:  {num_total_tokens / time} (batch size {batch_size}, prompt len {num_input_tokens})")

# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(lambda: time_func(lambda: generate(model, prompt, 50)), trace_path="trace_mistral_orig_unbatched.json")

# Use the muiLLM replacements layers with int8 round-to-nearest (RTN) quantization
from muillm.engine import init_engine
model = init_engine(model, quantization_method=Int8WeightOnlyQuantizationMethod(group_size=32, modules=["qkv_proj", "o_proj", "gate_proj", "up_proj","down_proj", "mlp"]))

print("Optimized models: ", model)

tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest")
print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

num_input_tokens = tokenized_prompts["input_ids"].shape[1]
batch_size = tokenized_prompts["input_ids"].shape[0]
num_output_tokens = 256
num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

# Have a look at the speed
text, time = time_func(lambda: generate(model, prompt, 10))
text, time = time_func(lambda: generate(model, prompt, num_output_tokens))
print("[Optimized] Completion: ", text)
print("[Optimized] Time: ", time)
print(f"tot toks/s:  {num_total_tokens / time} (batch size {batch_size}, prompt len {num_input_tokens})")

# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(lambda: time_func(lambda: generate(model, prompt, 50)), trace_path="trace_mistral_muillm_int8_unbatched.json")