
import os

os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["ROCM_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

model_id = "/storage/models/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id,padding_side="left")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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


from muillm.engine import init_engine
model = init_engine(model)

print("Optimized models: ", model)

text, time = time_func(lambda: generate(model, "Hello my name is", 50))
text, time = time_func(lambda: generate(model, "Hello my name is", 50))
text, time = time_func(lambda: generate(model, "Hello my name is", 50))
print("[Optimized] Completion: ", text)
print("[Optimized] Time: ", time)
text, time = profile_func(lambda: time_func(lambda: generate(model, "Hello my name is", 50)), trace_path="trace_muillm.json")
