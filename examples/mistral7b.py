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

# 5 tokens prompt
short_prompt = "Hello my name is"

# 467 tokens prompt
long_prompt = """Hello my name is Ashley and I am a 22 year old student at the University of California, Los Angeles (UCLA). I am currently studying business economics with a minor in psychology. I am a third year student and I am planning to graduate in 2018.
I am originally from Los Angeles, California and I have always been passionate about business and entrepreneurship. I have had various internships and volunteer positions throughout my college career, including interning at a startup in downtown LA and volunteering at a non-profit organization that provides financial literacy to underprivileged youth.
I am excited to be a part of the program and I am looking forward to learning from the other participants and gaining valuable experience in the field of business and entrepreneurship.
Hi Ashley, I am also from LA, I went to Caltech for undergrad and now I am at UCLA for my MBA. I am also interested in business and entrepreneurship and I have also had various internships and volunteer positions throughout my college career. It's great to hear that you are interested in business and entrepreneurship, what are your goals and aspirations after graduation?
Hi Ashley, I'm also from LA, I went to UCLA for undergrad and now I'm at USC for my MBA. I'm also interested in business and entrepreneurship and I've had various internships and volunteer positions throughout my college career. It's great to hear that you're interested in business and entrepreneurship, what are your goals and aspirations after graduation?
Hi Ashley, I'm a fellow Bruin! I'm a junior majoring in economics and I'm really interested in finance. I've been thinking about pursuing a career in investment banking or private equity. Have you thought about what you want to do after graduation? Do you have any specific goals or industries in mind?
Hi Ashley, I'm also a student at UCLA, I'm a senior majoring in business economics and I'm really interested in entrepreneurship. I've been working on a startup idea and I'm looking for potential partners or investors. Have you thought about starting your own business or working for a startup? I'd love to hear more about your interests and goals. Hi Ashley, I'm a fellow student at UCLA and I'm really interested in learning more about your experiences and goals. Can you tell me a bit more about your background and what you're hoping to"""

prompt = long_prompt

tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding="longest")
print("tokenized prompts: ", tokenized_prompt["input_ids"].shape)

num_input_tokens = tokenized_prompt["input_ids"].shape[1]
num_output_tokens = 256
num_total_tokens = num_input_tokens + num_output_tokens

# Have a look at the original speed
if num_total_tokens < 100:
    text, time = time_func(lambda: generate(model, prompt, num_output_tokens))
    text, time = time_func(lambda: generate(model, prompt, num_output_tokens))
    text, time = time_func(lambda: generate(model, prompt, num_output_tokens))
    print("[Original] Completion: ", text)
    print("[Original] Time: ", time)
    print("tot toks/s: ", num_total_tokens / time)

# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(lambda: time_func(lambda: generate(model, prompt, num_output_tokens)), trace_path="trace_orig.json")

# Use the muiLLM replacements layers
from muillm.engine import init_engine
model = init_engine(model)

print("Optimized models: ", model)

# Have a look at the speed (~140 token/s generation on MI300x)
text, time = time_func(lambda: generate(model, prompt, num_output_tokens))
text, time = time_func(lambda: generate(model, prompt, num_output_tokens))
text, time = time_func(lambda: generate(model, prompt, num_output_tokens))
print("[Optimized] Completion: ", text)
print("[Optimized] Time: ", time)
print("tot toks/s: ", num_total_tokens / time)

# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(lambda: time_func(lambda: generate(model, prompt, num_output_tokens)), trace_path="trace_muillm.json")