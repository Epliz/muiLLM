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
short_prompt0 = "Hello my name is Ashley"
short_prompt1 = "Hello my name is Bernard"

# 467 tokens prompt
long_prompt0 = """Hello my name is Ashley, I am a proud Mum to five beautiful children, I am currently living in South Africa but was born and bred in Zimbabwe. I had a wonderful childhood with both my parents and they always encouraged me to follow my dreams .I grew up in a community of where women were encouraged to go to school, university and pursue career. My mum worked for the government and my dad worked for a mining company. i became a teacher after finishing my studies in South Africa. i am now a head of mistress in a wonderful school and my passion is to inspire children to become the best that they can be and to have all opportunities open that they deserve.i am a published author and i love to write on various themes ranging from childrens literature to political and social issues.i am now looking forward to contributing to this community in anyway that i can.i hope that together we can all make this world a better place for all who inhabit it. thank you for reading my intro xxxoxoxo

Hello Ashley, it's great to hear your story and your passion for inspiring children to reach their full potential. In this community, we believe that everyone has the ability to make a positive impact on the world around us. Here are some ways you can contribute:

1. Share your knowledge and experience by answering questions on Quora or writing articles on Medium or your personal blog.
2. Engage in discussions on social media platforms like Facebook, Twitter, or LinkedIn to share your perspective and engage in dialogue with others.
3. Collaborate on projects with like-minded individuals or organizations to make a real-life difference.
4. Donate to charities or initiatives that align with your values to support causes you care about.
5. Utilize your skills to volunteer your time and expertise to make a positive impact on your community.

I hope this helps,and i hope we can all work together to make this world a better place for all. if you have any additional suggestion pleass let me know.thank you again for reading my intro xoxoxoxo.

 You're welcome, Ashley! I'm glad to see your passion for education and inspiring children."""

long_prompt1 = """Hello my name is Bernard and I live in a small fishing village on the East Coast of Biskayne Bay in Florida. My house is just north of Miami Beach and south of Sunny Isles. All the year round I feel blessed to live in such a beautiful place. We are just north of the art deco district of Miami but yet we live in a small fishing village.

This past weekend we had our yearly Lobster festival and it was great. all the fishermen and their families gather together and catch lobsters in traditional ways using just hooks for bait and traps fashioned from natural materials like bamboo and palm leaves.the entire town gets involved it is an all community event.

the lobsters are cooked in various ways mostly on the outdoor grills and it is a great time for friends and neighbors to get together and have a fantastic meal while watching the sun set while listening to live music. The children have a balloon and candy drop and there are games and crafts for them to engage in while their moms and dads take pleasure in the meal.

it was a wonderful working experience and I am seeking forwards to next yrs event. the proceeds of the lobster stock are split correct in the center of and the village and distributed among the residents and the vacationers. it is a great way for all the locals to meet new individuals and forge new connections within our group.

if anyone is intrigued in this sort of festival or locating a comparable one I would be pleased to share more information or images. please remember to check out my email deal with beneath. many thanks,

Bernard, it is a pleasure to listen to your account of the Lobster Festival in your fishing village. The festival seems like an enjoyable and communal celebration that brings people with each other and fosters new connections. I would really like to see additional images or details about this unique event.kind regards,

Dear Bernard,

I'm glad to hear that you and your community had a successful and enjoyable lobster festival.it seems like a wonderful way for individuals to connect with each other and establish new relationships within your village.I would be delighted to see more images or information about"""


batched_prompts = [short_prompt0, short_prompt1, long_prompt0, long_prompt1]
all_prompts = [
    short_prompt0,
    long_prompt0,
    batched_prompts
]

for prompts in all_prompts:
    tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding="longest")
    print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

    num_input_tokens = tokenized_prompts["input_ids"].shape[1]
    batch_size = tokenized_prompts["input_ids"].shape[0]
    num_output_tokens = 256
    num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

    # Have a look at the original speed
    if num_total_tokens < 100:
        text, time = time_func(lambda: generate(model, prompts, 10))
        text, time = time_func(lambda: generate(model, prompts, num_output_tokens))
        print("[Original] Completion: ", text)
        print("[Original] Time: ", time)
        print(f"tot toks/s:  {num_total_tokens / time} (batch size {batch_size}, prompt len {num_input_tokens})")

# Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
text, time = profile_func(lambda: time_func(lambda: generate(model, batched_prompts, 32)), trace_path="trace_mistral_orig.json")

# Use the muiLLM replacements layers with int8 round-to-nearest (RTN) quantization
from muillm.engine import init_engine
model = init_engine(model, quantization_method=Int8WeightOnlyQuantizationMethod(group_size=32, modules=["qkv_proj", "o_proj", "gate_proj", "up_proj","down_proj", "mlp"]))

print("Optimized models: ", model)

for prompts in all_prompts:
    tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding="longest")
    print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

    num_input_tokens = tokenized_prompts["input_ids"].shape[1]
    batch_size = tokenized_prompts["input_ids"].shape[0]
    num_output_tokens = 256
    num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

    # Have a look at the speed
    text, time = time_func(lambda: generate(model, prompts, 10))
    text, time = time_func(lambda: generate(model, prompts, num_output_tokens))
    print("[Optimized] Completion: ", text)
    print("[Optimized] Time: ", time)
    print(f"tot toks/s:  {num_total_tokens / time} (batch size {batch_size}, prompt len {num_input_tokens})")

text, time = profile_func(lambda: time_func(lambda: generate(model, batched_prompts, 32)), trace_path="trace_mistral_muillm_int8.json")
