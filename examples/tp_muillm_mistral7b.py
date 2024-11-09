"""run.py:"""
#!/usr/bin/env python

# Example showing how to use muiLLM's tensor parallelism support on the Mistral 7b model

import os
import gc

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from muillm.commmunication.communicator import MuiCommunicator

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
import torch
import torch.nn as nn

from typing import List, Union


def generate(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt:Union[str, List[str]], max_new_tokens=20):
  single_prompt = isinstance(prompt, str)
  if single_prompt:
    prompts = [prompt]
  else:
    prompts = prompt

  with torch.no_grad():
    inputs = tokenizer(prompts, return_tensors="pt", padding="longest").to(device="cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

  return outputs

def decode(outputs, tokenizer: PreTrainedTokenizerBase, prompt:Union[str, List[str]]) -> Union[str, List[str]]:
  single_prompt = isinstance(prompt, str)
  if single_prompt:
    prompts = [prompt]
  else:
    prompts = prompt

  with torch.no_grad():
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

  texts = [text[len(prompts[i]):] for i, text in enumerate(texts)]

  if single_prompt:
    return texts[0]
  else:
    return texts

def time_func(f):
  import time
  # disable the GC for timing
  gc.disable()
  start_time = time.time()
  ret = f()
  end_time = time.time()
  gc.enable()
  elapsed_time = end_time - start_time
  return ret, elapsed_time

def profile_func(f, trace_path= "trace.json"):
  from torch.profiler import profile, ProfilerActivity
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    ret = f()
  prof.export_chrome_trace(trace_path)
  return ret

def run(rank, size):

    # this example requires the Mistral 7b Instruct v0.2 model
    # Provided that you have a HF token to access the Mistral models, you can download it with 
    # huggingface-cli download --token <your_hf_token> mistralai/Mistral-7B-Instruct-v0.2 --local-dir Mistral-7B-Instruct-v0.2 --revision 41b61a33a2483885c981aa79e0df6b32407ed873
    # (the specific revision is required as Mistral changed the repo to use their own tokenizer past that revision)

    # either set this environment variable before running the example, or adapt the path
    model_id = os.getenv("MISTRAL_7B_PATH", "/storage/models/Mistral-7B-Instruct-v0.2/")

    print("Process pid ", os.getpid(), flush=True)

    import time
    time.sleep(60)

    print("starting.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id,padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model: nn.Module = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device="cuda", dtype=torch.float16)

    model.resize_token_embeddings(len(tokenizer))

    if dist.get_rank() == 0:
        print("Model : ", model)

    from muillm.engine import init_engine
    model = init_engine(model)

    if dist.get_rank() == 0:
        print("Optimized models: ", model)

    # Test barriers
    for _ in range(1024):
      model.muillm_config.communicator.barrier()

    torch.cuda.synchronize()

    print(f"(rank {rank}) Finished all barriers.", flush=True)

    failed = False
    # Test broadcast
    for i in range(1024):
      src_rank = i % size
      orig_tensor = torch.rand(size=(1,2,3), dtype=torch.float16).to(device="cuda")

      tensor = torch.clone(orig_tensor)
      dist_tensor = torch.clone(orig_tensor)
      model.muillm_config.communicator.broadcast(tensor, src=src_rank)

      dist.broadcast(dist_tensor, src=src_rank)

      if ((i % 13) == 0):
         # check that the result is correct every so often
         matches = torch.sum(dist_tensor - tensor).item() == 0.0
         if not matches:
            print(f"Not matching: orig_tensor {dist_tensor} tensor {tensor}")
            failed = True

    torch.cuda.synchronize()

    print(f"(rank {rank}) Finished all broadcasts.", flush=True)

    if failed:
       return
  
      # Test reduce
    for i in range(1024):
      src_rank = i % size
      orig_tensor = torch.rand(size=(1,2,3), dtype=torch.float16).to(device="cuda")

      tensor = torch.clone(orig_tensor)
      dist_tensor = torch.clone(orig_tensor)

      model.muillm_config.communicator.broadcast(tensor, src=src_rank)
      model.muillm_config.communicator.all_reduce_sum(tensor)

      dist.broadcast(dist_tensor, src=src_rank)
      dist.all_reduce(dist_tensor)

      if ((i % 13) == 0):
         # check that the result is correct every so often
         matches = torch.sum(dist_tensor - tensor).item() == 0.0
         if not matches:
            print(f"Not matching: expected_tensor {dist_tensor} tensor {tensor}")

    torch.cuda.synchronize()

    print(f"(rank {rank}) Finished all reductions.", flush=True)

    return

    prompt = "Hello my name is"
    outputs, time = time_func(lambda: generate(model, tokenizer, prompt, 50))
    outputs, time = time_func(lambda: generate(model, tokenizer, prompt, 50))
    outputs, time = time_func(lambda: generate(model, tokenizer, prompt, 50))
  
    text = decode(outputs, tokenizer, prompt)
    print(f"[Optimized] Completion: {text} (rank {rank})")
    print(f"[Optimized] Time: {time} (rank {rank})")
    
    # Save pytorch traces per GPU (visualizable for example with https://ui.perfetto.dev)
    text, time = profile_func(lambda: time_func(lambda: generate(model, tokenizer, prompt, 50)), trace_path=f"trace_tp_muillm_rank{rank}.json")

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['LOCAL_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

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
