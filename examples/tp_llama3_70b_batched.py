"""run.py:"""

#!/usr/bin/env python

# Example showing how to use muiLLM's tensor parallelism support on the Llama 3.3 70b model

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
import torch
import torch.nn as nn

from typing import List, Union


def generate(
    model, tokenizer, prompt: Union[str, List[str]], max_new_tokens=20
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

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

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


def run(rank, size):

    # this example requires the LLama 3.3 70B Instruct model
    # Provided that you have a HF token to access the Llama models, you can download it with
    # huggingface-cli download --token <your_token> meta-llama/Llama-3.3-70B-Instruct --local-dir Llama-3.3-70B-Instruct

    # either set this environment variable before running the example, or adapt the path
    model_id = os.getenv("LLAMA3_70B_PATH", "/storage/models/Llama-3.3-70B-Instruct/")

    ## Load the original model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    model: nn.Module = AutoModelForCausalLM.from_pretrained(
        model_id, tp_plan="auto"
    ).to(dtype=torch.float16)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    if rank == 0:
        print("Model : ", model)

    # 5 tokens prompt
    short_prompt0 = "Hello my name is Ashley"
    short_prompt1 = "Hello my name is Bernard"

    # 467 tokens prompt
    long_prompt0 = """Hello my name is Ashley.
    I am a 23 year old singer songwriter from the UK. I am a Christian and I enjoy expressing my faith through music. I released my first single 'The Waiting Room' in 2021 and I'm currently working on new material.
    My biggest influences are Bethel Music, Hillsong, and Kari Jobe. I am inspired by the way their music captures the heart of God and speaks to people on a deep level.
    My music style is a mix of worship and contemporary style music. I aim to create music that is authentic, raw and real. Music that speaks to the heart and inspires people to connect with God on a deeper level.
    I'm excited to share my music with you and I hope it encourages and inspires you in your walk with God. Thank you for taking the time to listen to my music.
    I'm so glad you're here and I look forward to sharing my music with you.
    Keep following for updates and new music.
    This is just the beginning... - Ashley - 🙌
    My first single "The Waiting Room" is available now on all music platforms.
    You can find me on Spotify, Apple Music, TikTok and Instagram.
    Subscribe to my YouTube channel and stay updated on new music, behind the scenes and more.
    If you want to stay up to date with my music and tours then join my newsletter here: [insert link]
    Let's do this - Ashley 💖
    #ashley #newmusic #Christianmusic #worshipmusic #singersongwriter #the waiting room #new release #Christianmusic artist #UK #music #musicproducer #singer #guitarist #musician #newmusic2021 #Christianband #Christianmusicindustry #musicforjesus #worshipleader #Christianlife #Christiannmusicblog #Christianmusicstore #Christianmusicstreaming #Christianmusicradio #Christianmusicnetwork #Christianmusician #Christianmusicforum #Christianmusicgroup #christianmusicinspiration ### 🎶🙌
    # More new music coming soon... 🤩
    # Stay tuned and thank you for being part of this journey with me. 💖
    Follow me:
    TikTok: Ashley
    Spotify: Ashley
    Instagram: Ashley
    YouTube: Ashley
    #staytuned #"""

    long_prompt1 = """Hello my name is Bernard, I am happy to introduce my partner and me to you and welcome to our home. The cottage, as you see, has been completely restored to it's original charm with lots of character and a hint of vintage elegance. We have taken care to preserve the original features and restored it to make it a beautiful comfortable home for you to stay in.
    The property is situated in the picturesque village of Coggeshall, with its picturesque high street lined with independent shops, tea rooms and traditional village pubs. It truly is a haven away from the hustle and bustle of city life.
    Our cottage sleeps 4 people and is perfect for couples, families or a group of friends looking for a relaxing stay in a beautiful Suffolk countryside setting.
    The property is fully equipped with all modern appliances and amenities, including a fully fitted kitchen with dishwasher, cooker, microwave, kettle, toaster and fridge freezer. There is also a large television, digital freeview and Wi-Fi for your use. The cottage also has 2 bedrooms, one double and one twin bedroom, each with its own en-suite bathroom. There is also a lounge with comfortable seating and a beautifully decorated dining area where you can enjoy meals.
    Outside, the cottage has a private garden with beautiful views of the surrounding countryside, a perfect place to sit and relax.
    We have numerous beautiful walks and cycle routes surrounding the village, so if you love the great outdoors, you will find plenty to enjoy. There are also numerous pubs, restaurants and tea rooms within walking distance, so you can enjoy a leisurely walk into the village and explore what has to offer.
    Please take a look around our cottage and I would be happy to answer any questions you may have.
    Welcome to our lovely cottage in the heart of the Suffolk countryside! We are an owner-managed property, so we pride ourselves on the highest standards of cleanliness and hospitality. Before you check-in, please take a look below at what you can expect, and feel free to reach out if you have any questions or need any assistance.
    **Cleaning and housekeeping**
    Our cottage is cleaned and inspected after every stay, using only eco-friendly cleaning products. We also provide fresh towels and bedding for each guest.
    **Check-in and check-out**
    Check-in is from 3 pm, and check-out is by"""

    batched_prompts = [short_prompt0, short_prompt1, long_prompt0, long_prompt1]
    all_prompts = [short_prompt0, long_prompt0, batched_prompts]

    # use auto-detected tensor parallelism level by setting to None
    from muillm.engine import init_engine

    model = init_engine(model, tensor_parallelism=None)

    if rank == 0:
        print("Optimized models: ", model)

    for prompts in all_prompts:
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding="longest")

        if rank == 0:
            print("tokenized prompts: ", tokenized_prompts["input_ids"].shape)

        num_input_tokens = tokenized_prompts["input_ids"].shape[1]
        batch_size = tokenized_prompts["input_ids"].shape[0]
        num_output_tokens = 256
        num_total_tokens = (num_input_tokens + num_output_tokens) * batch_size

        # Have a look at the speed
        text, time = time_func(lambda: generate(model, tokenizer, prompts, 10))
        text, time = time_func(
            lambda: generate(model, tokenizer, prompts, num_output_tokens)
        )

        # check how many tokens were actually generated
        tokenized_outputs = tokenizer(text, return_tensors="pt", padding="longest")
        num_total_tokens = (
            num_input_tokens + tokenized_outputs["input_ids"].shape[1]
        ) * batch_size

        if rank == 0:
            print("[Optimized] Completion: ", text)
            print("[Optimized] Time: ", time)
            print(
                f"tot toks/s:  {num_total_tokens / time} (batch size {batch_size}, prompt len {num_input_tokens})"
            )

    # Save a pytorch trace (visualizable for example with https://ui.perfetto.dev)
    text, time = profile_func(
        lambda: time_func(
            lambda: generate(model, tokenizer, batched_prompts, num_output_tokens)
        ),
        trace_path=f"trace_llama70b_muillm_tp{size}_batched_rank{rank}.json",
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
