import torch

# from https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
def muillm_multinomial_sample_one_no_sync(probs): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)
