import torch
import gc

def trigger_gc():
    torch.cuda.empty_cache()
    gc.collect()