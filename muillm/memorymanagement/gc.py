import torch
import gc


def trigger_gc():
    # make sure all kernels are finished
    # which allow the memory to be freed
    torch.cuda.synchronize()

    # trigger the destructor of the tensors
    gc.collect()

    # reclaim cuda memory
    torch.cuda.empty_cache()
