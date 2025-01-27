from typing import Any, Dict, Optional, Tuple
from transformers.cache_utils import StaticCache
from transformers.configuration_utils import PretrainedConfig

import torch

class MuiStaticCache(StaticCache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(
            self,
            config: PretrainedConfig,
            max_batch_size: int,
            max_cache_len: int,
            device,
            dtype=None
        ) -> None:
        super().__init__(config=config, max_batch_size=max_batch_size, max_cache_len=max_cache_len, device=device, dtype=dtype)
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # update like usual
        ret = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # and update the seen counter
        self._seen_tokens += key_states.shape[-2]

        return ret

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # compared to the HF implementation, that avoids a CPU<->GPU sync
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        super().reset()
        self._seen_tokens = 0

def _next_pow2(x: int) -> int:
    if x < 0:
        raise ValueError("x should be positive")
    
    p = 1
    while p < x:
        p = p * 2

    return p

def create_static_cache(config: PretrainedConfig, max_batch_size, seq_len, device, dtype) -> MuiStaticCache:
    # to avoid frequent re-allocations of the cache, we use a power of 2 schedule
    max_cache_len = _next_pow2(seq_len)
    return MuiStaticCache(config, max_batch_size, max_cache_len, device, dtype)

def grow_static_cache_if_needed(cache: MuiStaticCache, capacity: int, max_capacity: int) -> MuiStaticCache:
    required_capacity = _next_pow2(capacity)
    # models have a max supported sequence length, no need to go past that
    required_capacity = min(required_capacity, max_capacity)

    if cache.max_cache_len >= required_capacity:
        # already good
        return cache
    
    # we need to grow the cache
    prev_k_cache = cache.key_cache[0]

    dtype = prev_k_cache.dtype
    device = prev_k_cache.device
    prev_cache_shape = prev_k_cache.shape

    max_batch_size, num_key_value_heads, prev_max_cache_len, head_dim = prev_cache_shape

    # by how much we need to increase the cache size
    diff_cache_len = required_capacity - prev_max_cache_len

    diff_cache_shape = (max_batch_size, num_key_value_heads, diff_cache_len, head_dim)

    num_hidden_layers = len(cache.key_cache)
    for layer_idx in range(num_hidden_layers):
        prev_key_cache = cache.key_cache[layer_idx]
        prev_value_cache = cache.value_cache[layer_idx]

        diff_layer_key_cache = torch.zeros(diff_cache_shape, dtype=dtype, device=device)
        diff_layer_value_cache = torch.zeros(diff_cache_shape, dtype=dtype, device=device)
    
        new_layer_key_cache = torch.cat([prev_key_cache, diff_layer_key_cache], dim=-2)
        new_layer_value_cache = torch.cat([prev_value_cache, diff_layer_value_cache], dim=-2)

        # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
        # breaks when updating the cache.
        torch._dynamo.mark_static_address(new_layer_key_cache)
        torch._dynamo.mark_static_address(new_layer_value_cache)

        cache.key_cache[layer_idx] = (new_layer_key_cache)
        cache.value_cache[layer_idx] = (new_layer_value_cache)

        # delete now
        del prev_key_cache
        del prev_value_cache

    # update the capacity
    cache.max_cache_len = required_capacity
    
    return cache