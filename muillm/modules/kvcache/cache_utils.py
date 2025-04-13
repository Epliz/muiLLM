from typing import Any, Dict, Optional, Tuple
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.module import MuiModule
from transformers.cache_utils import StaticCache, DynamicCache
from transformers.configuration_utils import PretrainedConfig

import muillm_ext
import torch

class MuiCache:
    pass

class MuiDynamicCache(DynamicCache, MuiCache):
    def __init__(
            self,
            engine_config: MuiEngineConfig,
            *args, **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)

        self.engine_config = engine_config
        self.cpp_engine = engine_config.cpp_engine

        self.cpp_module = None

        # TODO: create all layer tensors with empty tensors?

        self._seen_tokens = 0

        # create the cpp module
        self.finalize_init()

    def finalize_init(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_dynamic_kvcache_module_deinit(self.cpp_module)

        # TODO: make sure the cache lists are always populated, even if with empty
        # tensors?
        self.cpp_module = muillm_ext.muillm_dynamic_kvcache_module_init(
            self.cpp_engine,
            self.key_cache,
            self.value_cache,
            self._seen_tokens
        )

    def sync_back(self):
        self.key_cache, self.value_cache = muillm_ext.muillm_dynamic_kvcache_module_sync_back(self.cpp_module)
        self._seen_tokens = muillm_ext.muillm_kvcache_module_get_seen_tokens(self.cpp_module)

class MuiStaticCache(StaticCache, MuiCache):
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
            engine_config: MuiEngineConfig,
            config: PretrainedConfig,
            batch_size: int = None,
            max_cache_len: int = None,
            device: torch.device = None,
            dtype=None,
            tensor_parallelism: int = 1,
            max_batch_size: Optional[int] = None,
        ) -> None:

        # hack to make the cache be the right size if we use tensor parallelism
        if config.num_key_value_heads is not None:
            config.num_key_value_heads = config.num_key_value_heads // tensor_parallelism
        if config.num_attention_heads is not None:
            config.num_attention_heads = config.num_attention_heads // tensor_parallelism
        if not hasattr(config, "head_dim"):
            # for some models, HF compute the head_dim as hidden_size / num_attention_heads
            # but we lower num_attention_heads, so we need to compensate for that otherwise
            # the head dim is wrong
            config.hidden_size  = config.hidden_size // tensor_parallelism

        super().__init__(config=config, batch_size=batch_size, max_cache_len=max_cache_len, device=device, dtype=dtype, max_batch_size=max_batch_size)

        # set back the right values in the config
        if config.num_key_value_heads is not None:
            config.num_key_value_heads = config.num_key_value_heads * tensor_parallelism
        if config.num_attention_heads is not None:
            config.num_attention_heads = config.num_attention_heads * tensor_parallelism
        if not hasattr(config, "head_dim"):
            config.hidden_size  = config.hidden_size * tensor_parallelism

        self._seen_tokens = 0

        self.engine_config = engine_config
        self.cpp_engine = engine_config.cpp_engine

        self.cpp_module = None

        # create the cpp module
        self.finalize_init()

    def finalize_init(self):
        if self.cpp_module is not None:
            muillm_ext.muillm_static_kvcache_module_deinit(self.cpp_module)

        self.cpp_module = muillm_ext.muillm_static_kvcache_module_init(
            self.cpp_engine,
            self.key_cache,
            self.value_cache,
            self._seen_tokens
        )

    def sync_back(self):
        self.key_cache, self.value_cache = muillm_ext.muillm_static_kvcache_module_sync_back(self.cpp_module)
        self._seen_tokens = muillm_ext.muillm_kvcache_module_get_seen_tokens(self.cpp_module)

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
    
    def grow_cache(self, capacity: int, max_capacity: int) -> None:
        required_capacity = _next_pow2(capacity)
        # models have a max supported sequence length, no need to go past that
        required_capacity = min(required_capacity, max_capacity)

        if self.max_cache_len >= required_capacity:
            # already good
            return self
        
        # we need to grow the cache
        prev_k_cache = self.key_cache[0]

        dtype = prev_k_cache.dtype
        device = prev_k_cache.device
        prev_cache_shape = prev_k_cache.shape

        max_batch_size, num_key_value_heads, prev_max_cache_len, head_dim = prev_cache_shape

        # by how much we need to increase the cache size
        diff_cache_len = required_capacity - prev_max_cache_len

        diff_cache_shape = (max_batch_size, num_key_value_heads, diff_cache_len, head_dim)

        num_hidden_layers = len(self.key_cache)
        for layer_idx in range(num_hidden_layers):
            prev_key_cache = self.key_cache[layer_idx]
            prev_value_cache = self.value_cache[layer_idx]

            diff_layer_key_cache = torch.zeros(diff_cache_shape, dtype=dtype, device=device)
            diff_layer_value_cache = torch.zeros(diff_cache_shape, dtype=dtype, device=device)
        
            new_layer_key_cache = torch.cat([prev_key_cache, diff_layer_key_cache], dim=-2)
            new_layer_value_cache = torch.cat([prev_value_cache, diff_layer_value_cache], dim=-2)

            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)

            self.key_cache[layer_idx] = (new_layer_key_cache)
            self.value_cache[layer_idx] = (new_layer_value_cache)

            # delete now
            del prev_key_cache
            del prev_value_cache

        # update the capacity
        self.max_cache_len = required_capacity

        # we need to reinitialize the cpp module as the tensors changed
        self.finalize_init()

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

def create_static_cache(engine_config: MuiEngineConfig, config: PretrainedConfig, max_batch_size, seq_len, device, dtype) -> MuiStaticCache:
    # to avoid frequent re-allocations of the cache, we use a power of 2 schedule
    max_cache_len = _next_pow2(seq_len)
    tensor_parallelism = engine_config.tensor_parallelism

    return MuiStaticCache(
        engine_config=engine_config,
        config=config,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
        tensor_parallelism=tensor_parallelism,
        max_batch_size=max_batch_size
    )

def grow_static_cache_if_needed(cache: MuiStaticCache, capacity: int, max_capacity: int) -> MuiStaticCache:
    cache.grow_cache(capacity=capacity, max_capacity=max_capacity)
    return cache