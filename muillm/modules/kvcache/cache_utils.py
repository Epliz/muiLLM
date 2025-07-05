from typing import Any, Dict, Optional, Tuple, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.module import MuiModule
from transformers.cache_utils import StaticCache, DynamicCache, HybridChunkedCache
from transformers.configuration_utils import PretrainedConfig

import muillm_ext
import torch


def _set_sharded_attention_config(
    config: PretrainedConfig, tensor_parallelism: int
) -> None:
    # hack to make the cache be the right size if we use tensor parallelism
    if config.num_key_value_heads is not None:
        config.num_key_value_heads = config.num_key_value_heads // tensor_parallelism
    if config.num_attention_heads is not None:
        config.num_attention_heads = config.num_attention_heads // tensor_parallelism
    if getattr(config, "head_dim", None) is None:
        # for some models, HF compute the head_dim as hidden_size / num_attention_heads
        # but we lower num_attention_heads, so we need to compensate for that otherwise
        # the head dim is wrong
        config.hidden_size = config.hidden_size // tensor_parallelism


def _reset_sharded_attention_config(
    config: PretrainedConfig, tensor_parallelism: int
) -> None:
    # set back the right values in the config
    if config.num_key_value_heads is not None:
        config.num_key_value_heads = config.num_key_value_heads * tensor_parallelism
    if config.num_attention_heads is not None:
        config.num_attention_heads = config.num_attention_heads * tensor_parallelism
    if getattr(config, "head_dim", None) is None:
        config.hidden_size = config.hidden_size * tensor_parallelism


def _get_num_key_value_heads(config: PretrainedConfig):
    num_key_value_heads = (
        config.num_attention_heads
        if getattr(config, "num_key_value_heads", None) is None
        else config.num_key_value_heads
    )
    return num_key_value_heads


class MuiCache:
    pass

    def sync_back(self):
        pass


class MuiDynamicCache(DynamicCache, MuiCache):
    def __init__(self, engine_config: MuiEngineConfig, *args, **kwargs) -> None:
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
            self.cpp_engine, self.key_cache, self.value_cache, self._seen_tokens
        )

    def _sync_seen_tokens(self):
        if self.cpp_module is not None:
            self._seen_tokens = muillm_ext.muillm_kvcache_module_get_set_seen_tokens(
                self.cpp_module, self._seen_tokens
            )

    def sync_back(self):
        self.key_cache, self.value_cache = (
            muillm_ext.muillm_dynamic_kvcache_module_sync_back(self.cpp_module)
        )
        self._sync_seen_tokens()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # the HF dynamic cache might have its own counter
        # this helps decouple a bit and not rely on knowing their logic
        prev_seen_tokens = self._seen_tokens

        k_out, v_out = super().update(key_states, value_states, layer_idx, cache_kwargs)

        if layer_idx == 0:
            # and update the seen counter (only for layer 0 to avoid double counting)
            self._seen_tokens = prev_seen_tokens + key_states.shape[-2]
            self._sync_seen_tokens()

        return k_out, v_out


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
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        tensor_parallelism: int = 1,
    ) -> None:

        # hack to make the cache be the right size if we use tensor parallelism
        _set_sharded_attention_config(config, tensor_parallelism)

        super().__init__(
            config=config,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            max_batch_size=max_batch_size,
        )

        # set back the right values in the config
        _reset_sharded_attention_config(config, tensor_parallelism)

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
            self.cpp_engine, self.key_cache, self.value_cache, self._seen_tokens
        )

    def _sync_seen_tokens(self):
        if self.cpp_module is not None:
            self._seen_tokens = muillm_ext.muillm_kvcache_module_get_set_seen_tokens(
                self.cpp_module, self._seen_tokens
            )

    def sync_back(self):
        self.key_cache, self.value_cache = (
            muillm_ext.muillm_static_kvcache_module_sync_back(self.cpp_module)
        )
        self._sync_seen_tokens()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # update like usual
        k_out, v_out = super().update(key_states, value_states, layer_idx, cache_kwargs)

        if layer_idx == 0:
            # and update the seen counter (only for layer 0 to avoid double counting)
            self._seen_tokens += key_states.shape[-2]
            self._sync_seen_tokens()

        # return the minimal slice of cache
        k_out = torch.narrow(k_out, 2, 0, self._seen_tokens)
        v_out = torch.narrow(v_out, 2, 0, self._seen_tokens)

        return k_out, v_out

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

        max_batch_size, num_key_value_heads, prev_max_cache_len, head_dim = (
            prev_cache_shape
        )

        # by how much we need to increase the cache size
        diff_cache_len = required_capacity - prev_max_cache_len

        diff_cache_shape = (
            max_batch_size,
            num_key_value_heads,
            diff_cache_len,
            head_dim,
        )

        num_hidden_layers = len(self.key_cache)
        for layer_idx in range(num_hidden_layers):
            prev_key_cache = self.key_cache[layer_idx]
            prev_value_cache = self.value_cache[layer_idx]

            diff_layer_key_cache = torch.zeros(
                diff_cache_shape, dtype=dtype, device=device
            )
            diff_layer_value_cache = torch.zeros(
                diff_cache_shape, dtype=dtype, device=device
            )

            new_layer_key_cache = torch.cat(
                [prev_key_cache, diff_layer_key_cache], dim=-2
            )
            new_layer_value_cache = torch.cat(
                [prev_value_cache, diff_layer_value_cache], dim=-2
            )

            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.

            # TODO: maybe unmark the previous cache?
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)

            self.key_cache[layer_idx] = new_layer_key_cache
            self.value_cache[layer_idx] = new_layer_value_cache

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


class _MuiHybridChunkedCacheUpdate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cpp_module,
        key_states,
        value_states,
        cache_position,
        layer_index,
    ):
        output = muillm_ext.muillm_hybrid_chunked_kvcache_module_update(
            cpp_module,
            key_states,
            value_states,
            cache_position,
            layer_index,
        )
        ctx.save_for_backward(key_states, value_states)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Hybrid Chunked cache backward not implemented")


class MuiHybridChunkedCache(HybridChunkedCache, MuiCache):

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.bfloat16,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
        tensor_parallelism: int = 1,
    ) -> None:

        self.device = device
        self.dtype = dtype

        # hack to make the cache be the right size if we use tensor parallelism
        # in particular number of heads is modified due to the tensor parallelism
        _set_sharded_attention_config(config, tensor_parallelism)

        super().__init__(
            config=config,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            max_batch_size=max_batch_size,
            layer_device_map=layer_device_map,
        )

        self.num_key_value_heads = _get_num_key_value_heads(config)
        self.num_hidden_layers = config.num_hidden_layers

        # set back the right values in the config
        _reset_sharded_attention_config(config, tensor_parallelism)

        # _seen_tokens is incremented in the update function
        self._seen_tokens = 0

        self.engine_config = engine_config

        # initialize all layers
        for l in range(self.num_hidden_layers):
            self._init_cache_layer(l)

        self.engine_config = engine_config
        self.cpp_engine = engine_config.cpp_engine

        self.cpp_module = None

        # create the cpp module
        self.finalize_init()

    def _check_dispatchable(self):
        dispatchable_type = (self.dtype == torch.float16) or (
            self.dtype == torch.bfloat16
        )
        self.dispatchable = dispatchable_type and self.key_cache[0].is_cuda

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        if self.cpp_module is not None:
            muillm_ext.muillm_hybrid_chunked_kvcache_module_deinit(self.cpp_module)

        self.cpp_module = muillm_ext.muillm_hybrid_chunked_kvcache_module_init(
            self.cpp_engine,
            self.key_cache,
            self.value_cache,
            self.is_sliding,
            self.sliding_window,
            self._seen_tokens,
        )

    def _sync_seen_tokens(self):
        if self.cpp_module is not None:
            self._seen_tokens = muillm_ext.muillm_kvcache_module_get_set_seen_tokens(
                self.cpp_module, self._seen_tokens
            )

    def sync_back(self):
        self.key_cache, self.value_cache = (
            muillm_ext.muillm_hybrid_chunked_kvcache_module_sync_back(self.cpp_module)
        )
        self._sync_seen_tokens()

    def _init_cache_layer(self, layer_idx):
        if len(self.key_cache) > layer_idx:
            return

        num_key_value_heads = self.num_key_value_heads
        device = self.device
        global_cache_shape = (
            self.max_batch_size,
            num_key_value_heads,
            self.max_cache_len,
            self.head_dim,
        )
        sliding_cache_shape = (
            self.max_batch_size,
            num_key_value_heads,
            self.sliding_window,
            self.head_dim,
        )
        # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
        # breaks when updating the cache.
        cache_shape = (
            sliding_cache_shape if self.is_sliding[layer_idx] else global_cache_shape
        )
        new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=device)
        new_layer_value_cache = torch.zeros(
            cache_shape, dtype=self._dtype, device=device
        )

        # TODO: we change the cache tensors in some cases
        # it might leak this initial memory as we mark them as static
        torch._dynamo.mark_static_address(new_layer_key_cache)
        torch._dynamo.mark_static_address(new_layer_value_cache)

        self.key_cache.append(new_layer_key_cache)
        self.value_cache.append(new_layer_value_cache)

    def _static_update(
        self,
        cache_position,
        layer_idx,
        key_states,
        value_states,
        k_out,
        v_out,
        max_cache_len,
    ):
        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        # return the minimal slice of cache
        k_out = torch.narrow(k_out, 2, 0, self._seen_tokens)
        v_out = torch.narrow(v_out, 2, 0, self._seen_tokens)

        return k_out, v_out

    def _sliding_update(
        self,
        cache_position,
        layer_idx,
        key_states,
        value_states,
        k_out,
        v_out,
        max_cache_len,
    ):
        # Many subtle cases here to take into account...
        # Depending on whether we are going over the sliding window size
        # due to several in-tokens, we might have to return different shapes

        num_new_tokens = key_states.shape[2]
        cumulative_length = self.cumulative_length[layer_idx]
        # Update it now that we saved the value above
        self.cumulative_length[layer_idx] += num_new_tokens

        if cumulative_length == 0:
            full_key_states = key_states
            full_value_states = value_states

            if num_new_tokens <= max_cache_len:
                # We are not full, so we can just return the new states
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            else:
                # We are full, so we need to keep the latest tokens
                k_out.copy_(key_states[:, :, -max_cache_len:, :])
                v_out.copy_(value_states[:, :, -max_cache_len:, :])

            # we should return the whole states instead of k_out, v_out to take the whole prompt
            # into consideration when building kv cache instead of just throwing away tokens outside of the window
            return full_key_states, full_value_states
        elif cumulative_length + num_new_tokens > max_cache_len:
            # Decoding and becoming full or already full
            # We erase old tokens and keep the latest ones assuming the forgetting is not catastrophic
            if num_new_tokens < max_cache_len:
                # we still need to copy some of the previous tokens
                full_key_states = torch.cat(
                    (k_out[:, :, num_new_tokens:, :], key_states), dim=-2
                )
                full_value_states = torch.cat(
                    (v_out[:, :, num_new_tokens:, :], value_states), dim=-2
                )
            else:
                # very large number of new tokens, we just keep the latest ones
                full_key_states = key_states[:, :, -max_cache_len:, :]
                full_value_states = value_states[:, :, -max_cache_len:, :]
            k_out.copy_(full_key_states)
            v_out.copy_(full_value_states)
            return k_out, v_out
        else:
            # Not full, not becoming full
            # Similar to a static cache update
            k_out.index_copy_(2, cache_position, key_states)
            v_out.index_copy_(2, cache_position, value_states)

            # we need to narrow
            k_out = torch.narrow(k_out, 2, 0, self._seen_tokens)
            v_out = torch.narrow(v_out, 2, 0, self._seen_tokens)

            return k_out, v_out

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if cache_kwargs is None:
            cache_kwargs = {}

        cache_position = cache_kwargs.get("cache_position")

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        if self.dispatchable:
            if layer_idx == 0:
                # update the seen counter (only for layer 0 to avoid double counting)
                # the C++ side increments it too, so no need to sync
                self._seen_tokens += key_states.shape[-2]

            # Use the C++ module to do the update
            k_out, v_out = _MuiHybridChunkedCacheUpdate.apply(
                self.cpp_module, key_states, value_states, cache_position, layer_idx
            )
            return k_out, v_out
        else:
            if layer_idx == 0:
                # update the seen counter (only for layer 0 to avoid double counting)
                self._seen_tokens += key_states.shape[-2]
                self._sync_seen_tokens()

            if self.is_sliding[layer_idx]:
                k_out, v_out = self._sliding_update(
                    cache_position,
                    layer_idx,
                    key_states,
                    value_states,
                    k_out,
                    v_out,
                    k_out.shape[2],
                )
            else:
                k_out, v_out = self._static_update(
                    cache_position,
                    layer_idx,
                    key_states,
                    value_states,
                    k_out,
                    v_out,
                    k_out.shape[2],
                )

            return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        if layer_idx != 0:
            raise ValueError(
                "`get_seq_length` on `HybridCache` may get inconsistent results depending on the layer index. "
                "Using the `layer_idx` argument is not supported."
            )
        if len(self.key_cache) == 0:
            return 0

        return self._seen_tokens

    def reset(self):
        super().reset()

        self._seen_tokens = 0

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

        num_key_value_heads = prev_k_cache.shape[1]
        prev_global_cache_shape = (
            self.max_batch_size,
            num_key_value_heads,
            self.max_cache_len,
            self.head_dim,
        )

        max_batch_size, num_key_value_heads, prev_max_cache_len, head_dim = (
            prev_global_cache_shape
        )

        # by how much we need to increase the cache size
        diff_cache_len = required_capacity - prev_max_cache_len

        diff_cache_shape = (
            max_batch_size,
            num_key_value_heads,
            diff_cache_len,
            head_dim,
        )

        num_hidden_layers = len(self.key_cache)
        for layer_idx in range(num_hidden_layers):
            if self.is_sliding[layer_idx]:
                # if the layer is sliding, we don't need to grow the cache
                # it is already at the window size
                continue

            # global attention cache need to grow
            prev_key_cache = self.key_cache[layer_idx]
            prev_value_cache = self.value_cache[layer_idx]

            diff_layer_key_cache = torch.zeros(
                diff_cache_shape, dtype=dtype, device=device
            )
            diff_layer_value_cache = torch.zeros(
                diff_cache_shape, dtype=dtype, device=device
            )

            new_layer_key_cache = torch.cat(
                [prev_key_cache, diff_layer_key_cache], dim=-2
            )
            new_layer_value_cache = torch.cat(
                [prev_value_cache, diff_layer_value_cache], dim=-2
            )

            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.

            # TODO: maybe unmark the previous cache?
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)

            self.key_cache[layer_idx] = new_layer_key_cache
            self.value_cache[layer_idx] = new_layer_value_cache

            # delete now
            del prev_key_cache
            del prev_value_cache

        # update the capacity
        self.max_cache_len = required_capacity

        # we need to reinitialize the cpp module as the tensors changed
        self.finalize_init()


def _next_pow2(x: int) -> int:
    if x < 0:
        raise ValueError("x should be positive")

    p = 1
    while p < x:
        p = p * 2

    return p


def create_static_cache(
    engine_config: MuiEngineConfig,
    config: PretrainedConfig,
    max_batch_size,
    seq_len,
    device,
    dtype,
) -> MuiStaticCache:
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
        max_batch_size=max_batch_size,
    )


def grow_static_cache_if_needed(
    cache: MuiStaticCache, capacity: int, max_capacity: int
) -> MuiStaticCache:
    cache.grow_cache(capacity=capacity, max_capacity=max_capacity)
    return cache


def create_hybrid_chunked_cache(
    engine_config: MuiEngineConfig,
    config: PretrainedConfig,
    max_batch_size,
    seq_len,
    device,
    dtype,
) -> MuiStaticCache:
    # to avoid frequent re-allocations of the cache, we use a power of 2 schedule
    max_cache_len = _next_pow2(seq_len)
    tensor_parallelism = engine_config.tensor_parallelism

    return MuiHybridChunkedCache(
        engine_config=engine_config,
        config=config,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
        tensor_parallelism=tensor_parallelism,
        max_batch_size=max_batch_size,
    )


def grow_hybrid_chunked_cache_if_needed(
    cache: MuiHybridChunkedCache, capacity: int, max_capacity: int
) -> MuiHybridChunkedCache:
    cache.grow_cache(capacity=capacity, max_capacity=max_capacity)
    return cache
