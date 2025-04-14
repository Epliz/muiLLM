from typing import Any, Dict, Optional, Tuple, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.kvcache.cache_utils import MuiCache
from muillm.modules.module import MuiModule
import torch
import torch.nn as nn

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import muillm_ext

    
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# TODO: custom kernel for BS=1, S=1 case at least
@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim:int=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # positions_ids  torch.Size([B, T])
    # cos  torch.Size([54, 128])
    # sin  torch.Size([54, 128])
    # q  torch.Size([B, 32, T, 128])
    # k  torch.Size([B, 8, T, 128])
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # cos[position_ids]  torch.Size([B, 1, T, 128])
    # sin[position_ids]  torch.Size([B, 1, T, 128])
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class _MuiRotaryNoCache(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions_ids, cos_cached, sin_cached, q, k):
        output = muillm_ext.muillm_rope_forward_no_cache(positions_ids, cos_cached, sin_cached, q, k)

        ctx.save_for_backward(positions_ids, cos_cached, sin_cached, q, k)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("rotary backward not implemented")

# Generic for all Mui cache modules
class _MuiRotaryCacheModule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, cache, q, k, v, position_ids, cos_sin, cache_positions):
        output = muillm_ext.muillm_rotary_embedding_module_forward(module, cache, q, k, v, position_ids, cos_sin, cache_positions)

        ctx.save_for_backward(q, k, v, position_ids, cos_sin, cache_positions)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("rotary backward not implemented")

class _MuiRotaryHFDynamicCache(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions_ids, cos_cached, sin_cached, q, k, v, prev_k_cache, prev_v_cache):
        output = muillm_ext.muillm_rope_forward_dynamic_cache(positions_ids, cos_cached, sin_cached, q, k, v, prev_k_cache, prev_v_cache)

        ctx.save_for_backward(positions_ids, cos_cached, sin_cached, q, k)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("rotary backward not implemented")


class _MuiRotaryHFStaticCache(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions_ids, cos_cached, sin_cached, q, k, v, k_cache, v_cache, cache_position, seen_tokens):
        output = muillm_ext.muillm_rope_forward_static_cache(positions_ids, cos_cached, sin_cached, q, k, v, k_cache, v_cache, cache_position, seen_tokens)

        ctx.save_for_backward(positions_ids, cos_cached, sin_cached, q, k)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("rotary backward not implemented")
    
class MuiRotaryEmbedding(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, config: Union[LlamaConfig, MistralConfig], rope_kwargs: Dict[str, Any] = None, layer_idx: int = 0, device=None, dtype=None):
        super().__init__(engine_config=engine_config)

        self.cpp_engine = engine_config.cpp_engine
        self.config = config
        self.rope_kwargs = {}

        self.layer_idx = layer_idx

        dtype = dtype if dtype is not None else torch.get_default_dtype()

        if config is not None:
            if isinstance(config, LlamaConfig) or isinstance(config, MistralConfig):
                # BC: "rope_type" was originally "type"
                if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
                    self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
                else:
                    self.rope_type = "default"
            else:
                self.rope_type = "default"
            
            max_position_embeddings = config.max_position_embeddings
        else:
            if rope_kwargs is None:
                raise ValueError("Either config or rope_kwargs should be not None")

            self.rope_kwargs = rope_kwargs
            self.rope_type = rope_kwargs["rope_type"]
            max_position_embeddings = rope_kwargs["max_position_embeddings"]

        self.max_position_embeddings = max_position_embeddings

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=device, dtype=dtype
        )

        self.dtype = dtype

        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        # the cpp module will be created at the end of all layer replacements
        self.cpp_module = None

    def _check_dispatchable(self):
        # cos and sin are of type self.dtype
        # but the frequencies are float32
        dispatchable_type = (self.dtype == torch.float16)
        dispatchable_device = self.inv_freq.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        if self.cpp_module is not None:
            muillm_ext.muillm_rotary_embedding_module_deinit(self.cpp_module)

        self.cpp_module = muillm_ext.muillm_rotary_embedding_module_init(
            self.cpp_engine,
            self.layer_idx,
            self.cos_cached,
            self.sin_cached,
        )

    @staticmethod
    def replace(prev_module: Union[LlamaRotaryEmbedding, MistralRotaryEmbedding], engine_config: MuiEngineConfig, device=None) -> "MuiRotaryEmbedding":
        if device is None:
            raise ValueError("device was None")

        device = prev_module.inv_freq.device if device is None else device
        dtype = prev_module.inv_freq.dtype

        # we either need a model config for Llama or rope_kwargs
        config = None
        rope_kwargs = None
        if isinstance(prev_module, LlamaRotaryEmbedding) or isinstance(prev_module, MistralRotaryEmbedding):
            config = prev_module.config
        else:
            raise ValueError(f"Unsupported type of module: {prev_module.__class__.__name__}")

        new_module = MuiRotaryEmbedding(engine_config=engine_config, config=config, rope_kwargs=rope_kwargs, device=device, dtype=dtype)

        return new_module

    def _set_cos_sin_cache(self, seq_len, device, dtype):

        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)

    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        cos = self.cos_cached[position_ids].to(dtype=x.dtype)
        sin = self.sin_cached[position_ids].to(dtype=x.dtype)
        return cos, sin
    
    def apply_rotary_pos_emb_write_kv_cache(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            position_ids: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
            v: Optional[torch.Tensor] = None,
            cache: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            #raise ValueError("should be provided")
            # Shape [S, E]
            cos, sin = self.forward(k, position_ids)
        else:
            # shape [B, T, E]
            cos, sin = position_embeddings

        if self.dispatchable:
            if cache is None:
                # No cache
                query_states, key_states = _MuiRotaryNoCache.apply(position_ids, cos, sin, q, k)
                # might need to return if if there is no cache
                value_states = v
            
            # Mui cache types
            elif isinstance(cache, MuiCache):
                return _MuiRotaryCacheModule.apply(self.cpp_module, cache.cpp_module, q, k, v, position_ids, (cos, sin), cache_position)
            # original HF cache types
            elif isinstance(cache, StaticCache):
                layer_idx = self.layer_idx
                if layer_idx == 0:
                    cache._seen_tokens = cache._seen_tokens + k.shape[-2]

                # Update the cache
                k_cache = cache.key_cache[layer_idx]
                v_cache = cache.value_cache[layer_idx]

                if cache_position is None:
                    raise ValueError("cache_position is needed")

                query_states, key_states, value_states = _MuiRotaryHFStaticCache.apply(position_ids, cos, sin, q, k, v, k_cache, v_cache, cache_position, cache._seen_tokens)

                return query_states, key_states, value_states
            elif isinstance(cache, DynamicCache):
                layer_idx = self.layer_idx
                if layer_idx == 0:
                    cache._seen_tokens = cache._seen_tokens + k.shape[-2]

                # Update the cache
                if len(cache.key_cache) <= layer_idx:
                    query_states, key_states = _MuiRotaryNoCache.apply(position_ids, cos, sin, q, k)
                    # we need the kv caches to be contiguous at all times, so need to make sure they are
                    # when we first create them
                    cache.key_cache.append(key_states.contiguous())
                    cache.value_cache.append(v.contiguous())
                else:
                    prev_k_cache = cache.key_cache[layer_idx]
                    prev_v_cache = cache.value_cache[layer_idx]

                    query_states, k_cache_out, v_cache_out = _MuiRotaryHFDynamicCache.apply(position_ids, cos, sin, q, k, v, prev_k_cache, prev_v_cache)
                    cache.key_cache[layer_idx] = k_cache_out
                    cache.value_cache[layer_idx] = v_cache_out

                key_states = cache.key_cache[layer_idx]
                value_states = cache.value_cache[layer_idx]

                return query_states, key_states, value_states
            else:
                # Not supporting this type of cache
                raise ValueError(f"Unsupported cache type: {type(cache).__name__}")

        else:
            query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin)
            # might need to return if if there is no cache
            value_states = v

        if cache is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = cache.update(key_states, v, self.layer_idx, cache_kwargs)

        return query_states, key_states, value_states