from typing import Optional, Tuple
from muillm.engineconfig import MuiEngineConfig
from muillm.muimodule import MuiModule
import torch
import torch.nn as nn

from transformers.cache_utils import Cache, DynamicCache

import muillm_ext

    
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# TODO: custom kernel for BS=1, S=1 case at least
@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim:int=1):
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
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


class _MuiRotaryDynamicCache(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions_ids, cos_cached, sin_cached, q, k, v, prev_k_cache, prev_v_cache):
        output = muillm_ext.muillm_rope_forward_dynamic_cache(positions_ids, cos_cached, sin_cached, q, k, v, prev_k_cache, prev_v_cache)

        ctx.save_for_backward(positions_ids, cos_cached, sin_cached, q, k)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("rotary backward not implemented")

class MuiMistralRotaryEmbedding(MuiModule):
    def __init__(self, engine_config: MuiEngineConfig, dim, max_position_embeddings=2048, base=10000, layer_idx: int = None, device=None, dtype=None):
        super().__init__(engine_config=engine_config)

        self.layer_idx = layer_idx

        dtype = dtype if dtype is not None else torch.get_default_dtype()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=dtype
        )

        dispatchable_type = (dtype == torch.float16)
        dispatchable_device = inv_freq.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            # TODO: amortize resizing?
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
    
    def apply_rotary_pos_emb_write_kv_cache(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, kv_seq_len: int, v: Optional[torch.Tensor] = None, cache: Optional[Cache] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cos, sin = self.forward(k, seq_len=kv_seq_len)

        if self.dispatchable:
            if isinstance(cache, DynamicCache):
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

                    query_states, key_states, k_cache_out, v_cache_out = _MuiRotaryDynamicCache.apply(position_ids, cos, sin, q, k, v, prev_k_cache, prev_v_cache)
                    cache.key_cache[layer_idx] = k_cache_out
                    cache.value_cache[layer_idx] = v_cache_out

                key_states = cache.key_cache[layer_idx]
                value_states = cache.value_cache[layer_idx]

                return query_states, key_states, value_states
            else:
                # Not supporting this type of cache
                query_states, key_states = _MuiRotaryNoCache.apply(position_ids, cos, sin, q, k)
                # might need to return if if there is no cache
                value_states = v

        else:
            query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            # might need to return if if there is no cache
            value_states = v

        if cache is not None:
            # TODO: bug here static cache not getting cache_position, need to get latest transformer library
            #cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            cache_kwargs = {}
            key_states, value_states = cache.update(key_states, v, self.layer_idx, cache_kwargs)

        return query_states, key_states, value_states


    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, kv_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, _ = self.apply_rotary_pos_emb_write_kv_cache(q, k, position_ids, kv_seq_len, v=None, cache=None)
        return query_states, key_states