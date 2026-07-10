from typing import Any, Dict, Optional, Union
from muillm.engineconfig import MuiEngineConfig
from muillm.modules.module import MuiModule
import torch

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama4.modeling_llama4 import Llama4TextRotaryEmbedding

import muillm_ext
from muillm.replacement.replacementcontext import MuiReplacementContext


class MuiRotaryEmbedding(MuiModule):
    def __init__(
        self,
        engine_config: MuiEngineConfig,
        config: Union[Gemma3Config, LlamaConfig, MistralConfig, Llama4TextConfig],
        rope_kwargs: Dict[str, Any] = None,
        layer_idx: int = 0,
        output_complex: Optional[bool] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(engine_config=engine_config)

        self.cpp_engine = engine_config.cpp_engine
        self.config = config
        self.rope_kwargs = {}

        self.output_complex = output_complex
        self.layer_idx = layer_idx

        # default to use use float32 for cos/sin caches
        dtype = dtype if dtype is not None else torch.float32

        if config is not None:
            if isinstance(config, LlamaConfig) or isinstance(config, MistralConfig):
                if output_complex is None:
                    # override the output_complex flag
                    output_complex = False
                    self.output_complex = False

                # BC: "rope_type" was originally "type"
                if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
                    self.rope_type = config.rope_scaling.get(
                        "rope_type", config.rope_scaling.get("type")
                    )
                else:
                    self.rope_type = "default"
            elif isinstance(config, Llama4TextConfig):
                if output_complex is None:
                    # override the output_complex flag
                    output_complex = True
                    self.output_complex = True

                if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
                    self.rope_type = "llama3"
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
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )

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
        dispatchable_type = (self.dtype == torch.float16) or (
            self.dtype == torch.bfloat16
        )
        dispatchable_device = self.inv_freq.is_cuda
        self.dispatchable = dispatchable_device and dispatchable_type

    def finalize_init(self):
        # cache the flags checking if it is dispatchable
        self._check_dispatchable()

        self._deinit_cpp_module()

        self.cpp_module = muillm_ext.muillm_rotary_embedding_module_init(
            self.cpp_engine,
            self.layer_idx,
            self.cos_cached,
            self.sin_cached,
        )

    def _deinit_cpp_module(self):
        if getattr(self, "cpp_module", None) is None:
            return

        deinit_fn = getattr(muillm_ext, "muillm_rotary_embedding_module_deinit", None)
        if callable(deinit_fn):
            deinit_fn(self.cpp_module)

        del self.cpp_module

    def finalize_deinit(self):
        self._deinit_cpp_module()

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_module: Union[
            Gemma3RotaryEmbedding, LlamaRotaryEmbedding, MistralRotaryEmbedding
        ],
    ) -> "MuiRotaryEmbedding":
        engine_config = replacement_context.engine_config
        device = replacement_context.device
        if device is None:
            raise ValueError("device was None")

        device = prev_module.inv_freq.device if device is None else device
        dtype = prev_module.inv_freq.dtype

        # we either need a model config for Llama or rope_kwargs
        config = None
        rope_kwargs = None
        if (
            isinstance(prev_module, Gemma3RotaryEmbedding)
            or isinstance(prev_module, LlamaRotaryEmbedding)
            or isinstance(prev_module, MistralRotaryEmbedding)
            or isinstance(prev_module, Llama4TextRotaryEmbedding)
        ):
            config = prev_module.config
        else:
            raise ValueError(
                f"Unsupported type of module: {prev_module.__class__.__name__}"
            )

        output_complex = isinstance(prev_module, Llama4TextRotaryEmbedding)

        new_module = MuiRotaryEmbedding(
            engine_config=engine_config,
            config=config,
            rope_kwargs=rope_kwargs,
            output_complex=output_complex,
            device=device,
            dtype=dtype,
        )

        return new_module

    def _set_cos_sin_cache(self, seq_len, device, dtype):

        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation

        # LLama 4 uses complex numbers and doesn't need this concatenation
        if not self.output_complex:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = freqs

        cos = emb.cos()
        sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)

    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        dtype = x.dtype

        if self.output_complex:
            # For Llama 4, always output complex floats
            cos = self.cos_cached[position_ids].to(dtype=torch.float32)
            sin = self.sin_cached[position_ids].to(dtype=torch.float32)
            return torch.complex(cos, sin)
        else:
            cos = self.cos_cached[position_ids].to(dtype=dtype)
            sin = self.sin_cached[position_ids].to(dtype=dtype)
            return cos, sin
