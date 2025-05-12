# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.attention.rotaryembedding import MuiRotaryEmbedding
from muillm.modules.decoder.llama4decoder import MuiLlama4TextDecoderLayer
from muillm.modules.decoder.parallelllama4decoder import (
    MuiParallelLlama4TextDecoderLayer,
)
from muillm.modules.linear import MuiLinear
from muillm.modules.module import MuiModule

from transformers.cache_utils import Cache, HybridChunkedCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)

from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)

from transformers.models.llama4.modeling_llama4 import (
    Llama4PreTrainedModel,
    Llama4TextModel,
    Llama4ForCausalLM,
    Llama4ForConditionalGeneration,
    Llama4CausalLMOutputWithPast,
    LLAMA4_INPUTS_DOCSTRING,
)

from muillm.modules.parallellinear import MuiParallelLinear
from muillm.modules.rmsnorm import MuiRMSNorm
from muillm.sampling.generation import MuiGenerationMixin

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Llama4Config"


class MuiLlama4TextModel(Llama4PreTrainedModel, MuiModule):
    _no_split_modules = ["Llama4TextDecoderLayer"]
    base_model_prefix = "model"
    config_class = Llama4TextConfig

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        config: Llama4TextConfig,
        embed_tokens,
        layers,
        norm: MuiRMSNorm,
        rotary_emb: MuiRotaryEmbedding,
        initialize: bool = True,
    ):
        Llama4PreTrainedModel.__init__(self, config)
        MuiModule.__init__(self, engine_config)

        self.cpp_engine = engine_config.cpp_engine
        self.comms = engine_config.comms

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = embed_tokens
        self.layers = layers
        self.norm = norm
        self.rotary_emb = rotary_emb
        self.gradient_checkpointing = False

        # the cpp module will be created at the end of all layer replacements
        self.cpp_module = None

        # Initialize weights and apply final processing
        if initialize:
            self.post_init()

    @staticmethod
    def _replace_layers(
        prev_layers: nn.ModuleList,
        engine_config: MuiEngineConfig,
        device=None,
    ) -> nn.ModuleList:
        layers = nn.ModuleList()

        for i in range(len(prev_layers)):
            decoder_layer = prev_layers[i]

            if engine_config.tensor_parallelism < 2:
                # no tensor parallelism
                layer = MuiLlama4TextDecoderLayer.replace(
                    prev_module=decoder_layer,
                    engine_config=engine_config,
                    device=device,
                )
            else:
                # tensor parallelism
                layer = MuiParallelLlama4TextDecoderLayer.replace(
                    prev_module=decoder_layer,
                    engine_config=engine_config,
                    device=device,
                )

            layers.append(layer)

            prev_layers[i] = None

            # trigger GC to save memory
            trigger_gc()

        return layers

    @staticmethod
    def replace(
        prev_model: Union["MuiLlama4TextModel", Llama4TextModel],
        engine_config: MuiEngineConfig,
        device=None,
    ) -> "MuiLlama4TextModel":
        if isinstance(prev_model, MuiLlama4TextModel):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_model

        config = prev_model.config
        embed_tokens = prev_model.embed_tokens
        prev_layers = prev_model.layers
        prev_norm = prev_model.norm

        # we will re-create it from scratch
        prev_model.rotary_emb = None

        # delete the previous module to save memory
        del prev_model

        # trigger GC to save memory
        trigger_gc()

        layers = MuiLlama4TextModel._replace_layers(
            prev_layers=prev_layers,
            engine_config=engine_config,
            device=device,
        )

        prev_layers = None

        # trigger GC to save memory
        trigger_gc()

        norm = MuiRMSNorm.replace(
            prev_module=prev_norm, engine_config=engine_config, device=device
        )

        rotary_emb = MuiRotaryEmbedding(
            engine_config=engine_config, config=config, layer_idx=0, device=device
        )

        new_model = MuiLlama4TextModel(
            engine_config=engine_config,
            config=config,
            embed_tokens=embed_tokens,
            layers=layers,
            norm=norm,
            rotary_emb=rotary_emb,
            # we should not initialize weights as we use the already
            # loaded ones
            initialize=False,
        )

        return new_model

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA4_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(
                input_ids.to(self.embed_tokens.weight.device)
            )

        if use_cache and past_key_values is None:
            past_key_values = HybridChunkedCache(
                self.config, inputs_embeds.shape[0], inputs_embeds.shape[1]
            )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask, chunk_causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
            use_cache=use_cache,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        freq_cis = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    chunk_causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    False,  # output_router_logits is False
                    use_cache,
                    cache_position,
                    freq_cis,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    chunk_causal_mask=chunk_causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=freq_cis,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    @torch.compiler.disable(
        recursive=False
    )  # the operations in this method are not compilable
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
        chunked_attention_mask=None,
        use_cache=True,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return (
                    attention_mask,
                    attention_mask,
                )  # flash does not support chunked attn TODO support flash
            return None, None

        if self.config._attn_implementation not in ["sdpa", "flex_attention", "eager"]:
            return None, None

        sequence_length = input_tensor.shape[1]
        cache_position = cache_position.to(self.device)
        attention_chunk_size = self.config.attention_chunk_size

        first_cache_position = cache_position[0]

        if past_key_values is not None:
            full_cache_length = past_key_values.get_max_cache_shape() or sequence_length
        else:
            full_cache_length = (
                attention_mask.shape[-1]
                if attention_mask is not None
                else sequence_length
            )

        cond1 = first_cache_position >= attention_chunk_size
        cond2 = (first_cache_position < attention_chunk_size) & (
            first_cache_position + sequence_length > attention_chunk_size
        )
        key_length = (
            torch.where(
                cond1,
                attention_chunk_size + sequence_length - 1,
                torch.where(
                    cond2, first_cache_position + sequence_length, attention_chunk_size
                ),
            )
            if use_cache
            else full_cache_length
        )

        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                offsets = (
                    first_cache_position,
                    max(first_cache_position - attention_chunk_size + 1, 0),
                )
                chunked_attention_mask = make_flex_block_causal_mask(
                    attention_mask,
                    self.config.attention_chunk_size,
                    sequence_length,
                    key_length,
                    offsets=offsets,
                )
                attention_mask = make_flex_block_causal_mask(
                    attention_mask,
                    query_length=sequence_length,
                    key_length=full_cache_length,
                    offsets=(first_cache_position, 0),
                )
                return attention_mask, chunked_attention_mask
            if isinstance(attention_mask, BlockMask):
                return attention_mask, chunked_attention_mask

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        dtype, device = input_tensor.dtype, input_tensor.device
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=max(full_cache_length, attention_chunk_size),
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        if full_cache_length > self.config.attention_chunk_size:
            start_idx = max(first_cache_position - attention_chunk_size + 1, 0)
            end_idx = start_idx + key_length
            chunked_attention_mask = self.create_chunked_attention_mask(
                self.config.attention_chunk_size,
                start=start_idx,  # same offset as with flex
                end=end_idx,
                device=device,
            )

            local_attention_mask = attention_mask[
                :, start_idx:end_idx
            ]  # offset here as well
            # It may be smaller than attention_chunk_size -> pad it
            requires_padding = local_attention_mask.shape[-1] < attention_chunk_size
            if requires_padding:
                local_attention_mask = nn.functional.pad(
                    local_attention_mask,
                    (0, attention_chunk_size - local_attention_mask.shape[-1]),
                )
            # Depending on the padding, take the query tokens from the end or the cache_position
            if not requires_padding:
                chunked_attention_mask = chunked_attention_mask[
                    None, None, -sequence_length:, :
                ]
            else:
                chunked_attention_mask = chunked_attention_mask[
                    None, None, cache_position, :
                ]

            chunked_attention_mask = chunked_attention_mask.expand(
                input_tensor.shape[0], -1, -1, -1
            )
            chunked_attention_mask = (
                chunked_attention_mask * local_attention_mask[:, None, None, :]
            )
            if self.config._attn_implementation == "eager":
                min_dtype = torch.finfo(dtype).min
                chunked_attention_mask = torch.where(
                    chunked_attention_mask == 0, min_dtype, 0.0
                ).to(dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and attention_mask.ndim == 4
            and not output_attentions  # Only unmask for 4d masks
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and chunked_attention_mask is not None
        ):
            chunked_attention_mask = chunked_attention_mask.bool()
            causal_mask = causal_mask.bool()
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=first_cache_position,
                is_training=self.training,
            ):
                causal_mask = None
        return causal_mask, chunked_attention_mask

    def create_chunked_attention_mask(
        self, attention_chunk_size: int, start: int, end: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate the following:

        'What'      :  0 ■ ⬚ ⬚ ⬚ ⬚ ⬚    |
        '▁is'       :  1 ■ ■ ⬚ ⬚ ⬚ ⬚     |
        '▁ch'       :  2 ■ ■ ■ ⬚ ⬚ ⬚     |
        'unked'     :  3 ⬚ ⬚ ⬚ ■ ⬚ ⬚    |
        '▁attention':  4 ⬚ ⬚ ⬚ ■ ■ ⬚    |
        '?'         :  5 ⬚ ⬚ ⬚ ■ ■ ■     |

        If the chunk size is 3.
        This can just be appplied over the already created attention mask
        """
        arange_vector = torch.arange(start, end, device=device)
        block_pos = torch.abs(
            arange_vector.unsqueeze(0) // attention_chunk_size
            - arange_vector.unsqueeze(1) // attention_chunk_size
        )
        token_pos = arange_vector.unsqueeze(0) - arange_vector.unsqueeze(1)
        mask = (block_pos == 0) & (token_pos <= 0)
        return mask.to(device)

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.to(device).reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        return causal_mask


class MuiLlama4ForCausalLM(Llama4PreTrainedModel, MuiGenerationMixin):
    base_model_prefix = "language_model"
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    config_class = Llama4TextConfig

    def __init__(
        self,
        model: MuiLlama4TextModel,
        lm_head: Union[MuiLinear, MuiParallelLinear],
        initialize: bool = True,
    ):
        MuiGenerationMixin.__init__(self, model.engine_config)
        # order matters: error if we call the llama constructor first
        Llama4PreTrainedModel.__init__(self, model.config)

        self.model = model
        self.vocab_size = model.vocab_size
        self.lm_head = lm_head

        # Used to avoid checking the mask over and over in _prepare_4d_causal_attention_mask_for_sdpa
        # set by _less_sync_sample in wrappedtransformers
        self.all_ones_mask = None

        # Initialize weights and apply final processing
        if initialize:
            self.post_init()

    def replace(
        prev_model: Union["MuiLlama4ForCausalLM", Llama4ForCausalLM],
        engine_config: MuiEngineConfig,
        device=None,
    ) -> "MuiLlama4ForCausalLM":
        if isinstance(prev_model, MuiLlama4ForCausalLM):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_model

        prev_lm_head = prev_model.lm_head
        prev_model_model = prev_model.model

        # delete the previous module to save memory
        del prev_model

        # trigger GC to save memory
        trigger_gc()

        # replace the LM head first to potentially save memory
        if engine_config.tensor_parallelism < 2:
            lm_head = MuiLinear.replace(
                prev_module=prev_lm_head,
                engine_config=engine_config,
                device=device,
            )
        else:
            lm_head = MuiParallelLinear.replace(
                prev_module=prev_lm_head,
                engine_config=engine_config,
                device=device,
            )

        model = MuiLlama4TextModel.replace(
            prev_model=prev_model_model, engine_config=engine_config, device=device
        )

        new_model = MuiLlama4ForCausalLM(
            model=model,
            lm_head=lm_head,
            # we should not initialize weights as we use the already
            # loaded ones
            initialize=False,
        )

        return new_model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA4_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Llama4ForCausalLM

        >>> model = Llama4ForCausalLM.from_pretrained("meta-llama4/Llama4-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama4/Llama4-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MuiLlama4ForConditionalGeneration(Llama4PreTrainedModel, MuiGenerationMixin):
    _tp_plan = {}
    base_model_prefix = ""
    config_class = Llama4Config
    _supports_flex_attn = True

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        model: Llama4ForConditionalGeneration,
        language_model: MuiLlama4ForCausalLM,
        initialize: bool = True,
    ):
        MuiGenerationMixin.__init__(self, engine_config)
        # order matters: error if we call the llama constructor first
        Llama4PreTrainedModel.__init__(self, model.config)

        self.vision_model = model.vision_model

        self.multi_modal_projector = model.multi_modal_projector
        self.language_model = language_model
        self.vocab_size = model.vocab_size

        self.pad_token_id = model.pad_token_id

        # Initialize weights and apply final processing
        if initialize:
            self.post_init()

    def replace(
        prev_model: Union[
            "MuiLlama4ForConditionalGeneration", Llama4ForConditionalGeneration
        ],
        engine_config: MuiEngineConfig,
        device=None,
    ) -> "MuiLlama4ForCausalLM":
        if isinstance(prev_model, MuiLlama4ForConditionalGeneration):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_model

        prev_language_model = prev_model.language_model
        prev_model.language_model = None

        language_model = MuiLlama4ForCausalLM.replace(
            prev_model=prev_language_model,
            engine_config=engine_config,
            device=device,
        )

        del prev_language_model

        # trigger GC to save memory
        trigger_gc()

        new_model = MuiLlama4ForConditionalGeneration(
            engine_config=engine_config,
            model=prev_model,
            language_model=language_model,
            # we should not initialize weights as we use the already
            # loaded ones
            initialize=False,
        )

        return new_model

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply al projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, List[int]]`):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                f"Unexpected select feature strategy: {self.vision_feature_select_strategy}"
            )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self.vision_model(
            pixel_values, output_hidden_states=False, **kwargs
        )
        hidden_state = image_outputs.last_hidden_state
        return hidden_state

    @replace_return_docstrings(
        output_type=Llama4CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: torch.Tensor = None,
        **lm_kwargs,
    ) -> Union[Tuple, Llama4CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )
            original_inputs_embeds_shape = inputs_embeds.shape

            vision_flat = image_features.view(-1, image_features.size(-1))
            projected_vision_flat = self.multi_modal_projector(vision_flat)

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(
                -1
            )
            final_mask = special_image_mask.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1))

            final_mask_1d = final_mask[..., 0].reshape(-1)
            num_tokens_to_fill = final_mask_1d.sum()

            if num_tokens_to_fill != projected_vision_flat.size(0):
                raise ValueError(
                    f"Mismatch: final_mask wants {num_tokens_to_fill} embeddings, "
                    f"but multi_modal_projector returned {projected_vision_flat.size(0)}"
                )

            expanded_mask = final_mask_1d.unsqueeze(-1).expand(
                -1, inputs_embeds.size(-1)
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                expanded_mask, projected_vision_flat
            )
            inputs_embeds = inputs_embeds.view(original_inputs_embeds_shape)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(
                    logits.device
                )
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Llama4CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        return causal_mask
