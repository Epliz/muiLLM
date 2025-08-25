# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
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
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.cache_utils import Cache, HybridCache, StaticCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    is_torch_flex_attn_available,
    is_torchdynamo_compiling,
    logging,
)

from transformers import Gemma3Model
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3TextScaledWordEmbedding,
    Gemma3PreTrainedModel,
    Gemma3TextModel,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3MultiModalProjector,
    Gemma3CausalLMOutputWithPast,
    Gemma3ModelOutputWithPast,
)
from transformers.models.gemma3.configuration_gemma3 import (
    Gemma3Config,
    Gemma3TextConfig,
)

from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.attention.rotaryembedding import MuiRotaryEmbedding
from muillm.modules.decoder.gemma3decoder import MuiGemma3DecoderLayer

from muillm.modules.kvcache.cache_utils import (
    MuiCache,
    MuiHybridChunkedCache,
    create_hybrid_chunked_cache,
    grow_hybrid_chunked_cache_if_needed,
)
from muillm.modules.linear import MuiLinear
from muillm.modules.module import MuiModule
from muillm.modules.norm.rmsnorm import MuiRMSNorm
from muillm.modules.parallellinear import MuiParallelLinear
from muillm.replacement.replacementcontext import MuiReplacementContext
from muillm.sampling.generation import MuiGenerationMixin


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

logger = logging.get_logger(__name__)


class MuiGemma3TextModel(Gemma3PreTrainedModel, MuiModule):
    config_class = Gemma3TextConfig

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        config: Gemma3TextConfig,
        embed_tokens: Gemma3TextScaledWordEmbedding,
        layers,
        norm: MuiRMSNorm,
        rotary_emb: MuiRotaryEmbedding,
        rotary_emb_local: MuiRotaryEmbedding,
        initialize: bool = True,
    ):
        Gemma3PreTrainedModel.__init__(self, config)
        MuiModule.__init__(self, engine_config)

        # HF has a computed property called device, so we name ours "mdevice"
        self.mdevice = embed_tokens.weight.device
        self.mdtype = embed_tokens.weight.dtype

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.sliding_window = config.sliding_window

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = embed_tokens

        self.layers = layers
        self.norm = norm
        self.rotary_emb = rotary_emb
        self.gradient_checkpointing = False

        self.rotary_emb_local = rotary_emb_local

        # Initialize weights and apply final processing
        if initialize:
            self.post_init()

    @staticmethod
    def _replace_layers(
        replacement_context: MuiReplacementContext,
        prev_layers: nn.ModuleList,
    ) -> nn.ModuleList:
        engine_config = replacement_context.engine_config
        layers = nn.ModuleList()
        # replace the layers in reverse order to be able to remove them from the previous list
        for i in reversed(range(len(prev_layers))):
            decoder_layer = prev_layers[i]
            if engine_config.tensor_parallelism < 2:
                # no tensor parallelism
                layer = MuiGemma3DecoderLayer.replace(
                    replacement_context,
                    prev_module=decoder_layer,
                )
            else:
                # tensor parallelism
                raise NotImplementedError("not implemented for Gemma3 yet")

            layers.insert(0, layer)  # add to the begining of the list

            # delete the previous decoder layer to save memory
            del decoder_layer
            del prev_layers[i]

            # trigger GC to save memory
            trigger_gc()

        return layers

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_model: Union[Gemma3TextModel, "MuiGemma3TextModel"],
    ):
        if isinstance(prev_model, MuiGemma3TextModel):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_model
        engine_config = replacement_context.engine_config
        device = replacement_context.device

        config = prev_model.config

        # TODO: replace
        embed_tokens = prev_model.embed_tokens

        layers = MuiGemma3TextModel._replace_layers(
            replacement_context,
            prev_layers=prev_model.layers,
        )

        del prev_model.layers

        norm = MuiRMSNorm.replace(
            replacement_context,
            prev_module=prev_model.norm,
        )

        del prev_model.norm

        rotary_emb = MuiRotaryEmbedding.replace(
            replacement_context,
            prev_module=prev_model.rotary_emb,
        )

        rotary_emb_local = MuiRotaryEmbedding.replace(
            replacement_context,
            prev_module=prev_model.rotary_emb_local,
        )

        new_model = MuiGemma3TextModel(
            engine_config=engine_config,
            config=config,
            embed_tokens=embed_tokens,
            layers=layers,
            norm=norm,
            rotary_emb=rotary_emb,
            rotary_emb_local=rotary_emb_local,
            # we should not initialize weights as we use the already
            # loaded ones
            initialize=False,
        )

        return new_model

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
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

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        inputs_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size = inputs_shape[0]
        q_len = inputs_shape[1]

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        tot_seq_len = past_seen_tokens + q_len

        if use_cache:
            if isinstance(past_key_values, Cache) and not isinstance(
                past_key_values, MuiCache
            ):
                # If we have a cache, but not a MuiCache, drop it
                past_key_values = None

            no_cache = past_key_values is None
            if no_cache:
                # create a hybrid chunked cache (similar to HybridCache)
                past_key_values = create_hybrid_chunked_cache(
                    engine_config=self.engine_config,
                    config=self.config,
                    max_batch_size=batch_size,
                    seq_len=tot_seq_len,
                    device=self.mdevice,
                    dtype=self.mdtype,
                )
            else:
                # we have a previous cache, just re-use it
                if isinstance(past_key_values, MuiHybridChunkedCache):
                    # grow cache if needed
                    max_cache_length = self.config.max_position_embeddings
                    grow_hybrid_chunked_cache_if_needed(
                        past_key_values, tot_seq_len, max_cache_length
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

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        sliding_window_mask = self._create_sliding_window_mask(
            causal_mask,
            cache_position,
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    position_embeddings_global,
                    position_embeddings_local,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    sliding_window_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    sliding_window_mask=sliding_window_mask,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if use_cache:
            if isinstance(past_key_values, MuiCache):
                # the C++ module increase the seen tokens counts, and we need the python part to see it too
                past_key_values.sync_back()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @torch.no_grad()
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridCache,
        output_attentions: bool = False,
    ):
        # Flash Attention currently doesn't support static cache but Gemma3Text work only with static cache.
        # So we will pass in attention mask as is in any case, not only when ther's padding. Then we'll use its shape
        # to cut out keys/values trailing 0 used in static cache. This workaround should be compile compatible
        # as it doesn't cause dynamic control issues.
        if self.config._attn_implementation == "flash_attention_2":
            return attention_mask
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if isinstance(past_key_values, MuiHybridChunkedCache):
            # TODO: use minimal size
            target_length = past_key_values.get_max_cache_shape()
        if isinstance(past_key_values, (HybridCache, StaticCache)):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if attention_mask is not None
                else input_tensor.shape[
                    1
                ]  # TODO: + past_key_values.get_seq_length() + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        return causal_mask

    @torch.no_grad()
    def _create_sliding_window_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.LongTensor,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            # no attention mask, so no sliding window mask
            return None

        # efficient SDPA and no padding
        # In prefill, we may be larger than sliding window
        effective_seq_len = max(cache_position.shape[0], self.sliding_window)
        # For FA2, the mask is 2D and is of shape [bs, processed_tokens] (not [bs, max_cache_len]),
        # thus we must slice from the right (at most `effective_seq_len` elements)
        if self.config._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask[:, -effective_seq_len:]
        # Otherwise, the mask is 4D of shape [bs, 1, query_len, max_cache_len] thus we must slice
        # from the left, with an offset if we are beyond the sliding window
        else:
            min_dtype = torch.finfo(attention_mask.dtype).min
            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool),
                diagonal=-self.sliding_window,
            )
            attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
            # In case we are beyond the sliding window, we need to correctly offset the mask slicing
            offset = cache_position[-1] - effective_seq_len + 1
            # Should only be used when beyond the sliding window (i.e. offset > 0)
            offset = torch.clamp(offset, min=0)
            # equivalent to: `attention_mask = attention_mask[:, :, :, offset : offset + effective_seq_len]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indexes = torch.arange(
                min(effective_seq_len, attention_mask.shape[-1]),
                device=attention_mask.device,
            )
            mask_indexes += offset
            attention_mask = attention_mask[:, :, :, mask_indexes]

        return attention_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
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
                device=cache_position.device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=cache_position.device
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


class MuiGemma3ForCausalLM(Gemma3PreTrainedModel, MuiGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = Gemma3TextConfig
    base_model_prefix = "language_model"

    def __init__(
        self,
        model: MuiGemma3TextModel,
        lm_head: Union[MuiLinear, MuiParallelLinear],
        initialize: bool = True,
    ):
        MuiGenerationMixin.__init__(self, model.engine_config)
        # order matters: error if we call the llama constructor first
        Gemma3PreTrainedModel.__init__(self, model.config)

        self.model = model
        self.vocab_size = model.config.vocab_size
        self.lm_head = lm_head

        # Initialize weights and apply final processing
        if initialize:
            self.post_init()

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_model: Union["MuiGemma3ForCausalLM", Gemma3ForCausalLM],
    ) -> "MuiGemma3ForCausalLM":
        if isinstance(prev_model, MuiGemma3ForCausalLM):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_model

        engine_config = replacement_context.engine_config

        # replace the LM head first to potentially save memory
        if engine_config.tensor_parallelism < 2:
            lm_head = MuiLinear.replace(
                replacement_context,
                prev_model.lm_head,
            )
        else:
            lm_head = MuiParallelLinear.replace(
                replacement_context,
                prev_model.lm_head,
            )

        del prev_model.lm_head

        model = MuiGemma3TextModel.replace(
            replacement_context,
            prev_model.model,
        )

        del prev_model.model

        new_model = MuiGemma3ForCausalLM(
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

        >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
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
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **loss_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten: has a special cache type, `HybridCache`

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # print(f"model_inputs keys: {list(model_inputs.keys())}")

        if logits_to_keep is None:
            _ = model_inputs.pop("logits_to_keep", None)

        if (
            (
                isinstance(past_key_values, HybridCache)
                or isinstance(past_key_values, MuiHybridChunkedCache)
            )
            and attention_mask.ndim == 2
            and not self.config._attn_implementation == "flash_attention_2"
        ):
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            attention_mask = (
                self.model._prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.lm_head.weight.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                )
            )
            model_inputs["attention_mask"] = attention_mask

        return model_inputs


class MuiGemma3Model(Gemma3PreTrainedModel, MuiModule):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        config: Gemma3Config,
        vision_tower,
        multi_modal_projector: Gemma3MultiModalProjector,
        language_model: MuiGemma3TextModel,
        initialize: bool = True,
    ):
        Gemma3PreTrainedModel.__init__(self, config)
        MuiModule.__init__(self, engine_config)

        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector
        self.vocab_size = config.text_config.vocab_size

        self.language_model = language_model

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        if initialize:
            self.post_init()

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_model: Union["MuiGemma3Model", Gemma3Model],
    ) -> "MuiGemma3Model":
        if isinstance(prev_model, MuiGemma3Model):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_model

        engine_config = replacement_context.engine_config
        config = prev_model.config

        vision_tower = prev_model.vision_tower

        multi_modal_projector = prev_model.multi_modal_projector

        language_model = MuiGemma3TextModel.replace(
            replacement_context,
            prev_model.language_model,
        )

        new_model = MuiGemma3Model(
            engine_config=engine_config,
            config=config,
            vision_tower=vision_tower,
            multi_modal_projector=multi_modal_projector,
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

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_tensor,
        is_training: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            return attention_mask

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted
            # form and requires no inversion or slicing.
            return attention_mask

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, MuiHybridChunkedCache):
            # TODO: use minimal size
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=self.dtype,
            device=cache_position.device,
        )

        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)

        causal_mask *= torch.arange(
            target_length, device=cache_position.device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)

        # Apply bidirectional mask on images if token type ids are provided
        if token_type_ids is not None and sequence_length != 1:
            token_type_mask = token_type_ids.unsqueeze(1) == token_type_ids.unsqueeze(2)
            token_type_mask[token_type_ids == 0] = (
                False  # if text token do not change anything
            )

            # Find where a new image block starts: 1 if image and previous not image
            # The images cannot attend to future images, but can attend to all prev images and to itself bidirectionally
            is_image = token_type_ids == 1
            new_image_start = (
                is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
            )
            image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
            image_group_ids = torch.where(
                is_image, image_group_ids, torch.full_like(token_type_ids, -1)
            )

            same_image_mask = image_group_ids.unsqueeze(1) == image_group_ids.unsqueeze(
                2
            )
            same_image_mask[image_group_ids == -1] = False  # remove non-image
            image_mask = (
                (token_type_mask & same_image_mask)
                .unsqueeze(1)
                .to(causal_mask.device, dtype=torch.bool)
            )

            causal_mask = causal_mask.clone()
            causal_mask[:, :, :, :sequence_length] = causal_mask[
                :, :, :, :sequence_length
            ].masked_fill(image_mask, 0.0)

        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]

            # Then apply padding mask (will mask pad tokens)
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                :, None, None, :
            ].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)

        return causal_mask

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **lm_kwargs,
    ) -> Union[Tuple, Gemma3ModelOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma32-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/gemma32-3b-mix-224")

        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
        ```"""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

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

        is_training = token_type_ids is not None and labels is not None

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(
                        self.config.image_token_id,
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                )
            else:
                special_image_mask = (
                    input_ids == self.config.image_token_id
                ).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                    inputs_embeds.device
                )

            if (
                not is_torchdynamo_compiling()
                and inputs_embeds[special_image_mask].numel() != image_features.numel()
            ):
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        causal_mask = self._update_causal_mask(
            attention_mask,
            token_type_ids,
            past_key_values,
            cache_position,
            inputs_embeds,
            is_training,
        )
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **lm_kwargs,
        )

        return Gemma3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class MuiGemma3ForConditionalGeneration(Gemma3PreTrainedModel, MuiGenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        model: MuiGemma3Model,
        lm_head: Union[MuiLinear, MuiParallelLinear],
        initialize: bool = True,
    ):
        MuiGenerationMixin.__init__(self, model.engine_config)
        # order matters: error if we call the gemma constructor first
        Gemma3PreTrainedModel.__init__(self, model.config)

        self.model = model
        self.lm_head = lm_head

        if initialize:
            # Initialize weights and apply final processing
            self.post_init()

    @staticmethod
    def replace(
        replacement_context: MuiReplacementContext,
        prev_model: Union[
            "MuiGemma3ForConditionalGeneration", Gemma3ForConditionalGeneration
        ],
    ) -> "MuiGemma3ForConditionalGeneration":
        if isinstance(prev_model, MuiGemma3ForConditionalGeneration):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_model

        engine_config = replacement_context.engine_config

        # replace the LM head first to potentially save memory
        if engine_config.tensor_parallelism < 2:
            lm_head = MuiLinear.replace(
                replacement_context,
                prev_model.lm_head,
            )
        else:
            lm_head = MuiParallelLinear.replace(
                replacement_context,
                prev_model.lm_head,
            )

        del prev_model.lm_head

        model = MuiGemma3Model.replace(
            replacement_context,
            prev_model.model,
        )

        new_model = MuiGemma3ForConditionalGeneration(
            model=model,
            lm_head=lm_head,
            # we should not initialize weights as we use the already
            # loaded ones
            initialize=False,
        )

        return new_model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Make modules available throught conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
        >>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

        >>> messages = [
        ...     {
        ...         "role": "system",
        ...         "content": [
        ...             {"type": "text", "text": "You are a helpful assistant."}
        ...         ]
        ...     },
        ...     {
        ...         "role": "user", "content": [
        ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "Where is the cat standing?"},
        ...         ]
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     tokenizer=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ...     add_generation_prompt=True
        ... )
        >>> # Generate
        >>> generate_ids = model.generate(**inputs)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
        ```
        """

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

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **lm_kwargs,
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
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(
                    logits.device
                )
                shift_logits = shift_logits[
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = shift_labels[
                    shift_attention_mask.to(shift_labels.device) != 0
                ].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # TODO: improve
        is_prefill = cache_position[0] == 0

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if is_prefill:
            model_inputs["pixel_values"] = pixel_values
        is_training = token_type_ids is not None and labels is not None
        if is_prefill and (
            isinstance(past_key_values, HybridCache)
            or isinstance(past_key_values, MuiHybridChunkedCache)
        ):
            input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            causal_mask = self.model._update_causal_mask(
                attention_mask,
                token_type_ids,
                past_key_values,
                cache_position,
                input_tensor,
                is_training,
            )
            model_inputs["attention_mask"] = causal_mask

        return model_inputs

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
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
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        # print(
        #     f"MuiGemma3ForConditionalGeneration._prepare_4d_causal_attention_mask_with_cache_position"
        # )
        # if attention_mask is not None:
        #     print(f"  attention_mask: {attention_mask.shape}")
        # print(f"  sequence_length: {sequence_length} target_length: {target_length}")

        if attention_mask is not None:
            # because we are growing the cache in forward(), which is called after preparing inputs
            # (which is when this method is called), we need to ensure that the
            # target_length is correct according to the mask length
            mask_length = attention_mask.shape[-1]
            target_length = max(target_length, mask_length)

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=cache_position.device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=cache_position.device
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
