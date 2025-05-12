# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from muillm.engineconfig import MuiEngineConfig
from muillm.memorymanagement.gc import trigger_gc
from muillm.modules.attention.rotaryembedding import MuiRotaryEmbedding
from muillm.modules.attention.sdpaattention import _ignore_causal_mask_sdpa
from muillm.modules.decoder.decoder import MuiDecoderLayer
from muillm.modules.decoder.paralleldecoder import MuiParallelDecoderLayer
from muillm.modules.decoder.paralleldecoderstack import _MuiParallelDecoderStack
from muillm.modules.kvcache.cache_utils import (
    MuiCache,
    MuiDynamicCache,
    MuiStaticCache,
    create_static_cache,
    grow_static_cache_if_needed,
)
from muillm.modules.linear import MuiLinear
from muillm.modules.module import MuiModule

from muillm.modules.parallellinear import MuiParallelLinear
from muillm.modules.rmsnorm import MuiRMSNorm
import muillm_ext

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaForCausalLM,
    LLAMA_INPUTS_DOCSTRING,
    LLAMA_START_DOCSTRING,
)

from muillm.sampling.generation import MuiGenerationMixin


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class MuiLlamaModel(LlamaPreTrainedModel, MuiModule):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(
        self,
        engine_config: MuiEngineConfig,
        config: LlamaConfig,
        embed_tokens,
        layers,
        norm: MuiRMSNorm,
        rotary_emb: MuiRotaryEmbedding,
        initialize: bool = True,
    ):
        LlamaPreTrainedModel.__init__(self, config)
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

    def finalize_init(self):
        if self.comms == None:
            # in the single GPU case we don't have comms, and we can't use the parallel decoder stack
            self.cpp_module = None
            return

        if self.cpp_module is not None:
            muillm_ext.muillm_parallel_decoder_stack_deinit(self.cpp_module)

        self.cpp_module = muillm_ext.muillm_parallel_decoder_stack_init(
            self.cpp_engine,
            self.comms.comms,
            [layer.cpp_module for layer in self.layers],
        )

    def replace(
        prev_model: Union["MuiLlamaModel", LlamaModel],
        engine_config: MuiEngineConfig,
        device=None,
    ) -> "MuiLlamaModel":
        if isinstance(prev_model, MuiLlamaModel):
            # nothing would be changing if we created a new module, so might as well return the previous one
            return prev_model

        config = prev_model.config
        embed_tokens = prev_model.embed_tokens
        prev_layers = prev_model.layers
        prev_norm = prev_model.norm

        # delete the previous module to save memory
        del prev_model

        # trigger GC to save memory
        trigger_gc()

        if engine_config.tensor_parallelism < 2:
            # no tensor parallelism
            layers = nn.ModuleList(
                [
                    MuiDecoderLayer.replace(
                        prev_module=decoder_layer,
                        engine_config=engine_config,
                        device=device,
                    )
                    for decoder_layer in prev_layers
                ]
            )
        else:
            # tensor parallelism
            layers = nn.ModuleList(
                [
                    MuiParallelDecoderLayer.replace(
                        prev_module=decoder_layer,
                        engine_config=engine_config,
                        device=device,
                    )
                    for decoder_layer in prev_layers
                ]
            )

        norm = MuiRMSNorm.replace(
            prev_module=prev_norm, engine_config=engine_config, device=device
        )
        rotary_emb = MuiRotaryEmbedding(
            engine_config=engine_config, config=config, layer_idx=0, device=device
        )

        new_model = MuiLlamaModel(
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

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        all_ones_mask: Optional[bool] = None,
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
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size = inputs_embeds.shape[0]

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        tot_seq_len = past_seen_tokens + inputs_embeds.shape[1]

        if use_cache:
            no_cache = past_key_values is None
            use_legacy_cache = (not isinstance(past_key_values, Cache)) and (
                not no_cache
            )

            if no_cache:
                # create a cache from scratch
                device = inputs_embeds.device
                dtype = torch.float16
                past_key_values = create_static_cache(
                    self.engine_config,
                    self.config,
                    batch_size,
                    tot_seq_len,
                    device,
                    dtype,
                )
            elif use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            else:
                # we have a previous cache, just re-use it
                if isinstance(past_key_values, MuiStaticCache):
                    # grow cache if needed
                    max_cache_length = self.config.max_position_embeddings
                    grow_static_cache_if_needed(
                        past_key_values, tot_seq_len, max_cache_length
                    )

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, tot_seq_len, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
            all_ones_mask,
        )
        hidden_states = inputs_embeds

        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, :tot_seq_len]

        if all_ones_mask is None:
            # if not specified, assume it might not have just ones
            all_ones_mask = False
        if all_ones_mask:
            causal_mask = None

        position_ids = position_ids.contiguous()

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = cos, sin

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # determine if we can use the decoder stack
        bsz, q_len, _ = hidden_states.size()
        grad_checkpointing = self.gradient_checkpointing and self.training
        mui_cache = isinstance(past_key_values, MuiCache)
        dispatchable_dtype = hidden_states.dtype == torch.float16
        no_outputs = (not output_hidden_states) and (not output_attentions)
        dispatchable_input = (bsz == 1) and (q_len == 1) and dispatchable_dtype
        dispatchable_to_stack = (
            (self.cpp_module is not None)
            and no_outputs
            and (not grad_checkpointing)
            and mui_cache
            and dispatchable_input
        )

        if dispatchable_to_stack:
            hidden_states = _MuiParallelDecoderStack.apply(
                self.cpp_module,
                past_key_values.cpp_module,
                hidden_states,
                causal_mask,
                position_ids,
                position_embeddings,
                cache_position,
            )

            next_decoder_cache = past_key_values
        else:
            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if grad_checkpointing:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        all_ones_mask,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        all_ones_mask=all_ones_mask,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            if isinstance(next_decoder_cache, MuiCache):
                # the C++ module increase the seen tokens counts, and we need the python part to see it too
                next_decoder_cache.sync_back()

            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        all_ones_mask: Optional[bool] = None,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
            if _ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
                all_ones_mask=all_ones_mask,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if False:  # using_static_cache:
            # modification compared to normal HF transformers
            # we use the same normal code as for dynamic cache
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
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

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

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
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        return causal_mask


class MuiLlamaForCausalLM(LlamaPreTrainedModel, MuiGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        model: MuiLlamaModel,
        lm_head: Union[MuiLinear, MuiParallelLinear],
        initialize: bool = True,
    ):
        MuiGenerationMixin.__init__(self, model.engine_config)
        # order matters: error if we call the llama constructor first
        LlamaPreTrainedModel.__init__(self, model.config)

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
        prev_model: Union["MuiLlamaForCausalLM", LlamaForCausalLM],
        engine_config: MuiEngineConfig,
        device=None,
    ) -> "MuiLlamaForCausalLM":
        if isinstance(prev_model, MuiLlamaForCausalLM):
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

        model = MuiLlamaModel.replace(
            prev_model=prev_model_model, engine_config=engine_config, device=device
        )

        new_model = MuiLlamaForCausalLM(
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

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
        num_logits_to_keep: int = 0,
        all_ones_mask: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        if all_ones_mask is not None:
            self.all_ones_mask = all_ones_mask

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
            all_ones_mask=self.all_ones_mask,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            raise ValueError("Not supported")

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            raise ValueError("Not supported")

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {
                "input_ids": input_ids.contiguous()
            }  # `contiguous()` needed for compilation use cases

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )

        if self.all_ones_mask is not None:
            # was set externally, add it to the model inputs to avoid
            # calls to torch.all in _prepare_4d_causal_attention_mask_for_sdpa
            model_inputs["all_ones_mask"] = self.all_ones_mask

        return model_inputs
