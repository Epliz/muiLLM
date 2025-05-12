from typing import List, Optional, Union
import warnings

from muillm.engineconfig import MuiEngineConfig
from muillm.modules.module import MuiModule
from muillm.synchronization.synchronizer import Synchronizer

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import (
    GenerationMixin,
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    validate_stopping_criteria,
    StoppingCriteriaList,
)

import torch
import torch.nn as nn

from muillm.sampling.multinomial import muillm_multinomial_sample_one_no_sync


from transformers.generation import GenerationMixin


class MuiGenerationMixin(MuiModule, GenerationMixin):
    def __init__(self, engine_config: MuiEngineConfig = None, **kargs):
        # engine_config is None when called from the parent class
        MuiModule.__init__(self, engine_config=engine_config, **kargs)

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        engine_config = self.engine_config

        synchronizer = engine_config.synchronizer
        comms = engine_config.comms
        tensor_parallelism = engine_config.tensor_parallelism

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )

        # parallelmodel needs to receive a list of caches
        # but HF gives a single one
        # until we figure out how to shard into several, remove the one passed by HF
        # (required to support static caches)
        if "mamba" in self.__class__.__name__.lower():
            cache_name = "cache_params"
        else:
            cache_name = "past_key_values"

        model_kwargs[cache_name] = None

        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # extract information about how many tokens we will generate max
        # so that we can sync GPU and CPU less but still every now and then
        # and respecting the max amount of tokens to generate
        max_remaining_generate = None
        max_length = stopping_criteria.max_length
        if max_length is not None:
            cur_len = input_ids.shape[-1]
            max_remaining_generate = max_length - cur_len

        last_sync = 0
        sync_frequency = 4

        # help removes a GPU sync point when preparing masks
        checked_mask_content = False
        self.all_ones_mask = None

        # Force a CPU GPU sync
        torch.cuda.synchronize()

        # All peers are syncrhonized by broadcasting the tokens, so they all finish at the same
        # time
        while not this_peer_finished:

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if not checked_mask_content:
                checked_mask_content = True

                if "attention_mask" in model_inputs:
                    attention_mask = model_inputs["attention_mask"]
                    # used to avoid some checks when updating masks later on
                    # as we generate we just add more ones, so the flag won't change
                    self.all_ones_mask = torch.all(attention_mask == 1).item()

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # avoid GPU sync
                next_tokens = muillm_multinomial_sample_one_no_sync(probs).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # broadcast the next_tokens from the rank 0 if we are using tensor parallelism
            if tensor_parallelism > 1:
                next_tokens = comms.broadcast(next_tokens, src=0)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                raise ValueError("Streaming is not supported at the moment")
                # streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )

            if max_remaining_generate is not None:
                max_remaining_generate = max_remaining_generate - 1

            last_sync = last_sync + 1
            if (last_sync >= sync_frequency) or (max_remaining_generate <= 0):
                last_sync = 0
                # make sure we get a python boolean out to avoid sync anytime we access
                # the variable
                if synchronizer is not None:
                    this_peer_finished = synchronizer.item(
                        unfinished_sequences.max() == 0
                    )
                else:
                    this_peer_finished = (unfinished_sequences.max() == 0).item()

                if sync_frequency < 16:
                    # decrease the sync frequency up to every 16 tokens
                    sync_frequency = sync_frequency * 2

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        # Force a CPU GPU sync
        torch.cuda.synchronize()

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
