from typing import List, Optional, Union
import warnings

from muillm.engineconfig import MuiEngineConfig
from muillm.commmunication.communicator import Communicator
from muillm.synchronization.synchronizer import Synchronizer
from transformers.generation.utils import GenerationMixin, GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.streamers import BaseStreamer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import validate_stopping_criteria, StoppingCriteriaList

import torch
import torch.nn as nn
import torch.distributed as dist

from muillm.sampling.multinomial import muillm_multinomial_sample_one_no_sync

# HuggingFace Transformers uses torch.multinomial that causes quite a lot of GPU sync
# so re-implement it to avoid it
def _less_sync_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional[BaseStreamer] = None,
    synchronizer: Synchronizer = None,
    communicator: Communicator = None,
    tensor_parallelism:int = 1,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    
    if tensor_parallelism > 1:
        # we are using tensor parallelism, so we need to sync the GPUs
        synced_gpus = True

        # determine what rank we are
        rank = dist.get_rank()
    else:
        rank = 0

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]


    this_peer_finished = False
    has_unfinished_sequences = True

    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

    # extract information about how many tokens we will generate max
    # so that we can sync GPU and CPU less but still every now and then
    # and respecting the max amount of tokens to generate 
    max_remaining_generate = None
    max_length = stopping_criteria.max_length
    if max_length is not None:
        cur_len = input_ids.shape[-1]
        max_remaining_generate = max_length - cur_len

    last_sync = 0
    sync_frequency = 32
    max_sync_frequency = 128

    # help removes a GPU sync point when preparing masks
    checked_mask_content = False
    self.all_ones_mask = None

    while has_unfinished_sequences:
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)


        if not checked_mask_content:
            checked_mask_content = True

            if "attention_mask" in model_inputs:
                attention_mask = model_inputs["attention_mask"]
                # used to avoid some checks when updating masks later on
                # as we generate we just add more ones, so the flag won't change
                self.all_ones_mask = torch.all(attention_mask == 1).item()

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # # This code was in HF Transformers
        # # But for us, we can't actually skip the rest if we are using tensor parallelism
        # # As we need all GPUs to reach the broadcast and so on
        # # We could skip the prob computations, but we don't right now
        # if synced_gpus and this_peer_finished:
        #     continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # sample
        # (when we have multiple GPUs, we don't really need to do that on all GPUs, but it keeps
        # the execution more similar)
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        #next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        # avoid GPU sync
        next_tokens = muillm_multinomial_sample_one_no_sync(probs).squeeze(1)

        if tensor_parallelism > 1:
            # synchronize the samples token to make sure all GPUs get the same output
            communicator.broadcast(next_tokens, src=0)
            #dist.broadcast(next_tokens, src=0)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)


        if max_remaining_generate is not None:
            max_remaining_generate = max_remaining_generate - 1

        last_sync = last_sync + 1
        if (last_sync >= sync_frequency) or (max_remaining_generate <= 0):
            # in this block, we basically block and check if the generation finished
            # we do it not for every token so that we can enqueue more kernels ahead

            last_sync = 0
            # make sure we get a python boolean out to avoid sync anytime we access
            # the variable

            # this contains "True" if this GPU has finished
            this_peer_finished_gpu = (unfinished_sequences.max() == 0)

            if synchronizer is not None:
                this_peer_finished = synchronizer.item(this_peer_finished_gpu)
            else:
                this_peer_finished = this_peer_finished_gpu.item()

            # Also refresh that one, that is conditioning the looping
            # HF Transformers was doing an all reduce in _has_unfinished_sequences()
            # But because we broadcast the same tokens, this_peer_finished should always be the same
            # on all GPUs
            # TODO: revisit when we support other inference modes than tensor parallelism
            has_unfinished_sequences = not this_peer_finished

            if sync_frequency < max_sync_frequency:
                # decrease the sync frequency up to every 16 tokens
                sync_frequency = sync_frequency * 2


    # Reset just in case we generate without this method next time
    self.all_ones_mask = False

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


def _wrap_transformers_model(model: GenerationMixin, engine_config: MuiEngineConfig) -> GenerationMixin:
    # monkey patch
    model._sample_bak = model._sample

    def _less_sync_sample_binder(*args, **kwargs):
        return _less_sync_sample(
            model,
            synchronizer = engine_config.synchronizer,
            communicator = engine_config.communicator,
            tensor_parallelism=engine_config.tensor_parallelism,
            *args,
            **kwargs
        )

    model._sample = _less_sync_sample_binder

    return model