from contextlib import nullcontext
import json
import multiprocessing as mp
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from muillm.server.hfmessages import convert_to_hf_messages, convert_to_hf_tools

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from muillm.engine import init_engine
from muillm.server.chatcompletion import ChatCompletionFunctionTool, ChatCompletionJsonSchemaResponseFormat, ChatCompletionRequest, ChatCompletionResult, ChatMessage
from muillm.server.defaults import DEFAULT_MAX_CONTEXT_LENGTH, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from muillm.server.filehelpers import read_file_content
from muillm.server.idutils import generate_id
from muillm.server.outputparsers.outputparser import OutputParser


def detect_tp_size(requested: Optional[int]) -> int:
    if requested is not None and requested > 0:
        return requested
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1

def _messages_to_prompt(
        tokenizer: Any,
        messages: Sequence[ChatMessage],
        tools: Optional[List[ChatCompletionFunctionTool]],
        output_schema: Optional[str] = None
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                convert_to_hf_messages(messages),
                tools=convert_to_hf_tools(tools),
                output_schema=output_schema,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            pass

    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

def get_num_completions(payload: ChatCompletionRequest) -> int:
    if payload.n is not None and payload.n > 0:
        return payload.n

    return 1

def get_output_schema(payload: ChatCompletionRequest) -> Optional[str]:
    if payload.response_format is not None and isinstance(payload.response_format, ChatCompletionJsonSchemaResponseFormat):
        return json.dumps(payload.response_format.output_json_schema)

    return None

def get_structured_output_format(payload: ChatCompletionRequest) -> Optional[ChatCompletionJsonSchemaResponseFormat]:
    if payload.response_format is not None and isinstance(payload.response_format, ChatCompletionJsonSchemaResponseFormat):
        return payload.response_format

    return None

def validate_structured_output(content: str, output_json_schema: dict) -> None:
    from jsonschema import validate, ValidationError

    try:
        validate(instance=json.loads(content), schema=output_json_schema)
    except ValidationError as e:
        raise ValueError(f"Parsed content does not match the output schema: {e.message}")

def _generate(model: Any, tokenizer: Any, payloads: List[ChatCompletionRequest], device: torch.device, rank: int, profile: bool) -> List[str]:
    batch_size = len(payloads)

    # TODO: check that all generation args are the same
    generation_args = build_generation_args(payloads[0])

    # apply the chat template
    prompts = [
        _messages_to_prompt(
            tokenizer,
            payload.messages,
            payload.tools,
            get_output_schema(payload)
        )
        for payload in payloads
        for _ in range(get_num_completions(payload))
    ]

    print("-----")
    print(f"Flattened prompts:")
    for i, prompt in enumerate(prompts):
        print("--")
        print(f"Prompt {i}: {prompt}")
        print("--")

    start_time = time.time()

    profile_ctx = None
    try:
        with create_profiling_context(profile) as profile_ctx:
            with torch.no_grad():
                # Prompt already contains chat-template boundary tokens.
                inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding="longest", padding_side="left")
                inputs = inputs.to(device)
                prompt_len = inputs["input_ids"].shape[1]

                outputs = model.generate(
                    **inputs,
                    **generation_args,
                    do_sample=True,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
    finally:
        save_trace(profile_ctx, rank, batch_size)

    end_time = time.time()

    # Decode only generated continuation, not prompt + continuation.
    generated_ids = outputs[:, prompt_len:]
    texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

    output_len = generated_ids.shape[1]
    total_tokens = batch_size * output_len

    tokens_per_seconds = total_tokens / (end_time - start_time)
    print(f"Output tokens: {output_len} Batch size: {batch_size} Total tokens {total_tokens} ({tokens_per_seconds} total tokens/s)")

    print(f"Outputs:")
    for i, text in enumerate(texts):
        print("--")
        print(f"Output {i}: {text}")
        print("--")
    print("-----")

    # outputs will be parsed on rank 0
    offset = 0
    outputs = []
    for payload in payloads:
        n = get_num_completions(payload)
        outputs.append(texts[offset:offset + n])
        offset += n

    return outputs

def build_generation_args(payload: ChatCompletionRequest) -> Dict[str, Any]:
    max_tokens = payload.max_tokens
    temperature = payload.temperature
    top_p = payload.top_p

    generation_args = {}

    if max_tokens is not None:
        generation_args["max_new_tokens"] = max_tokens
    if temperature is not None:
        generation_args["temperature"] = temperature
    if top_p is not None:
        generation_args["top_p"] = top_p
    return generation_args

def create_profiling_context(profile: bool):

    if profile:
        print("Starting profiling...")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        profile_ctx = torch.profiler.profile(
            activities=activities,
        )
    else:
        profile_ctx = nullcontext()

    return profile_ctx

def save_trace(profile_ctx, rank: int, batch_size: int):
    if (profile_ctx is not None) and isinstance(profile_ctx, torch.profiler.profile):
        profile_output_dir = "profiler"
        os.makedirs(profile_output_dir, exist_ok=True)

        trace_id = generate_id("trace_", length=6)
        trace_file = os.path.join(profile_output_dir, f"trace_{trace_id}_bs{batch_size}_rank{rank}.json")

        print(f"Profiling trace saved to: {trace_file}")
        profile_ctx.export_chrome_trace(trace_file)


def _worker_entrypoint(
    rank: int,
    world_size: int,
    model_path: str,
    lora_path: Optional[str],
    tokenizer_path: str,
    chat_template_path: Optional[str],
    output_parser_name: Optional[str],
    model_dtype,
    request_queue: Any,
    response_queue: Any,
    ready_queue: Any,
    profile: bool,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if world_size > 1 and torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(tokenizer_path, chat_template_path)

    model = load_model(rank, model_path, lora_path, model_dtype, device)

    output_parser = create_output_parser(model, output_parser_name)

    ready_queue.put(rank)

    while True:
        payloads: List[ChatCompletionRequest] = request_queue.get()
        if payloads is None:
            break

        response_texts_per_request = _generate(model, tokenizer, payloads, device, rank, profile)

        if rank == 0:
            # Only rank 0 returns the result
            all_parsed_responses: List[List[ChatMessage]] = []
            for r, response_texts in enumerate(response_texts_per_request):
                payload = payloads[r]

                output_schema = get_structured_output_format(payload)
                strict_output_schema = output_schema.strict if output_schema is not None else False

                parsed_responses: List[ChatMessage] = []
                for response_text in response_texts:
                    parsed_response = output_parser.parse(response_text)

                    if strict_output_schema:
                        # check that the parsed response matches the output schema
                        validate_structured_output(parsed_response.content, output_schema.output_json_schema)

                    parsed_responses.append(parsed_response)

                print(f"Request {r} parsed responses: {parsed_responses}")
                all_parsed_responses.append(parsed_responses)

            all_responses = [
                ChatCompletionResult(
                    request_id=payload.request_id,
                    responses=parsed_responses
                )
                for payload, parsed_responses in zip(payloads, all_parsed_responses)
            ]

            response_queue.put(all_responses)

def create_output_parser(model, output_parser_name: Optional[str]) -> OutputParser:
    return OutputParser.create_output_parser(model.__class__.__name__, output_parser_name)

def load_model(rank, model_path, lora_path, model_dtype, device):
    print(f"Loading model on rank {rank}...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            tp_plan="auto",
            torch_dtype=model_dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path)

    if lora_path is not None:
        from peft import PeftModelForCausalLM

        print(f"Applying LoRA weights from {lora_path}...")
        model = PeftModelForCausalLM.from_pretrained(model, lora_path)

    if torch.cuda.is_available():
        # put the dtype again here in case LoRa weights were loaded in a different dtype
        model = model.to(device=device, dtype=model_dtype)

    if lora_path is not None:
        # merge the LoRA weights into the base model
        print(f"Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model = init_engine(model, tensor_parallelism=None)

    print(f"Model loaded on rank {rank}.")

    return model

def load_tokenizer(tokenizer_path, chat_template_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")

    if chat_template_path is not None:
        chat_template = read_file_content(chat_template_path)
        tokenizer.chat_template = chat_template

        print("---")
        print(f"Chat template:")
        print(chat_template)
        print("---")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class ModelWorkerManager:
    def __init__(self, args: Any, ctx: Optional[mp.context.BaseContext] = None) -> None:
        self.args = args
        self.ctx = ctx or mp.get_context("spawn")
        self.tp_size = detect_tp_size(args.tp_size)
        self.request_queues = [self.ctx.Queue() for _ in range(self.tp_size)]
        self.response_queue = self.ctx.Queue()
        self.ready_queue = self.ctx.Queue()
        self.processes: List[Any] = []
        self._next_queue = 0
        self._started = False
        self._dispatch_lock = threading.Lock()

        self.served_model_id = args.served_model_id or args.model_path
        self.model_path = args.model_path
        self.lora_path = args.lora_path
        # if a tokenizer path was not provided, use the model path as the tokenizer path
        self.tokenizer_path = args.tokenizer_path if args.tokenizer_path is not None else self.model_path
        self.chat_template_path = args.chat_template_path
        self.output_parser_name = args.output_parser

        self.profile = args.profile

        # load the model configuration to determine the maximum context length and output length
        model_config = AutoConfig.from_pretrained(args.model_path)

        model_dtype = model_config.torch_dtype if hasattr(model_config, "torch_dtype") else None
        model_context_size = model_config.max_position_embeddings if hasattr(model_config, "max_position_embeddings") else None

        default_max_context_size = model_context_size if model_context_size is not None else DEFAULT_MAX_CONTEXT_LENGTH

        self.model_dtype = model_dtype
        self.max_batch_size = args.max_batch_size if args.max_batch_size is not None else 8
        self.max_context_length = args.max_context_length if args.max_context_length is not None else default_max_context_size
        self.max_output_length = args.max_output_length if args.max_output_length is not None else DEFAULT_MAX_OUTPUT_TOKENS

        print("----")
        print(f"Initializing model with: ")
        print(f"served_model_id={self.served_model_id}")
        print(f"model_path={self.model_path}")
        if self.tokenizer_path is not None:
            print(f"tokenizer_path={self.tokenizer_path}")
        if self.chat_template_path is not None:
            print(f"chat_template_path={self.chat_template_path}")
        if self.output_parser_name is not None:
            print(f"output_parser={self.output_parser_name}")
        print(f"model_dtype={self.model_dtype}")
        print(f"tp_size={self.tp_size}")
        print(f"max_batch_size={self.max_batch_size}")
        print(f"max_context_length={self.max_context_length}")
        print(f"max_output_length={self.max_output_length}")
        print("----")

        if self.profile:
            print("Profiling is enabled. Profiling traces will be saved in the 'profiler' directory.")

    def max_tokens(self, requested: Optional[int]) -> int:
        if requested is not None and requested > 0:
            return requested

        return DEFAULT_MAX_OUTPUT_TOKENS

    def temperature(self, requested: Optional[float]) -> Optional[float]:
        if requested is not None and requested >= 0.0:
            return requested

        return DEFAULT_TEMPERATURE

    def top_p(self, requested: Optional[float]) -> Optional[float]:
        if requested is not None and 0.0 <= requested <= 1.0:
            return requested

        return DEFAULT_TOP_P

    def start(self) -> None:
        if self._started:
            return

        for rank in range(self.tp_size):
            process = self.ctx.Process(
                target=_worker_entrypoint,
                args=(
                    rank,
                    self.tp_size,
                    self.model_path,
                    self.lora_path,
                    self.tokenizer_path,
                    self.chat_template_path,
                    self.output_parser_name,
                    self.model_dtype,
                    self.request_queues[rank],
                    self.response_queue,
                    self.ready_queue,
                    self.profile,
                ),
            )
            process.start()
            self.processes.append(process)

        for _ in range(self.tp_size):
            self.ready_queue.get()

        self._started = True

        print(f"---")
        print(f"all workers started.")
        print(f"---")

    def stop(self) -> None:
        if not self._started:
            return
        for queue in self.request_queues:
            queue.put(None)
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        self._started = False

    def dispatch(self, payloads: List[ChatCompletionRequest]) -> List[ChatCompletionResult]:
        if not self._started:
            raise RuntimeError("Workers have not been started")

        with self._dispatch_lock:
            if self.tp_size > 1:
                # All TP ranks must run the same generation step together.
                for queue in self.request_queues:
                    queue.put(payloads)
            else:
                # If no tensor parallelism, just send the payload to the next queue in a
                # round-robin fashion.
                queue_index = self._next_queue % self.tp_size
                self._next_queue = (self._next_queue + 1) % self.tp_size
                self.request_queues[queue_index].put(payloads)

            responses: List[ChatCompletionResult] = self.response_queue.get()

            for i, response in enumerate(responses):
                if response.request_id != payloads[i].request_id:
                    raise RuntimeError("Received mismatched response from worker queue")

            return responses
