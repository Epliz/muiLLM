import argparse
from typing import Any, Dict, List
from unittest import result

import uvicorn

import time
from http import HTTPStatus

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from muillm.server.chatcompletion import ChatCompletionRequest
from muillm.server.idutils import generate_id
from muillm.server.outputparsers.outputparser import OutputParser
from muillm.server.runtime import ModelWorkerManager


def parse_args():
    parser = argparse.ArgumentParser(description="muiLLM server")
    group = parser.add_argument_group("model", "model options")
    group.add_argument(
        "--model-path",
        dest="model_path",
        type=str,
        default=None,
        help="Path to model to server or HuggingFace model ID",
    )
    group.add_argument(
        "--lora-path",
        dest="lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights to apply to the model",
    )
    group.add_argument(
        "--served-model-id",
        dest="served_model_id",
        type=str,
        default=None,
        help="Model ID exposed by the server",
    )
    group = parser.add_argument_group("parallelization", "parallelization options")
    group.add_argument(
        "--tensor-parallelism-size",
        dest="tp_size",
        type=int,
        default=0,
        help="Tensor parallel size",
    )
    group = parser.add_argument_group("server", "server options")
    group.add_argument("--host", dest="host", type=str, default="0.0.0.0", help="Server host")
    group.add_argument("--port", dest="port", type=int, default=8000, help="Server port")
    group = parser.add_argument_group("templates", "chat template options")
    group.add_argument(
        "--tokenizer-path",
        dest="tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer configuration folder",
    )
    group.add_argument(
        "--chat-template-path",
        dest="chat_template_path",
        type=str,
        default=None,
        help="Path to chat template file",
    )
    group.add_argument(
        "--output-parser",
        dest="output_parser",
        type=str,
        default=None,
        help=f"Name of the output parser to use ({','.join(OutputParser.list_output_parsers())})"
    )
    group = parser.add_argument_group("kvcache", "kv cache options")
    group.add_argument("--max-batch-size", dest="max_batch_size", type=int, help="Maximum batch size for KV cache")
    group.add_argument("--max-context-length", dest="max_context_length", type=int, help="Maximum context length for KV cache")
    group.add_argument("--max-output-length", dest="max_output_length", type=int, help="Maximum output length of one completion")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--profile", dest="profile", action="store_true", help="Enable profiling")
    group.add_argument("--no-profile", dest="profile", action="store_false", help="Disable profiling")
    parser.set_defaults(profile=None)

    args, _ = parser.parse_known_args()

    if not args.model_path and not args.served_model_id:
        parser.error("either --model-path or --served-model-id must be provided")

    return args


def build_app(manager: ModelWorkerManager) -> FastAPI:
    app = FastAPI(title="muiLLM Server", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"status": "ok", "workers": manager.tp_size}

    @app.get("/v1/models")
    def list_models() -> JSONResponse:
        model_info = {
            "id": manager.served_model_id,
            "object": "model",
            "owned_by": "muiLLM",
            "permission": [],
        }
        return JSONResponse(status_code=HTTPStatus.OK, content={"data": [model_info]})

    def check_completion_request(request: ChatCompletionRequest) -> None:
        if not request.messages:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="messages must not be empty")

        if request.stream:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="streaming is not yet supported")

        if request.model is not None and request.model != manager.served_model_id:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"model '{request.model}' is not served by this server (served model: '{manager.served_model_id}')",
            )

    def prepare_completion_request(request: ChatCompletionRequest) -> ChatCompletionRequest:
        request_id = generate_id("req_", length=8) if request.request_id is None else request.request_id

        return ChatCompletionRequest(
            request_id=request_id,
            model=request.model or manager.served_model_id,
            messages=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            max_tokens=manager.max_tokens(request.max_tokens),
            temperature=manager.temperature(request.temperature),
            top_p=manager.top_p(request.top_p),
        )

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> JSONResponse:
        check_completion_request(request)

        prepared_request = prepare_completion_request(request)

        try:
            results = manager.dispatch([prepared_request])
        except Exception as exc:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        created = int(time.time())

        if len(results) != 1:
            raise ValueError(f"Mismatched number of results returned from dispatch: expected 1, got {len(results)}")

        result = results[0]

        generation_response = result.response
        print(f"response text {generation_response}")

        response = {
            "id": prepared_request.request_id,
            "object": "chat.completion",
            "created": created,
            "model": prepared_request.model,
            "choices": [
                {
                    "index": 0,
                    "message": result.response.model_dump(),
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        return JSONResponse(status_code=HTTPStatus.OK, content=response)

    @app.post("/v1/chat/batchcompletions")
    def chat_batchcompletions(requests: List[ChatCompletionRequest]) -> JSONResponse:
        for request in requests:
            check_completion_request(request)

        prepared_requests = [prepare_completion_request(request) for request in requests]

        try:
            results = manager.dispatch(prepared_requests)
        except Exception as exc:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        created = int(time.time())

        responses = [{
            "id": prepared_request.request_id,
            "object": "chat.completion",
            "created": created,
            "model": prepared_request.model,
            "choices": [
                {
                    "index": 0,
                    "message": result.response.model_dump(),
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        } for prepared_request, result in zip(prepared_requests, results)]

        return JSONResponse(status_code=HTTPStatus.OK, content=responses)

    return app


def main():
    args = parse_args()

    # load the model and start the engine first
    manager = ModelWorkerManager(args)
    manager.start()

    # then make the endpoints available
    app = build_app(manager)


    @app.on_event("shutdown")
    def _shutdown() -> None:
        manager.stop()

    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
