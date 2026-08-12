import json
from typing import Any, Dict, List, Optional
from muillm.server.chatcompletion import ChatCompletionAssistantMessage, ChatCompletionFunctionTool, ChatCompletionFunctionToolCall, ChatCompletionToolCall, ChatCompletionToolMessage, ChatMessage

def convert_tool_call_to_hf_message(tool_call: ChatCompletionToolCall) -> Dict[str, Any]:
    if not isinstance(tool_call, ChatCompletionFunctionToolCall):
        raise ValueError(f"Unsupported tool call type: {type(tool_call)}")

    function_tool_call = tool_call.function

    # the chat completion tool call is a single string containing the JSON arguments,
    # we need to parse it into a dictionary for the HF message format
    tool_call_arguments = json.loads(function_tool_call.arguments)

    tool_call_dict = {
        "type": "function",
        "function": {
            "name": function_tool_call.name,
            "arguments": tool_call_arguments
        }
    }

    if tool_call.id is not None:
        tool_call_dict["id"] = tool_call.id

    return tool_call_dict

def convert_to_hf_message(message: ChatMessage) -> Dict[str, Any]:
    message_dict = {"role": message.role, "content": message.content}

    # name for multi-participant conversations
    if message.name is not None:
        message_dict["name"] = message.name

    if message.reasoning_content is not None:
        message_dict["reasoning_content"] = message.reasoning_content

    if isinstance(message, ChatCompletionAssistantMessage) and message.tool_calls is not None:
        message_dict["tool_calls"] = [convert_tool_call_to_hf_message(tc) for tc in message.tool_calls]

    if isinstance(message, ChatCompletionToolMessage) and message.tool_call_id is not None:
        message_dict["tool_call_id"] = message.tool_call_id

    return message_dict

def convert_to_hf_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    return [convert_to_hf_message(message) for message in messages]

def convert_to_hf_tools(tools: Optional[List[ChatCompletionFunctionTool]]) -> Optional[List[dict[str, Any]]]:
    if tools is None:
        return None

    hf_tools = []
    for tool in tools:
        if isinstance(tool, ChatCompletionFunctionTool):
            hf_tool = {
                "type": "function",
                "function": {
                    "name": tool.function.name,
                    "description": getattr(tool.function, "description", None),
                    "parameters": getattr(tool.function, "parameters", None),
                },
            }
            hf_tools.append(hf_tool)
        else:
            raise ValueError(f"Unsupported tool type: {type(tool)}")
    return hf_tools