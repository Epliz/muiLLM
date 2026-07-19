from abc import ABC
from typing import Annotated, ClassVar, List, Literal, Optional, Union
from typing_extensions import Literal

from pydantic import BaseModel, Field


class ChatCompletionToolCall(BaseModel, ABC):
    type: ClassVar[str]

    id: Optional[str] = None

class ChatCompletionFunctionCall(BaseModel):
    name: str
    # JSON object containing the parameters
    arguments: str

class ChatCompletionFunctionToolCall(ChatCompletionToolCall):
    type: Literal["function"] = "function"
    function: ChatCompletionFunctionCall

# Union of all tool call types
ToolCallUnion = Union[ChatCompletionFunctionToolCall]

class ChatMessage(BaseModel, ABC):
    role: ClassVar[str]
    # name of the participant to distinguish
    # in multi-participant conversations
    name: Optional[str] = None

    reasoning_content: Optional[str] = None
    content: str

class ChatCompletionSystemMessage(ChatMessage):
    role: Literal["system"] = "system"

class ChatCompletionUserMessage(ChatMessage):
    role: Literal["user"] = "user"

class ChatCompletionAssistantMessage(ChatMessage):
    role: Literal["assistant"] = "assistant"

    tool_calls: Optional[List[Annotated[ToolCallUnion, Field(discriminator="type")]]] = None

class ChatCompletionToolMessage(ChatMessage):
    role: Literal["tool"] = "tool"

    tool_call_id: str

ChatMessageTypes = Union[
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage
]

class ChatCompletionFunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    # JSON schema for the function parameters,
    # following OpenAI's function calling spec
    parameters: Optional[dict] = None

class ChatCompletionTool(BaseModel, ABC):
    type: ClassVar[str]

class ChatCompletionFunctionTool(ChatCompletionTool):
    type: Literal["function"] = "function"
    function: ChatCompletionFunctionDefinition

# Union of all tool types
ToolUnion = Union[ChatCompletionFunctionTool]

class ChatCompletionRequest(BaseModel):
    request_id: Optional[str] = None

    model: Optional[str] = None

    messages: List[Annotated[ChatMessageTypes, Field(discriminator="role")]] = Field(default_factory=list)

    # tool related parameters
    tools: Optional[List[Annotated[ToolUnion, Field(discriminator="type")]]] = None
    tool_choice: Optional[Literal["none", "auto", "manual"]] = "auto"

    # generation parameters
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False

class ChatCompletionResult(BaseModel):
    request_id: Optional[str] = None

    response: Annotated[ChatMessageTypes, Field(discriminator="role")]