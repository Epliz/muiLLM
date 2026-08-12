from muillm.server.chatcompletion import ChatCompletionAssistantMessage, ChatMessage
from muillm.server.outputparsers.outputparser import OutputParser

_LLAMA3_END_OF_TURN = "<|eot_id|>"

class Llama3OutputParser(OutputParser):
    parser_name = "llama3"

    def __init__(self):
        super().__init__()

    @classmethod
    def matches(cls, model_class_name: str) -> bool:
        lowered_name = model_class_name.lower()
        return "llama" in lowered_name and "thinking" not in lowered_name

    def _remove_end_of_turn_markers(self, content: str) -> str:
        eot_idx = content.find(_LLAMA3_END_OF_TURN)

        if eot_idx != -1:
            # there are end of turn markers
            return content[:eot_idx]

        return content

    def parse(self, content: str) -> ChatMessage:
        # Remove end markers if present
        content = self._remove_end_of_turn_markers(content)

        # The Llama 3.1 model doesn't support thinking natively in its chat template.
        # The tool calls are also realistically not parsable as they are just JSON objects.
        # So we don't support either.
        return ChatCompletionAssistantMessage(
            reasoning_content="",
            content=content,
            tool_calls=[],
        )