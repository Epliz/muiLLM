import json

from muillm.server.chatcompletion import ChatCompletionAssistantMessage, ChatCompletionFunctionCall, ChatCompletionFunctionToolCall, ChatMessage
from muillm.server.idutils import generate_id
from muillm.server.outputparsers.outputparser import OutputParser

_LLAMA3_THINKING_START = "<thoughts>"
_LLAMA3_THINKING_END = "</thoughts>"
_LLAMA3_THINKING_TOOL_CALL_START = "<tool_call>"
_LLAMA3_THINKING_TOOL_CALL_END = "</tool_call>"
_LLAMA3_THINKING_END_OF_TURN = "<|eot_id|>"

class Llama3ThinkingOutputParser(OutputParser):
    def _extract_reasoning_content(self, content: str) -> tuple[str, str]:
        """
        Extract the reasoning content from the text, if present.
        """
        reasoning_content = None

        # Check for reasoning content
        start_index = content.find(_LLAMA3_THINKING_START)
        end_index = content.find(_LLAMA3_THINKING_END)

        if start_index > end_index:
            # There are probably several blocks of thoughts
            # the first one starting in the prompt, the second after the end marker
            # in that case remove the one that starts the second block of thoughts
            start_index = -1

        if start_index != -1 and end_index != -1:
            # There is both the beginning and end tags
            reasoning_content = content[start_index + len(_LLAMA3_THINKING_START):end_index].strip()
            content = content[:start_index] + content[end_index + len(_LLAMA3_THINKING_END):]
        elif end_index != -1:
            # There is an end tag but no start tag
            reasoning_content = content[:end_index].strip()
            content = content[end_index + len(_LLAMA3_THINKING_END):]
        else:
            # No tags at all
            reasoning_content = None

        return reasoning_content, content.strip()

    def _remove_end_of_turn_markers(self, content: str) -> str:
        eot_idx = content.find(_LLAMA3_THINKING_END_OF_TURN)

        if eot_idx != -1:
            # there are end of turn markers
            return content[:eot_idx]

        return content

    def _extract_tool_calls(self, content: str) -> tuple[str, list[ChatCompletionFunctionToolCall]]:
        """
        Extract the tool calls from the text, if present.
        """
        tool_calls = []

        while True:
            start_index = content.find(_LLAMA3_THINKING_TOOL_CALL_START)
            end_index = content.find(_LLAMA3_THINKING_TOOL_CALL_END)

            if start_index != -1 and end_index != -1:
                # There is both the beginning and end tags
                tool_call_content = content[start_index + len(_LLAMA3_THINKING_TOOL_CALL_START):end_index].strip()

                try:
                    tool_call_dict = json.loads(tool_call_content)

                    tool_call = ChatCompletionFunctionToolCall(
                        id=generate_id("tool_call_", length=4),
                        function=ChatCompletionFunctionCall(
                            name = tool_call_dict["name"],
                            arguments = json.dumps(tool_call_dict["parameters"]),
                        ),
                    )
                    tool_calls.append(tool_call)
                except Exception as e:
                    print(f"Error parsing tool call JSON: {e}. Content: {tool_call_content}")

                # remove the tool call from the content string
                # (we allow several tool calls, intertwined with the content)
                content = content[:start_index] + content[end_index + len(_LLAMA3_THINKING_TOOL_CALL_END):]
            else:
                break

        return content.strip(), tool_calls

    def parse(self, content: str) -> ChatMessage:
        # Remove end markers if present
        content = self._remove_end_of_turn_markers(content)

        # Extract the reasoning content from the text
        reasoning_content, content = self._extract_reasoning_content(content)

        content, tool_calls = self._extract_tool_calls(content)

        return ChatCompletionAssistantMessage(
            reasoning_content=reasoning_content,
            content=content,
            tool_calls=tool_calls,
        )