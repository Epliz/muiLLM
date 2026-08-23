import json

from muillm.server.chatcompletion import ChatCompletionAssistantMessage, ChatCompletionFunctionCall, ChatCompletionFunctionToolCall, ChatMessage
from muillm.server.idutils import generate_id
from muillm.server.outputparsers.outputparser import OutputParser

_LLAMA3_THINKING_START = "<thoughts>"
_LLAMA3_THINKING_END = "</thoughts>"
_LLAMA3_THINKING_TOOL_CALL_START = "<tool_call>"
_LLAMA3_THINKING_TOOL_CALL_END = "</tool_call>"
_LLAMA3_THINKING_STRUCTURED_OUTPUT_START = "<structured_output>"
_LLAMA3_THINKING_STRUCTURED_OUTPUT_END = "</structured_output>"
_LLAMA3_THINKING_END_OF_TURN = "<|eot_id|>"

class Llama3ThinkingOutputParser(OutputParser):
    parser_name = "llama3thinking"

    def __init__(self):
        super().__init__()

    @classmethod
    def matches(cls, model_class_name: str) -> bool:
        lowered_name = model_class_name.lower()
        return "llama" in lowered_name and "thinking" in lowered_name

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

    def _extract_structured_output(self, content: str) -> tuple[str, dict]:
        """
        Extract the structured output from the text, if present.
        """
        structured_output = None

        start_index = content.find(_LLAMA3_THINKING_STRUCTURED_OUTPUT_START)
        end_index = content.find(_LLAMA3_THINKING_STRUCTURED_OUTPUT_END)

        if start_index != -1 and end_index != -1:
            # There is both the beginning and end tags
            structured_output_content = content[start_index + len(_LLAMA3_THINKING_STRUCTURED_OUTPUT_START):end_index].strip()

            try:
                structured_output = json.loads(structured_output_content)
            except Exception as e:
                print(f"Error parsing structured output JSON: {e}. Content: {structured_output_content}")

            # remove the structured output from the content string
            content = content[:start_index] + content[end_index + len(_LLAMA3_THINKING_STRUCTURED_OUTPUT_END):]

        return content.strip(), structured_output

    def parse(self, content: str) -> ChatMessage:
        # Remove end markers if present
        content = self._remove_end_of_turn_markers(content)
        # Keep a copy of the original content (before parsing tool calls and structured output)
        unparsed_content = content

        # Extract the reasoning content from the text
        reasoning_content, content = self._extract_reasoning_content(content)

        content, tool_calls = self._extract_tool_calls(content)

        content, structured_output = self._extract_structured_output(content)

        if structured_output is not None:
            # If the structured output is present, it replaces the content of the message
            content = json.dumps(structured_output)

        return ChatCompletionAssistantMessage(
            unparsed_content=unparsed_content,
            reasoning_content=reasoning_content,
            content=content,
            tool_calls=tool_calls,
        )