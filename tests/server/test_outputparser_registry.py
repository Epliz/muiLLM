from muillm.server.outputparsers.llama3 import Llama3OutputParser
from muillm.server.outputparsers.llama3thinking import Llama3ThinkingOutputParser
from muillm.server.outputparsers.outputparser import OutputParser


def test_list_output_parsers_contains_registered_parsers() -> None:
    parser_names = OutputParser.list_output_parsers()

    assert "llama3" in parser_names
    assert "llama3thinking" in parser_names


def test_create_output_parser_by_name() -> None:
    parser = OutputParser.create_output_parser("Llama3ForCausalLM", "llama3")

    assert isinstance(parser, Llama3OutputParser)


def test_create_output_parser_by_model_name() -> None:
    parser = OutputParser.create_output_parser("Llama3ThinkingForCausalLM", None)

    assert isinstance(parser, Llama3ThinkingOutputParser)
