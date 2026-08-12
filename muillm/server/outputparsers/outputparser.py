from abc import ABC, ABCMeta, abstractmethod
from typing import Optional

from muillm.server.chatcompletion import ChatMessage


class OutputParserMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)

        if name == "OutputParser":
            cls._registry = {}
        elif not getattr(cls, "__abstractmethods__", None):
            parser_name = getattr(cls, "parser_name", "")
            if parser_name:
                cls._registry[parser_name] = cls

        return cls


class OutputParser(ABC, metaclass=OutputParserMeta):
    parser_name: str = ""
    _registry: dict[str, type["OutputParser"]] = {}

    def __init__(self, parser_name: Optional[str] = None):
        self.parser_name = parser_name or self.__class__.parser_name

    @classmethod
    def list_output_parsers(cls) -> list[str]:
        """
        List all available output parsers.
        """
        return sorted(cls._registry)

    @classmethod
    def create_output_parser(
        cls,
        model_class_name: str,
        parser_name: Optional[str],
    ) -> "OutputParser":
        """
        Create an output parser based on the provided name.
        """
        if parser_name is not None:
            parser_cls = cls._registry.get(parser_name)
            if parser_cls is None:
                raise ValueError(f"Unknown output parser: {parser_name}")

            return parser_cls()

        for parser_cls in cls._registry.values():
            if parser_cls.matches(model_class_name):
                return parser_cls()

        raise ValueError(f"No output parser found for model class: {model_class_name}")

    @classmethod
    @abstractmethod
    def matches(cls, model_class_name: str) -> bool:
        pass

    @abstractmethod
    def parse(self, text: str) -> ChatMessage:
        """
        Parse the output text from the model and return the final response.
        """
        ...