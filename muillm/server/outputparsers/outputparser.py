from abc import ABC, abstractmethod

from muillm.server.chatcompletion import ChatMessage

class OutputParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> ChatMessage:
        """
        Parse the output text from the model and return the final response.
        """
        ...