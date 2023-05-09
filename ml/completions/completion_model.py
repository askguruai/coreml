from abc import abstractmethod

from utils.schemas import CompletionsInput


class CompletionModel:
    @abstractmethod
    def get_completion(self, completions_input: CompletionsInput) -> str:
        pass

    @staticmethod
    def get_completion_subprocess(cls, completions_input: CompletionsInput, state: dict):
        state[cls.__class__.__name__] = cls.get_completion(completions_input=completions_input)

    @staticmethod
    def postprocess_output(s: str) -> str:
        return s[: max(s.rfind("."), s.rfind("?"), s.rfind("!"), s.rfind("\n")) + 1]
