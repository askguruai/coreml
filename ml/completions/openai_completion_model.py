from pprint import pformat

import openai
from loguru import logger

from ml.completions import CompletionModel
from utils import CONFIG
from utils.schemas import CompletionsInput


class OpenAICompletionModel(CompletionModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_completion(self, completions_input: CompletionsInput) -> str:
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(completions_input),
            },
        ]
        if completions_input.chat:
            messages += completions_input.chat

        if completions_input.query:
            messages.append(
                {
                    "role": "user",
                    "content": completions_input.query,
                }
            )

        logger.info("completions request:" + '\n' + pformat(messages))
        answer = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.9,
            max_tokens=300,
            presence_penalty=0.6,
        )["choices"][0]["message"]["content"].lstrip()
        logger.info("completions result:" + '\n' + answer)
        return answer

    @staticmethod
    def get_system_prompt(completions_input: CompletionsInput) -> str:
        prompt = ""
        match completions_input.mode:
            case "general":
                prompt += "You are helpful assistant which follows given instructions and is seeking to answer a user's question."
            case "support":
                prompt += "You are a customer support agent who is seeking to provide a complete, simple and helpful answer to a customer in a friendly manner. If text does not provide relevant information, say that you are not able to help because knowledge base does not contain necessary information."

        if completions_input.info:
            prompt += f"\n\nYou found following relevant information (which is not visible to a user):\n\"\"\"\n{completions_input.info}"

        return prompt
