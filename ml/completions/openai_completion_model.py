import logging

import openai

from data import (
    COMPLETIONS_PROMPT_OPENAI,
    COMPLETIONS_PROMPT_OPENAI_NO_INFO,
    COMPLETIONS_PROMPT_OPENAI_SYSTEM,
)
from ml.completions import CompletionModel
from utils import CONFIG
from utils.schemas import CompletionsInput


class OpenAICompletionModel(CompletionModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_completion(self, completions_input: CompletionsInput) -> str:
        prompt = (
            COMPLETIONS_PROMPT_OPENAI(completions_input.info, completions_input.query)
            if completions_input.info
            else COMPLETIONS_PROMPT_OPENAI_NO_INFO(completions_input.query)
        )
        logging.info("completions request:" + '\n' + prompt)
        answer = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": COMPLETIONS_PROMPT_OPENAI_SYSTEM[completions_input.mode],
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=300,
            presence_penalty=0.6,
        )["choices"][0]["message"]["content"].lstrip()
        logging.info("completions result:" + '\n' + answer)
        return answer
