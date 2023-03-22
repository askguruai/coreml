import logging

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from data import COMPLETIONS_PROMPT_CUSTOM, COMPLETIONS_PROMPT_OPENAI_NO_INFO
from ml.completions import CompletionModel
from utils.schemas import CompletionsInput


class T5CompletionModel(CompletionModel):
    def __init__(self, model_name: str, device: str):
        logging.info(f"Loading {model_name} model to {device}")
        self.device = torch.device(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        # self.model.share_memory()

    def get_completion(self, completions_input: CompletionsInput) -> str:
        prompt = (
            COMPLETIONS_PROMPT_CUSTOM(
                completions_input.info.replace("\n\n", "\n"), completions_input.query
            )
            if completions_input.info
            else COMPLETIONS_PROMPT_OPENAI_NO_INFO(completions_input.query)
        )
        logging.info("completions request:" + "\n" + prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            temperature=1.0,
            min_length=10,
            max_new_tokens=300,
            repetition_penalty=2.5,
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info("completions result:" + "\n" + answer)
        return answer
