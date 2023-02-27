import logging

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class CompletionModel:
    def __init__(self, model_name: str, device: str):
        logging.info(f"Loading {model_name} model to {device}")
        self.device = torch.device(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )

    def get_completion(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        outputs = self.model.generate(
            input_ids,
            temperature=1.0,
            min_length=10,
            max_new_tokens=300,
            repetition_penalty=2.5,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
