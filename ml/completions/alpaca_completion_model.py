import torch
from loguru import logger
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from ml.completions.completion_model import CompletionModel
from utils.schemas import CompletionsInput


class AlpacaCompletionModel(CompletionModel):
    def __init__(self, model_name: str, device: str):
        logger.info(f"Loading {model_name} model to {device}")
        self.device = torch.device(device)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        # self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(
        #     self.device
        # )
        self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.model.eval()
        # self.model = torch.compile(self.model)

        self.generation_config = GenerationConfig(
            # temperature=0.7,
            # top_p=0.75,
            # top_k=40,
            max_new_tokens=300,
            temperature=0.2,
            top_p=0.75,
            top_k=40,
            num_beams=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            encoder_repetition_penalty=1.0,
            typical_p=1.0,
            length_penalty=1.2,
            do_sample=True,
        )

    def get_completion(self, completions_input: CompletionsInput) -> str:
        prompt = AlpacaCompletionModel.generate_prompt(completions_input.query, completions_input.info)

        logger.info("completions request:" + "\n" + prompt)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
            )

        answer = self.tokenizer.decode(outputs[0][len(input_ids[0]) :], skip_special_tokens=True).lstrip()

        logger.info("completions result:" + "\n" + answer)
        return answer

    @staticmethod
    def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
        if input_ctxt:
            # return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            return f"""Below is a question from a customer to an agent, paired with a potentially useful context. Answer customer's question using the context. Return the answer and provide additional information if necessary.

### Instruction:
{instruction}

### Input:
{input_ctxt}

### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
