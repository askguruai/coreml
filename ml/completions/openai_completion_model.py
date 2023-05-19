from pprint import pformat

import openai
from fastapi.responses import StreamingResponse
from loguru import logger

from ml.completions import CompletionModel
from utils import CONFIG
from utils.schemas import CompletionsInput, CompletionsResponse


class OpenAICompletionModel(CompletionModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def get_completion(self, completions_input: CompletionsInput) -> str:
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(completions_input),
            },
        ]

        if completions_input.info:
            messages.append(
                {
                    "role": "user",
                    "content": f"You are given the parts of the documents from a knowledge base and a question, compile a final answer.\nIn your answer, use only provided parts of the document.\nIf you don't know the answer, just say that you were unable to find an answer in the knowledge base.\nDon't try to make up an answer.\nParts of the documents:\n\"\"\"\n{completions_input.info}\n\"\"\"",
                }
            )

        if completions_input.chat:
            messages += completions_input.chat

        if completions_input.query:
            messages.append(
                {
                    "role": "user",
                    "content": completions_input.query,
                }
            )

        logger.info(
            "completions request:"
            + '\n'
            + '\n=============================\n'.join(
                [f"{message['role']}: {message['content']}" for message in messages]
            )
        )
        answer = await openai.ChatCompletion.acreate(
            model=self.model_name,
            messages=messages,
            temperature=0.4,
            max_tokens=300,
            stream=completions_input.stream,
        )
        if completions_input.stream:
            answer = (
                CompletionsResponse(
                    data=message["choices"][0]["delta"]["content"]
                    if "content" in message["choices"][0]["delta"]
                    else ""
                ).json()
                async for message in answer
            )
            return StreamingResponse(answer, media_type='text/event-stream', headers={'X-Accel-Buffering': 'no'})
        answer = self.postprocess_output(answer["choices"][0]["message"]["content"].lstrip())
        logger.info("completions result:" + '\n' + answer)
        return CompletionsResponse(data=answer)

    @staticmethod
    def get_system_prompt(completions_input: CompletionsInput) -> str:
        prompt = ""
        match completions_input.mode:
            case "general":
                prompt += "You are helpful assistant which follows given instructions and is seeking to answer a user's question."
            case "support":
                prompt += "You are an AI which acts as a customer support agent who is seeking to provide a complete, simple, helpful AND TRUTHFUL answer to a customer.\nIn your answer do not refer to customer support because you ARE customer support."

        return prompt
