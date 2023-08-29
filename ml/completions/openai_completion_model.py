import asyncio
from pprint import pformat

import openai
import tiktoken
from fastapi.responses import StreamingResponse
from loguru import logger

from ml.completions import CompletionModel
from utils import CONFIG
from utils.misc import retry_with_time_limit
from utils.schemas import AnswerInContextResponse, ApiVersion, CompletionsInput, CompletionsResponse, SummarizationInput


class OpenAICompletionModel(CompletionModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.enc = tiktoken.get_encoding("cl100k_base")

    async def if_answer_in_context(
        self, completions_input: CompletionsInput, api_version: ApiVersion
    ) -> AnswerInContextResponse:
        messages = [
            {
                "role": "system",
                # "content": "You are a classification AI. You tell 1 if the answer is in the context and 0 if it is not.",
                "content": "You are a classification AI. You respond with a confidence score from 1 to 10. You are trying to predict if the answer is in the context.",
            },
            {
                "role": "user",
                # "content": f"You are given some texts and a question. You need to tell if the answer to the question is in the texts.\nThe text might be relevant to the question, but the answer might not be in the text.\nYou should respond only with one symbol: 1 if answer to a given query contained in the provided text and 0 otherwise.\n\nText:\n\"\"\"\n"
                "content": f"You are given some texts and a question. You need to tell if the answer to the question is in the texts.\nThe text might be relevant to the question, but the answer might not be in the text.\nYou should respond with confidence score from 1 to 10 if answer to a given query contained in the text.\n\nText:\n\"\"\"\n"
                + (completions_input.info or "")
                + (
                    "\n".join([f"{msg['role']}: {msg['content']}" for msg in completions_input.chat])
                    if completions_input.chat
                    else ""
                )
                + "\n\n\"\"\"\n\n\nQuestion:\n\"\"\"\n"
                + (completions_input.query or "")
                + "\n\"\"\"",
            },
        ]
        logger.info(
            "completions request:"
            + '\n'
            + '\n=============================\n'.join(
                [f"{message['role']}: {message['content']}" for message in messages]
            )
        )
        args = {
            "model": CONFIG["v1.completions"]["model"]
            if api_version == ApiVersion.v1
            else CONFIG["v2.completions"]["model"],
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 5,
        }
        answer = await retry_with_time_limit(openai.ChatCompletion.acreate, time_limit=5, max_retries=3, **args)
        answer = int(answer["choices"][0]["message"]["content"].lstrip())
        logger.info("completions result:" + '\n' + str(answer))
        answer = answer >= 5
        # answer = bool(int(answer["choices"][0]["message"]["content"].lstrip()))
        return AnswerInContextResponse(answer=answer)

    async def get_completion(self, completions_input: CompletionsInput, api_version: ApiVersion) -> CompletionsResponse:
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
                    "content": (
                        "You are a customer support agent. Reply in the first person on behalf of the company.\n"
                        if completions_input.mode == "support"
                        else ""
                    )
                    + f"You are given the parts of the documents from a knowledge base and a question, compile a final answer.\nIn your answer, use only provided parts of the document.\nIf you don't know the answer, just say that you were unable to find an answer in the knowledge base.\nDon't try to make up an answer.\n\n"
                    + (
                        f"Each document contains `doc_idx`. When you say something, refer to the document you are using in a format `{{doc_idx: <idx of the document>}}`, e.g. {{doc_idx: 3}}.\n"
                        if api_version != ApiVersion.v1
                        else f""
                    )
                    + f"Parts of the documents:\n\"\"\"\n{completions_input.info}\n\"\"\"",
                    # "content": f"You are given the parts of the documents from a knowledge base and a question, compile a final answer.\nIn your answer, use only provided parts of the document.\nIf you don't know the answer, just say that you were unable to find an answer in the knowledge base.\nDon't try to make up an answer.\n\n" + (f"Each document contains `doc_id` and `doc_collection`. When you say something, refer to the document you are using in a format `{{'doc_id': '<id of the document>', 'doc_collection': '<collection of the document>'}}`.\n" if api_version == ApiVersion.v2 else f"")  + f"Parts of the documents:\n\"\"\"\n{completions_input.info}\n\"\"\"",
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "Provide a simple and detailed answer to a question in a language it was asked",
                }
            )

        if completions_input.chat:
            messages += [msg.dict() for msg in completions_input.chat]

        if completions_input.query and not completions_input.chat:
            messages.append(
                {
                    "role": "user",
                    "content": completions_input.query + "\nIf there is a link to an image give me that in a markdown format.",
                }
            )

        logger.info(
            "completions request:"
            + '\n'
            + '\n=============================\n'.join(
                [f"{message['role']}: {message['content']}" for message in messages]
            )
        )
        args = {
            "model": CONFIG["v1.completions"]["model"]
            if api_version == ApiVersion.v1
            else CONFIG["v2.completions"]["model"],
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 600,
            "stream": completions_input.stream,
        }
        if not completions_input.stream:
            answer = await openai.ChatCompletion.acreate(**args)
            answer = self.postprocess_output(answer["choices"][0]["message"]["content"].lstrip())
            logger.info("completions result:" + '\n' + answer)
            return CompletionsResponse(data=answer)
        else:
            answer = await retry_with_time_limit(openai.ChatCompletion.acreate, time_limit=4, max_retries=3, **args)
            answer = (
                CompletionsResponse(
                    data=message["choices"][0]["delta"]["content"]
                    if "content" in message["choices"][0]["delta"]
                    else ""
                ).json()
                async for message in answer
            )
            return StreamingResponse(answer, media_type='text/event-stream', headers={'X-Accel-Buffering': 'no'})

    async def get_summarization(
        self, summarization_input: SummarizationInput, api_version: ApiVersion
    ) -> CompletionsResponse:
        prompt = (
            lambda text: f"Can you provide a comprehensive summary of the given article from a knowledge base? The summary should cover all the key points and main ideas presented in the original text, while also condensing the information into a concise and easy-to-understand format. Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information.\n\nText:\n\"\"\"\n{text}\n\"\"\""
        )
        max_request_size = 15_000
        args = {
            "model": "gpt-3.5-turbo-16k",
            "temperature": 0.6,
            "max_tokens": summarization_input.max_tokens,
        }
        text, length = summarization_input.info, len(self.enc.encode(summarization_input.info))
        while length > max_request_size:
            logger.warning(f"Splitting summarization request into {length // max_request_size + 1} parts")
            n_parts = length // max_request_size + 1
            tasks = []
            for i in range(n_parts):
                start = i * length // n_parts
                end = (i + 1) * length // n_parts
                tasks.append(
                    openai.ChatCompletion.acreate(
                        messages=[{"role": "user", "content": prompt(text[start:end])}],
                        **args,
                    )
                )
            results = await asyncio.gather(*tasks)
            text = "".join(
                [self.postprocess_output(result["choices"][0]["message"]["content"].lstrip()) for result in results]
            )
            length = len(self.enc.encode(text))

        args["messages"] = [{"role": "user", "content": prompt(text)}]
        args["stream"] = summarization_input.stream
        if not summarization_input.stream:
            answer = await openai.ChatCompletion.acreate(**args)
            answer = self.postprocess_output(answer["choices"][0]["message"]["content"].lstrip())
            return CompletionsResponse(data=answer)
        else:
            answer = await retry_with_time_limit(openai.ChatCompletion.acreate, time_limit=4, max_retries=3, **args)
            answer = (
                CompletionsResponse(
                    data=message["choices"][0]["delta"]["content"]
                    if "content" in message["choices"][0]["delta"]
                    else ""
                ).json()
                async for message in answer
            )
            return StreamingResponse(answer, media_type='text/event-stream', headers={'X-Accel-Buffering': 'no'})

    @staticmethod
    def get_system_prompt(completions_input: CompletionsInput) -> str:
        prompt = ""
        match completions_input.mode:
            case "general":
                prompt += "You are helpful AskGuru assistant which follows given instructions and is seeking to answer a user's question in a language it was asked."
            case "support":
                prompt += "You are a customer support agent who is seeking to provide a complete, simple, helpful and truthful answer to a customer. Reply with only answer to a query and nothing else."

        return prompt
