from pprint import pformat

import openai
from fastapi.responses import StreamingResponse
from loguru import logger

from ml.completions import CompletionModel
from utils import CONFIG
from utils.schemas import AnswerInContextResponse, ApiVersion, CompletionsInput, CompletionsResponse


class OpenAICompletionModel(CompletionModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

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
                + (completions_input.chat or "")
                + "\n\n\"\"\"\nQuestion:\n\"\"\"\n"
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
        answer = await openai.ChatCompletion.acreate(
            model=CONFIG["v1.completions"]["model"]
            if api_version == ApiVersion.v1
            else CONFIG["v2.completions"]["model"],
            messages=messages,
            temperature=0.4,
            max_tokens=5,
        )
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
            model=CONFIG["v1.completions"]["model"]
            if api_version == ApiVersion.v1
            else CONFIG["v2.completions"]["model"],
            messages=messages,
            temperature=0.4,
            max_tokens=500,
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
                prompt += "You are helpful AskGuru assistant which follows given instructions and is seeking to answer a user's question in a language it was asked."
            case "support":
                prompt += "You are an AI which acts as a customer support agent who is seeking to provide a complete, simple, helpful AND TRUTHFUL answer to a customer.\nIn your answer do not refer to customer support because you ARE customer support."

        return prompt
