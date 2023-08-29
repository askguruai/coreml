from enum import Enum
from typing import Dict, List, Literal

from pydantic import BaseModel, Field, validator


class ApiVersion(str, Enum):
    v1 = "v1"
    v2 = "v2"


class AnswerInContextResponse(BaseModel):
    answer: bool = Field(
        description="Whether the answer is in context.",
        example=True,
    )


class HTTPExceptionResponse(BaseModel):
    detail: str = Field(example="Internal Server Error")


class EmbeddingsInput(BaseModel):
    input: List[str] = Field(description="Texts to embed.", example=["Hello world!"])

    def __hash__(self):
        return hash(tuple(self.input))


class EmbeddingsInputInstruction(BaseModel):
    input: List[str] = Field(description="Texts to embed.", example=["Hello world!"])
    instruction: str | None = Field(
        description="Instruction to for embeddings model.",
        example="Represent the knowledge base search query for retrieving relevant information.",
    )


class EmbeddingsResponse(BaseModel):
    data: List[List[float]] = Field(
        description="List of embeddings.",
        example=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10]],
    )


class CompletionsMode(str, Enum):
    general = "general"
    support = "support"


class SummarizationInput(BaseModel):
    info: str = Field(
        description="Information to be summarized. Size does not matter.",
        example="Voters in Switzerland have backed a new climate bill designed to cut fossil fuel use and reach net-zero carbon emissions by 2050. The government says the country needs to protect its energy security and the environment, as glaciers melt rapidly in the Swiss Alps. The law will require a move away from dependence on imported oil and gas towards the use of renewable sources. In Sunday's referendum 59.1% of voters backed the green energy proposals. Opponents had argued the measures would push up energy prices. Nearly all of Switzerland's major parties supported the bill, except the right-wing Swiss People's Party (SVP), which triggered the referendum after pushing back against the government's proposals. Switzerland imports about three-quarters of its energy, with all the oil and natural gas consumed coming from abroad. The climate bill pledges financial support of 2bn Swiss francs ($2.2bn; £1.7bn) over a decade to promote the replacement of gas or oil heating systems with climate-friendly alternatives, and SFr1.2bn to push businesses towards green innovation.",
    )
    max_tokens: int = Field(
        default=250,
        description="Maximum number of tokens to generate.",
        example=250,
    )
    stream: bool = Field(
        default=False,
        description="If true, the response will be streamed as it is generated. This is useful for long-running requests.",
        example=False,
    )


class Role(str, Enum):
    user = 'user'
    assistant = 'assistant'


class Message(BaseModel):
    role: Role = Field(description="Role of the user", example=Role.user)
    content: str = Field(description="Content of the message", example="How to fix income mail view?")


class CompletionsInput(BaseModel):
    query: str = Field(description="A query to get an asnwer to.", example="Do you offer screen sharing chat?")
    info: str | None = Field(
        default=None,
        description="Information to use for answering the query. If not provided, the query will be answered without any context.",
        example="Available options: Teamviewer and join.me",
    )
    chat: List[Message] | None = Field(
        default=None,
        description="Chat history from communication between user and agent.",
        example=[
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "do you offer screen sharing chat"},
            {"role": "assistant", "content": "Hello, I will check, thanks for waiting."},
            {"role": "user", "content": "Sure."},
        ],
    )
    mode: CompletionsMode | None = Field(
        default=CompletionsMode.general,
        description=f"Mode of the completion. If not provided, the default mode will be used. Possible values: {', '.join([mode.value for mode in CompletionsMode])}",
        example=CompletionsMode.support,
    )
    stream: bool = Field(
        default=False,
        description="If true, the response will be streamed as it is generated. This is useful for long-running requests.",
    )
    include_image_urls: bool | None = Field(
        default=False,
        description="If include image urls in the output answer"
    )

    def __hash__(self):
        # to fasten the hash of chat history, only the first and last message are considered
        # chat_hashable = None if not self.chat else tuple((self.chat[idx]["role"], self.chat[idx]["content"]) for idx in [0, -1])
        chat_hashable = None if not self.chat else tuple((msg.role, msg.content) for msg in self.chat)
        return hash((self.query, self.info, chat_hashable, self.mode))


class CompletionsResponse(BaseModel):
    data: str = Field(
        description="Response to the given info and query.",
        example="I used to play drums.",
    )
