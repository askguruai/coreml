from enum import Enum
from typing import Dict, List, Literal

from pydantic import BaseModel, Field, validator


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


class ChatRoles(str, Enum):
    user = "user"
    assistant = "assistant"


class CompletionsInput(BaseModel):
    query: str = Field(
        description="A query to get an asnwer to.", example="Do you offer screen sharing chat?"
    )
    info: str | None = Field(
        default=None,
        description="Information to use for answering the query. If not provided, the query will be answered without any context.",
        example="Available options: Teamviewer and join.me",
    )
    chat: List[Dict[Literal["role", "content"], str]] | None = Field(
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

    @validator("chat", each_item=True)
    def validate_chat_author(cls, v):
        if v["role"] not in [role.value for role in ChatRoles]:
            raise ValueError(
                f"Role must be one of {', '.join([role.value for role in ChatRoles])}"
            )
        return v

    def __hash__(self):
        # to fasten the hash of chat history, only the first and last message are considered
        # chat_hashable = None if not self.chat else tuple((self.chat[idx]["role"], self.chat[idx]["content"]) for idx in [0, -1])
        chat_hashable = (
            None if not self.chat else tuple((msg["role"], msg["content"]) for msg in self.chat)
        )
        return hash((self.query, self.info, chat_hashable, self.mode))


class CompletionsResponse(BaseModel):
    data: str = Field(
        description="Response to the given info and query.",
        example="I used to play drums.",
    )
