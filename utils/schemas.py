from typing import List

from pydantic import BaseModel, Field


class HTTPExceptionResponse(BaseModel):
    detail: str = Field(example="Internal Server Error")


class EmbeddingsInput(BaseModel):
    input: List[str] = Field(description="Texts to embed.", example=["Hello world!"])


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


class CompletionsInput(BaseModel):
    query: str = Field(
        description="A query to get an asnwer to.", example="What did you do as a kid?"
    )
    info: str | None = Field(
        default=None,
        description="Information to use for answering the query. If not provided, the query will be answered without any context.",
        example="When I was a kid I used to play drums",
    )


class CompletionsResponse(BaseModel):
    data: str = Field(
        description="Response to the given info and query.",
        example="I used to play drums.",
    )
