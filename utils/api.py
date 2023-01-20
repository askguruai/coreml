from typing import List, Optional

from pydantic import BaseModel, Field


class HTTPExceptionResponse(BaseModel):
    detail: str = Field(example="OpenAI API returned an API Error: 400 Bad Request")


class EmbeddingObject(BaseModel):
    object: str = Field(example="embedding")
    index: int = Field(example=0)
    embedding: List[float] = Field(example=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10])


class EmbeddingsInput(BaseModel):
    input: str | List[str] = Field(description="Text(s) to embed.", example="Hello world!")


class EmbeddingsResponse(BaseModel):
    data: List[EmbeddingObject] = Field(
        description="List of EmbeddingObject.",
        example=[
            EmbeddingObject(
                object="embedding",
                index=0,
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10],
            )
        ],
    )


class CompletionsInput(BaseModel):
    info: str | None = Field(
        default=None,
        description="Information to use for answering the query. If not provided, the query will be answered without any context.",
        example="When I was a kid I used to play drums",
    )
    query: str = Field(
        description="A query to get an asnwer to.", example="What did you do as a kid?"
    )


class CompletionsResponse(BaseModel):
    data: str = Field(
        description="Response to the given info and query.", example="I used to play drums."
    )
