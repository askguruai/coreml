from typing import List, Optional

from pydantic import BaseModel


class EmbeddingsInput(BaseModel):
    input: str | List[str]


class CompletionsInput(BaseModel):
    query: str
    info: Optional[str] = None
