from typing import List

from pydantic import BaseModel


class EmbeddingsInput(BaseModel):
    input: str | List[str]
