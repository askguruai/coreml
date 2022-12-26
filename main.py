from typing import List

import openai
import uvicorn
from fastapi import FastAPI

from utils import CONFIG
from utils.api import EmbeddingsInput
from utils.logging import run_uvicorn_loguru

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/embeddings/")
async def get_embeddings(embeddings_input: EmbeddingsInput):
    return {
        "data": openai.Embedding.create(
            input=embeddings_input.input, model=CONFIG["model"]["embeddings_model"]
        )["data"]
    }


if __name__ == "__main__":

    run_uvicorn_loguru(
        uvicorn.Config(
            "main:app",
            host=CONFIG["app"]["host"],
            port=int(CONFIG["app"]["port"]),
            log_level=CONFIG["app"]["log_level"],
        )
    )
