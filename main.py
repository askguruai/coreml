import logging
from pprint import pformat
from typing import List

import openai
import uvicorn
from fastapi import FastAPI

from utils import CONFIG
from utils.api import CompletionsInput, EmbeddingsInput
from utils.logging import run_uvicorn_loguru

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/embeddings/")
async def get_embeddings(embeddings_input: EmbeddingsInput):
    logging.info(
        f"Number of texts to embed: {1 if type(embeddings_input.input) == str else len(embeddings_input.input)}"
    )
    return {
        "data": openai.Embedding.create(
            input=embeddings_input.input, model=CONFIG["embeddings"]["model"]
        )["data"]
    }


@app.post("/completions/")
async def get_completions(completions_input: CompletionsInput):
    prompt = (
        (
            CONFIG["completions"]["intro_info"]
            + "\n\n"
            + completions_input.info
            + "\n\n"
            + "Question: "
            + completions_input.query
            + '\n'
            + "Answer:"
        )
        if completions_input.info
        else (
            CONFIG["completions"]["intro_general"]
            + "\n\n"
            + "Question: "
            + completions_input.query
            + '\n'
            + "Answer:"
        )
    )
    logging.info("completions request:" + '\n' + prompt)
    answer = openai.Completion.create(
        model=CONFIG["completions"]["model"],
        prompt=prompt,
        temperature=0.9,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0.0,
        # frequency_penalty=15,
        presence_penalty=0.6
        # stop=['\n']
    )
    logging.info("completions result:" + '\n' + pformat(answer))
    return {"data": answer["choices"][0]["text"].lstrip()}


if __name__ == "__main__":

    run_uvicorn_loguru(
        uvicorn.Config(
            "main:app",
            host=CONFIG["app"]["host"],
            port=int(CONFIG["app"]["port"]),
            log_level=CONFIG["app"]["log_level"],
        )
    )
