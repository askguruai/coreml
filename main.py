import logging
from pprint import pformat
from typing import List

import openai
import uvicorn
from fastapi import FastAPI, HTTPException, status

from data import PROMPT_GENERAL, PROMPT_NO_INFO
from utils import CONFIG
from utils.api import (
    CompletionsInput,
    CompletionsResponse,
    EmbeddingsInput,
    EmbeddingsResponse,
    HTTPExceptionResponse,
)
from utils.logging import run_uvicorn_loguru

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post(
    "/embeddings/",
    response_model=EmbeddingsResponse,
    responses={500: {"model": HTTPExceptionResponse}},
)
async def get_embeddings(embeddings_input: EmbeddingsInput):
    logging.info(
        f"Number of texts to embed: {1 if type(embeddings_input.input) == str else len(embeddings_input.input)}"
    )
    try:
        print(kek)
        embeddings = openai.Embedding.create(
            input=embeddings_input.input, model=CONFIG["embeddings"]["model"]
        )["data"]
        return EmbeddingsResponse(data=embeddings)
    # except openai.error.APIError as e:
    except Exception as e:
        logging.error(f"OpenAI API returned an API Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenAI API returned an API Error: {e}",
        )


@app.post(
    "/completions/",
    response_model=CompletionsResponse,
    responses={500: {"model": HTTPExceptionResponse}},
)
async def get_completions(completions_input: CompletionsInput):
    prompt = (
        PROMPT_GENERAL(completions_input.info, completions_input.query)
        if completions_input.info
        else PROMPT_NO_INFO(completions_input.query)
    )
    logging.info("completions request:" + '\n' + prompt)
    try:
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
        return CompletionsResponse(data=answer["choices"][0]["text"].lstrip())
    except openai.error.APIError as e:
        logging.error(f"OpenAI API returned an API Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenAI API returned an API Error: {e}",
        )


if __name__ == "__main__":

    run_uvicorn_loguru(
        uvicorn.Config(
            "main:app",
            host=CONFIG["app"]["host"],
            port=int(CONFIG["app"]["port"]),
            log_level=CONFIG["app"]["log_level"],
        )
    )
