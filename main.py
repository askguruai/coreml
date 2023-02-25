import logging
from pprint import pformat
from typing import List

import openai
import uvicorn
from fastapi import FastAPI, HTTPException, status

from data import EMBEDDING_INSTRUCTION, PROMPT_GENERAL, PROMPT_NO_INFO
from ml import CompletionModel, EmbeddingModel
from utils import CONFIG
from utils.api import (CompletionsInput, CompletionsResponse, EmbeddingsInput,
                       EmbeddingsResponse, HTTPExceptionResponse)
from utils.logging import run_uvicorn_loguru

embedding_model = EmbeddingModel(
    model_name=CONFIG["embeddings"]["model"], device=CONFIG["embeddings"]["device"]
)
completion_model = CompletionModel()

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post(
    "/embeddings/",
    response_model=EmbeddingsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
async def get_embeddings(embeddings_input: EmbeddingsInput):
    logging.info(f"Number of texts to embed: {len(embeddings_input.input)}")
    try:
        embeddings = embedding_model.get_embeddings(
            input=embeddings_input.input,
            instruction=embeddings_input.instruction
            if embeddings_input.instruction
            else EMBEDDING_INSTRUCTION,
        )
        return EmbeddingsResponse(data=embeddings)
    except Exception as e:
        logging.error(f"{e.__class__.__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{e.__class__.__name__}: {e}",
        )


@app.post(
    "/completions/",
    response_model=CompletionsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
async def get_completions(completions_input: CompletionsInput):
    prompt = (
        PROMPT_GENERAL(completions_input.info, completions_input.query)
        if completions_input.info
        else PROMPT_NO_INFO(completions_input.query)
    )
    logging.info("completions request:" + "\n" + prompt)
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
        logging.info(
            f"completions result:\n{answer['choices'][0]['text'].lstrip()}\n\ntokens used: {answer['usage']['total_tokens']}"
        )
        return CompletionsResponse(data=answer["choices"][0]["text"].lstrip())
    except Exception as e:
        logging.error(f"{e.__class__.__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{e.__class__.__name__}: {e}",
        )


if __name__ == "__main__":

    run_uvicorn_loguru(
        uvicorn.Config(
            "main:app",
            host=CONFIG["app"]["host"],
            port=int(CONFIG["app"]["port"]),
            log_level=CONFIG["app"]["log_level"],
            root_path=CONFIG["app"]["root_path"],
        )
    )
