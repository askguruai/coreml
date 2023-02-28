import logging
from pprint import pformat
from typing import List

import openai
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse

from data import (
    COMPLETIONS_PROMPT_CUSTOM,
    COMPLETIONS_PROMPT_OPENAI,
    COMPLETIONS_PROMPT_OPENAI_NO_INFO,
    EMBEDDING_INSTRUCTION,
)
from ml import CompletionModel, EmbeddingModel
from utils import CONFIG
from utils.api import catch_errors
from utils.logging import run_uvicorn_loguru
from utils.schemas import (
    CompletionsInput,
    CompletionsResponse,
    EmbeddingsInput,
    EmbeddingsInputInstruction,
    EmbeddingsResponse,
    HTTPExceptionResponse,
)

app = FastAPI()


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/v1")


v1 = FastAPI()


@v1.get("/")
async def docs_redirect():
    return RedirectResponse(url="/v1/docs")


@v1.post(
    "/embeddings/",
    response_model=EmbeddingsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_embeddings(embeddings_input: EmbeddingsInput):
    logging.info(f"Number of texts to embed: {len(embeddings_input.input)}")
    embeddings = openai.Embedding.create(
        input=embeddings_input.input, model=CONFIG["v1.embeddings"]["model"]
    )["data"]
    embeddings = [embedding["embedding"] for embedding in embeddings]
    return EmbeddingsResponse(data=embeddings)


@v1.post(
    "/completions/",
    response_model=CompletionsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_completions(completions_input: CompletionsInput):
    prompt = (
        COMPLETIONS_PROMPT_OPENAI(completions_input.info, completions_input.query)
        if completions_input.info
        else COMPLETIONS_PROMPT_OPENAI_NO_INFO(completions_input.query)
    )
    logging.info("completions request:" + '\n' + prompt)
    answer = openai.Completion.create(
        model=CONFIG["v1.completions"]["model"],
        prompt=prompt,
        temperature=0.9,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
    )["choices"][0]["text"].lstrip()
    logging.info("completions result:" + '\n' + answer)
    return CompletionsResponse(data=answer)


app.mount("/v1", v1)


v2 = FastAPI()


@v2.get("/")
async def docs_redirect():
    return RedirectResponse(url="/v2/docs")


@app.on_event("startup")
def init_models():
    global embedding_model, completion_model
    completion_model = CompletionModel(
        model_name=CONFIG["v2.completions"]["model"],
        device=CONFIG["v2.completions"]["device"],
    )
    embedding_model = EmbeddingModel(
        model_name=CONFIG["v2.embeddings"]["model"], device=CONFIG["v2.embeddings"]["device"]
    )


@v2.post(
    "/embeddings/",
    response_model=EmbeddingsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_embeddings(embeddings_input: EmbeddingsInputInstruction):
    logging.info(f"Number of texts to embed: {len(embeddings_input.input)}")
    embeddings = embedding_model.get_embeddings(
        input=embeddings_input.input,
        instruction=embeddings_input.instruction
        if embeddings_input.instruction
        else EMBEDDING_INSTRUCTION,
    )
    return EmbeddingsResponse(data=embeddings)


@v2.post(
    "/completions/",
    response_model=CompletionsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_completions(completions_input: CompletionsInput):
    prompt = (
        COMPLETIONS_PROMPT_CUSTOM(
            completions_input.info.replace("\n\n", "\n"), completions_input.query
        )
        if completions_input.info
        else COMPLETIONS_PROMPT_OPENAI_NO_INFO(completions_input.query)
    )
    logging.info("completions request:" + "\n" + prompt)
    completion = completion_model.get_completion(prompt=prompt)
    logging.info(f"completions result:\n{completion}")
    return CompletionsResponse(data=completion)


app.mount("/v2", v2)


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
