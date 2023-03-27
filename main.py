import logging
import time
from pprint import pformat
from typing import List

import openai
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from torch import multiprocessing

from data import (
    COMPLETIONS_PROMPT_OPENAI,
    COMPLETIONS_PROMPT_OPENAI_NO_INFO,
    COMPLETIONS_PROMPT_OPENAI_SYSTEM,
    EMBEDDING_INSTRUCTION,
)
from ml import EmbeddingModel
from ml.completions import CompletionModel, OpenAICompletionModel, T5CompletionModel
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


@app.on_event("startup")
def init_globals():
    global manager
    manager = multiprocessing.Manager()

    global embedding_model, t5_completion_model, openai_completion_model
    openai_completion_model = OpenAICompletionModel(
        model_name=CONFIG["v1.completions"]["model"],
    )
    t5_completion_model = T5CompletionModel(
        model_name=CONFIG["v2.completions"]["model"],
        device=CONFIG["v2.completions"]["device"],
    )
    embedding_model = EmbeddingModel(
        model_name=CONFIG["v2.embeddings"]["model"], device=CONFIG["v2.embeddings"]["device"]
    )


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url=f"{CONFIG['app']['root_path']}/v1/docs")


v1 = FastAPI()


@v1.get("/")
async def docs_redirect():
    return RedirectResponse(url=f"{CONFIG['app']['root_path']}/v1/docs")


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
    answer = openai_completion_model.get_completion(completions_input=completions_input)
    return CompletionsResponse(data=answer)


app.mount("/v1", v1)


v2 = FastAPI()


@v2.get("/")
async def docs_redirect():
    return RedirectResponse(url=f"{CONFIG['app']['root_path']}/v2/docs")


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
    completion = t5_completion_model.get_completion(completions_input=completions_input)
    return CompletionsResponse(data=completion)


app.mount("/v2", v2)


v3 = FastAPI()


@v3.get("/")
async def docs_redirect():
    return RedirectResponse(url=f"{CONFIG['app']['root_path']}/v3/docs")


@v3.post(
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


@v3.post(
    "/completions/",
    response_model=CompletionsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_completions(completions_input: CompletionsInput):
    logging.info("Received completions request")

    state = manager.dict()
    jobs = []
    for completion_model in [openai_completion_model, t5_completion_model]:
        job = multiprocessing.Process(
            target=CompletionModel.get_completion_subprocess,
            args=(completion_model, completions_input, state),
        )
        job.start()
        jobs.append(job)

    logging.info(f"Started {len(jobs)} jobs")

    start = time.time()
    while time.time() - start < int(CONFIG["v3.completions"]["general_timeout"]):
        if openai_completion_model.__class__.__name__ in state or (
            time.time() - start > int(CONFIG["v3.completions"]["openai_timeout"]) and state
        ):
            break
        time.sleep(0.1)

    logging.info(f"Finished in {round(time.time() - start, 2)} seconds with state {state}")

    for job in jobs:
        job.terminate()

    if not state:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Timeout",
        )

    completion = (
        state[openai_completion_model.__class__.__name__]
        if openai_completion_model.__class__.__name__ in state
        else state[t5_completion_model.__class__.__name__]
    )

    return CompletionsResponse(data=completion)


app.mount("/v3", v3)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    run_uvicorn_loguru(
        uvicorn.Config(
            "main:app",
            host=CONFIG["app"]["host"],
            port=int(CONFIG["app"]["port"]),
            log_level=CONFIG["app"]["log_level"],
            root_path=CONFIG["app"]["root_path"],
        )
    )
