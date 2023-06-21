import threading
import time
from pprint import pformat
from typing import List

import openai
import uvicorn
from async_lru import alru_cache
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from loguru import logger
from torch import multiprocessing

from data import EMBEDDING_INSTRUCTION
from ml import EmbeddingModel
from ml.completions import AlpacaCompletionModel, CompletionModel, OpenAICompletionModel, T5CompletionModel
from utils import CONFIG
from utils.api import catch_errors
from utils.gunicorn_logging import run_gunicorn_loguru
from utils.misc import retry_with_time_limit
from utils.schemas import (
    AnswerInContextResponse,
    ApiVersion,
    CompletionsInput,
    CompletionsResponse,
    EmbeddingsInput,
    EmbeddingsInputInstruction,
    EmbeddingsResponse,
    HTTPExceptionResponse,
    SummarizationInput,
)

app = FastAPI()


@app.on_event("startup")
def init_globals():
    global manager, multiprocessing_context_fork, multiprocessing_context_forkserver
    manager = multiprocessing.Manager()
    multiprocessing_context_fork = multiprocessing.get_context("fork")
    multiprocessing_context_forkserver = multiprocessing.get_context("forkserver")

    global embedding_model, alpaca_completion_model, openai_completion_model
    openai_completion_model = OpenAICompletionModel(
        model_name=CONFIG["v1.completions"]["model"],
    )
    # alpaca_completion_model = AlpacaCompletionModel(
    #     model_name=CONFIG["v2.completions"]["model"],
    #     device=CONFIG["v2.completions"]["device"],
    # )
    # embedding_model = EmbeddingModel(
    #     model_name=CONFIG["v2.embeddings"]["model"], device=CONFIG["v2.embeddings"]["device"]
    # )


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url=f"/docs")


@app.post(
    "/{api_version}/embeddings/",
    response_model=EmbeddingsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
@alru_cache(maxsize=512)
async def get_embeddings(api_version: ApiVersion, embeddings_input: EmbeddingsInput):
    logger.info(f"Number of texts to embed: {len(embeddings_input.input)}")
    args = {
        "input": embeddings_input.input,
        "model": CONFIG["v1.embeddings"]["model"],
    }
    if len(embeddings_input.input) == 1:
        embeddings = await retry_with_time_limit(openai.Embedding.acreate, time_limit=2, max_retries=10, **args)
    else:
        embeddings = await openai.Embedding.acreate(**args)
    embeddings = [embedding["embedding"] for embedding in embeddings["data"]]
    return EmbeddingsResponse(data=embeddings)


# @alru_cache(maxsize=512)
@app.post(
    "/{api_version}/completions/",
    response_model=CompletionsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_completions(api_version: ApiVersion, completions_input: CompletionsInput):
    return await openai_completion_model.get_completion(completions_input=completions_input, api_version=api_version)


@app.post(
    "/{api_version}/summarization/",
    response_model=CompletionsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_completions(api_version: ApiVersion, summarization_input: SummarizationInput):
    return await openai_completion_model.get_summarization(
        summarization_input=summarization_input, api_version=api_version
    )


# check if answer is in the context
@app.post(
    "/{api_version}/if_answer_in_context/",
    response_model=AnswerInContextResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def if_answer_in_context(api_version: ApiVersion, completions_input: CompletionsInput):
    return await openai_completion_model.if_answer_in_context(
        completions_input=completions_input, api_version=api_version
    )


v2 = FastAPI()


@v2.get("/")
async def docs_redirect():
    return RedirectResponse(url=f"/v2/docs")


@v2.post(
    "/embeddings/",
    response_model=EmbeddingsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_embeddings(embeddings_input: EmbeddingsInputInstruction):
    logger.info(f"Number of texts to embed: {len(embeddings_input.input)}")
    embeddings = embedding_model.get_embeddings(
        input=embeddings_input.input,
        instruction=embeddings_input.instruction if embeddings_input.instruction else EMBEDDING_INSTRUCTION,
    )
    return EmbeddingsResponse(data=embeddings)


@v2.post(
    "/completions/",
    response_model=CompletionsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_completions(completions_input: CompletionsInput):
    completion = alpaca_completion_model.get_completion(completions_input=completions_input)
    return CompletionsResponse(data=completion)


# app.mount("/v2", v2)


v3 = FastAPI()


@v3.get("/")
async def docs_redirect():
    return RedirectResponse(url=f"/v3/docs")


# @v3.post(
#     "/embeddings/",
#     response_model=EmbeddingsResponse,
#     responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
# )
# @catch_errors
# async def get_embeddings(embeddings_input: EmbeddingsInputInstruction):
#     logger.info(f"Number of texts to embed: {len(embeddings_input.input)}")
#     embeddings = embedding_model.get_embeddings(
#         input=embeddings_input.input,
#         instruction=embeddings_input.instruction
#         if embeddings_input.instruction
#         else EMBEDDING_INSTRUCTION,
#     )
#     return EmbeddingsResponse(data=embeddings)


@v3.post(
    "/embeddings/",
    response_model=EmbeddingsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_embeddings(embeddings_input: EmbeddingsInput):
    logger.info(f"Number of texts to embed: {len(embeddings_input.input)}")
    embeddings = openai.Embedding.create(input=embeddings_input.input, model=CONFIG["v1.embeddings"]["model"])["data"]
    embeddings = [embedding["embedding"] for embedding in embeddings]
    return EmbeddingsResponse(data=embeddings)


@v3.post(
    "/completions/",
    response_model=CompletionsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_completions(completions_input: CompletionsInput):
    logger.info("Received completions request")

    state = manager.dict()
    for completion_model, context in zip(
        [openai_completion_model, alpaca_completion_model],
        [multiprocessing_context_fork, multiprocessing_context_forkserver],
    ):
        job = context.Process(
            target=CompletionModel.get_completion_subprocess,
            args=(completion_model, completions_input, state),
            name=completion_model.__class__.__name__,
        )
        # job.start takes a lot of time itself
        # and while we don't want to wait for starting
        # second job if first one is already finished
        thread = threading.Thread(target=job.start)
        thread.start()
        logger.info(f"Started thread with job {job.name}")

    start = time.time()
    while time.time() - start < int(CONFIG["v3.completions"]["general_timeout"]):
        if openai_completion_model.__class__.__name__ in state or (
            time.time() - start > int(CONFIG["v3.completions"]["openai_timeout"]) and state
        ):
            break
        time.sleep(0.1)

    logger.info(f"Finished in {round(time.time() - start, 2)} seconds with state {state}")

    if not state:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Timeout",
        )

    completion = (
        state[openai_completion_model.__class__.__name__]
        if openai_completion_model.__class__.__name__ in state
        else state[alpaca_completion_model.__class__.__name__]
    )

    return CompletionsResponse(data=completion)


# app.mount("/v3", v3)


if __name__ == "__main__":
    options = {
        "bind": CONFIG["app"]["host"] + ':' + CONFIG["app"]["port"],
        "workers": CONFIG["app"]["workers"],
        "timeout": CONFIG["app"]["timeout"],
    }
    run_gunicorn_loguru(app, options)
