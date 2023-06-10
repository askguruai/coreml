import asyncio
import time

from loguru import logger


async def retry_with_time_limit(func, time_limit=20, max_retries=3, *args, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=time_limit)
            return result
        except asyncio.TimeoutError:
            retries += 1
            logger.warning(f"Retry #{retries} out of {max_retries}")

    logger.error("Max retries reached, returning None")
    return None
