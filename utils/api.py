import traceback
from functools import wraps

from fastapi import HTTPException, status
from loguru import logger


def catch_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{e.__class__.__name__}: {e}")
            traceback.print_exc()
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{e.__class__.__name__}: {e}",
            )

    return wrapper
