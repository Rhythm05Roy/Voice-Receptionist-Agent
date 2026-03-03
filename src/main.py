import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from httpx import AsyncClient
from loguru import logger

from src.api.routers import api_router
from src.api.exceptions import register_exception_handlers
from src.config import get_settings
from src.middleware import RequestIDMiddleware, get_request_id

settings = get_settings()

logger.remove()
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra[request_id]} | {message}",
    enqueue=True,
    backtrace=False,
    diagnose=False,
)
logger.configure(extra={"request_id": "-"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client = AsyncClient(timeout=settings.request_timeout)
    app.state.http_client = http_client
    logger.info("HTTP client created")
    try:
        yield
    finally:
        await http_client.aclose()
        logger.info("HTTP client closed")


app = FastAPI(title="AI Voice Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)
register_exception_handlers(app)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/health")
async def health():
    return {"status": "healthy"}
