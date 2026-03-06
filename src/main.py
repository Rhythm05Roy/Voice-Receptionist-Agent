import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import AsyncClient
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.deps import limiter
from src.api.routers import api_router
from src.api.exceptions import register_exception_handlers
from src.config import get_settings
from src.middleware import RequestIDMiddleware, get_request_id, APIKeyAuthMiddleware

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


def _create_session_manager(settings_obj):
    """Create the appropriate session manager based on configuration."""
    if settings_obj.redis_url:
        try:
            from redis.asyncio import Redis
            from src.core.conversation.redis_session import RedisSessionManager

            redis_client = Redis.from_url(settings_obj.redis_url, decode_responses=True)
            logger.info("Using Redis session manager", url=settings_obj.redis_url)
            return redis_client, RedisSessionManager(redis=redis_client)
        except ImportError:
            logger.warning("redis package not installed, falling back to in-memory sessions")
        except Exception:  # noqa: BLE001
            logger.warning("Failed to connect to Redis, falling back to in-memory")

    from src.core.conversation.engine import CallSessionManager
    logger.info("Using in-memory session manager")
    return None, CallSessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client = AsyncClient(timeout=settings.request_timeout)
    app.state.http_client = http_client
    logger.info("HTTP client created")

    # Initialize session manager
    redis_client, session_manager = _create_session_manager(settings)
    app.state.session_manager = session_manager
    app.state.redis_client = redis_client

    # Initialize ConversationEngine singleton during startup
    from src.core.services.backend_client import BackendClient
    from src.core.services.elevenlabs import ElevenLabsClient
    from src.core.services.openai import OpenAIClient
    from src.core.conversation.engine import ConversationEngine

    backend_client = BackendClient(
        client=http_client,
        base_url=str(settings.backend_base_url),
        api_key=settings.backend_api_key,
        local_test_mode=settings.local_test_mode,
    )
    llm_client = OpenAIClient(api_key=settings.openai_api_key, model="gpt-4o")
    tts_client = ElevenLabsClient(
        client=http_client,
        api_key=settings.elevenlabs_api_key,
        default_voice_id=settings.elevenlabs_voice_id,
    )
    engine = ConversationEngine(
        backend_client=backend_client,
        llm_client=llm_client,
        tts_client=tts_client,
        environment=settings.environment,
        session_manager=session_manager,
    )
    app.state.conversation_engine = engine
    logger.info("ConversationEngine initialized")

    try:
        yield
    finally:
        if redis_client:
            await redis_client.aclose()
            logger.info("Redis connection closed")
        await http_client.aclose()
        logger.info("HTTP client closed")


app = FastAPI(title="AI Voice Service", version="0.2.0", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key auth for management endpoints
app.add_middleware(APIKeyAuthMiddleware)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

register_exception_handlers(app)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/health")
async def health():
    """Health check with dependency status."""
    checks: dict[str, str] = {}

    # Redis health
    session_manager = getattr(app.state, "session_manager", None)
    if session_manager:
        try:
            redis_ok = await session_manager.ping()
            checks["sessions"] = "healthy" if redis_ok else "degraded"
        except Exception:  # noqa: BLE001
            checks["sessions"] = "unhealthy"
    else:
        checks["sessions"] = "not_configured"

    # Backend health
    http_client = getattr(app.state, "http_client", None)
    if http_client:
        try:
            resp = await http_client.get(f"{settings.backend_base_url}/health", timeout=3)
            checks["backend"] = "healthy" if resp.status_code == 200 else "degraded"
        except Exception:  # noqa: BLE001
            checks["backend"] = "unreachable"
    else:
        checks["backend"] = "not_configured"

    overall = "healthy" if all(v in ("healthy", "not_configured") for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}
