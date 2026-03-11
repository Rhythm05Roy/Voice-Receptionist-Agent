"""API key authentication middleware for management endpoints.

Webhook endpoints bypass this (they use Vonage signature verification).
"""

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.config import get_settings


# Paths that skip API key auth (webhooks, health, docs)
_BYPASS_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/telephony/webhook",
    "/api/v1/twilio/webhook",
)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Require Bearer token for management endpoints when API_AUTH_TOKEN is set."""

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()

        # Skip auth if no token configured or path is bypassed
        if not settings.api_auth_token:
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(prefix) for prefix in _BYPASS_PREFIXES):
            return await call_next(request)

        # Check Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(
                content='{"detail":"Missing authentication token"}',
                status_code=status.HTTP_401_UNAUTHORIZED,
                media_type="application/json",
            )

        token = auth_header.split(" ", 1)[1].strip()
        if token != settings.api_auth_token:
            return Response(
                content='{"detail":"Invalid authentication token"}',
                status_code=status.HTTP_403_FORBIDDEN,
                media_type="application/json",
            )

        return await call_next(request)
