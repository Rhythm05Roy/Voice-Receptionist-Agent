"""Redis-backed session manager for production use.

Drop-in replacement for CallSessionManager that persists sessions
across restarts and multiple service instances.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from .engine import CallSession


class RedisSessionManager:
    """Async session storage backed by Redis with automatic TTL expiry.

    Maintains the same logical interface as the in-memory
    CallSessionManager, but all operations are async and durable.
    """

    def __init__(self, redis: Redis, ttl_seconds: int = 1800):
        self.redis = redis
        self.ttl = ttl_seconds

    @staticmethod
    def _key(call_id: str) -> str:
        return f"call:{call_id}"

    async def get(self, call_id: str) -> CallSession | None:
        from .engine import CallSession  # avoid circular import

        raw = await self.redis.get(self._key(call_id))
        if raw is None:
            return None

        try:
            session = CallSession.model_validate_json(raw)
            session.touch()
            # Refresh TTL on access
            await self.redis.setex(self._key(call_id), self.ttl, session.model_dump_json())
            return session
        except Exception:  # noqa: BLE001
            logger.warning("Failed to deserialize session from Redis", call_id=call_id)
            await self.redis.delete(self._key(call_id))
            return None

    async def save(self, session: CallSession) -> None:
        session.touch()
        await self.redis.setex(
            self._key(session.call_id),
            self.ttl,
            session.model_dump_json(),
        )

    async def delete(self, call_id: str) -> None:
        await self.redis.delete(self._key(call_id))

    async def prune(self) -> None:
        # Redis handles expiry automatically via TTL; no-op here.
        pass

    async def ping(self) -> bool:
        """Health check for Redis connectivity."""
        try:
            return await self.redis.ping()
        except Exception:  # noqa: BLE001
            return False


class InMemorySessionManagerAsync:
    """Async wrapper around in-memory storage for consistency with RedisSessionManager."""

    def __init__(self, ttl_seconds: int = 1800):
        self.ttl = ttl_seconds
        self._store: dict[str, CallSession] = {}

    @staticmethod
    def _key(call_id: str) -> str:
        return f"call:{call_id}"

    async def get(self, call_id: str) -> CallSession | None:
        session = self._store.get(self._key(call_id))
        if not session:
            return None

        now = time.monotonic()
        if now - session.last_activity > self.ttl:
            await self.delete(call_id)
            return None

        session.touch()
        return session

    async def save(self, session: CallSession) -> None:
        session.touch()
        self._store[self._key(session.call_id)] = session

    async def delete(self, call_id: str) -> None:
        self._store.pop(self._key(call_id), None)

    async def prune(self) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, session in self._store.items()
            if now - session.last_activity > self.ttl
        ]
        for key in expired:
            self._store.pop(key, None)

    async def ping(self) -> bool:
        return True
