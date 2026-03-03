import time
from typing import Any

from httpx import AsyncClient, HTTPError, HTTPStatusError, TimeoutException
from loguru import logger
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.api.exceptions import BackendCommunicationError
from src.core.types import AgentConfig


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, TimeoutException):
        return True
    if isinstance(exc, HTTPStatusError) and exc.response is not None:
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return False


class BackendClient:
    def __init__(self, client: AsyncClient, base_url: str, api_key: str, local_test_mode: bool = False):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.local_test_mode = local_test_mode
        self.failure_count = 0
        self.circuit_open_until: float | None = None

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _ensure_circuit(self) -> None:
        if self.circuit_open_until and time.monotonic() < self.circuit_open_until:
            raise BackendCommunicationError("Backend circuit open, skipping call")
        if self.circuit_open_until and time.monotonic() >= self.circuit_open_until:
            self.failure_count = 0
            self.circuit_open_until = None

    def _record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= 3:
            self.circuit_open_until = time.monotonic() + 30
            logger.warning("Backend circuit opened")

    def _record_success(self) -> None:
        self.failure_count = 0
        self.circuit_open_until = None

    def _local_fallback_enabled(self) -> bool:
        return self.local_test_mode or "api.your-backend.com" in self.base_url

    @staticmethod
    def _fallback_agent_config(agent_id: str | None = None) -> AgentConfig:
        return AgentConfig(
            agent_id=agent_id or "local-agent",
            greeting="Hello, this is your local AI assistant. How can I help you today?",
            intake_questions=[
                "What service do you need?",
                "Which area are you located in?",
                "What time works best for a visit?",
            ],
            language="en",
            fallback_phone=None,
        )

    @retry(
        reraise=True,
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
    async def _get(self, url: str) -> dict[str, Any]:
        self._ensure_circuit()
        try:
            response = await self.client.get(url, headers=self._headers)
            response.raise_for_status()
            self._record_success()
            return response.json()
        except HTTPError:
            self._record_failure()
            logger.exception("Backend communication failed", url=url)
            raise

    @retry(
        reraise=True,
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
    async def _post(self, url: str, json: dict[str, Any]) -> dict[str, Any]:
        self._ensure_circuit()
        try:
            response = await self.client.post(url, headers=self._headers, json=json)
            response.raise_for_status()
            self._record_success()
            return response.json()
        except HTTPError:
            self._record_failure()
            logger.exception("Backend communication failed", url=url)
            raise

    async def fetch_agent_config(self, agent_id: str | None = None) -> AgentConfig:
        if self._local_fallback_enabled():
            return self._fallback_agent_config(agent_id)

        url = f"{self.base_url}/agents/{agent_id or 'default'}/voice-config"
        try:
            payload = await self._get(url)
            logger.info("Fetched agent config", agent_id=agent_id)
            return AgentConfig(**self._normalize_agent_config(payload))
        except Exception as exc:  # noqa: BLE001
            raise BackendCommunicationError(str(exc)) from exc

    async def fetch_intake_questions(self, agent_id: str | None = None) -> list[str]:
        if self._local_fallback_enabled():
            return list(self._fallback_agent_config(agent_id).intake_questions)

        url = f"{self.base_url}/agents/{agent_id or 'default'}/intake-questions"
        try:
            payload = await self._get(url)
            questions = payload.get("questions", []) if isinstance(payload, dict) else payload
            logger.debug("Fetched intake questions", count=len(questions))
            return [str(q) for q in questions]
        except Exception as exc:  # noqa: BLE001
            raise BackendCommunicationError(str(exc)) from exc

    async def book_service(self, agent_id: str, answers: dict[str, str]) -> dict[str, Any]:
        if self._local_fallback_enabled():
            return {
                "status": "mock_confirmed",
                "message": "Local test booking recorded. Team will contact you shortly.",
                "answers": answers,
            }

        url = f"{self.base_url}/agents/{agent_id}/bookings"
        try:
            payload = await self._post(url, json={"answers": answers})
            logger.info("Booking created", agent_id=agent_id)
            return payload
        except Exception as exc:  # noqa: BLE001
            raise BackendCommunicationError(str(exc)) from exc

    def _normalize_agent_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "agent_id": payload.get("id") or payload.get("agent_id", "default"),
            "greeting": payload.get("greeting") or payload.get("opening_line", "Hello!"),
            "intake_questions": payload.get("intake_questions", []),
            "language": payload.get("language", "ar"),
            "fallback_phone": payload.get("fallback_phone"),
        }
