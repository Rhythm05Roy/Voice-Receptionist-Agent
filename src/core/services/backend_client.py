import json
import os
import time
from pathlib import Path
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
        self._dummy_db_cache: dict[str, Any] | None = None
        self._runtime_bookings: dict[str, dict[str, Any]] = {}

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

    def _dummy_db_path(self) -> Path:
        configured = os.getenv("DUMMY_DB_PATH", "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        project_root = Path(__file__).resolve().parents[3]
        return project_root / "tests" / "dummy_business_db.json"

    def _default_dummy_db(self) -> dict[str, Any]:
        return {
            "agents": {
                "default": {
                    "agent_id": "local-agent",
                    "business_name": "Bahrain HomeCare Group",
                    "greeting": "Hello, this is Bahrain HomeCare Group. How can I help you today?",
                    "language": "en",
                    "coverage_country": "Bahrain",
                    "coverage_areas": [
                        "Manama",
                        "Riffa",
                        "Muharraq",
                        "Isa Town",
                        "Hamad Town",
                        "A'ali",
                        "Saar",
                        "Seef",
                    ],
                    "fallback_phone": "+97317000000",
                    "intake_questions": [
                        "What service are you looking for?",
                        "Where are you located?",
                        "When is the best time for a visit?",
                    ],
                    "booking_required_fields": [
                        "service_type",
                        "location",
                        "preferred_time",
                        "contact_number",
                    ],
                    "service_catalog": [
                        {
                            "service_id": "ac_repair",
                            "name": "AC Repair and Maintenance",
                            "description": "Diagnosis, gas check, filter cleaning, cooling optimization.",
                            "base_price_bhd": "10-35 BHD",
                            "price_note": "Final quote depends on spare parts and compressor condition.",
                            "keywords": ["ac", "air conditioner", "cooling", "hvac", "split unit"],
                        },
                        {
                            "service_id": "deep_cleaning",
                            "name": "Home Deep Cleaning",
                            "description": "Kitchen, bathroom, floors, glass, and optional sanitization package.",
                            "base_price_bhd": "12-45 BHD",
                            "price_note": "Price varies by home size and selected add-ons.",
                            "keywords": ["cleaning", "deep clean", "house cleaning", "sanitize", "maid"],
                        },
                        {
                            "service_id": "salon_home",
                            "name": "At-Home Salon",
                            "description": "Haircut, beard trim, basic grooming at customer location.",
                            "base_price_bhd": "8-25 BHD",
                            "price_note": "Premium stylist and late-night slots cost extra.",
                            "keywords": ["salon", "haircut", "beard", "grooming", "stylist"],
                        },
                        {
                            "service_id": "general_maintenance",
                            "name": "General Maintenance",
                            "description": "Minor electrical, plumbing, and fixture fixes.",
                            "base_price_bhd": "9-30 BHD",
                            "price_note": "Materials are billed separately when needed.",
                            "keywords": ["maintenance", "plumbing", "electric", "fix", "repair"],
                        },
                    ],
                    "faqs": {
                        "pricing": "Pricing depends on service type, location, urgency, and parts.",
                        "availability": "Same-day slots are possible based on technician availability.",
                        "payment": "Cash, card, and benefit pay are supported.",
                    },
                    "notes": [
                        "This is dummy data for local development.",
                        "No real booking is created outside this service.",
                    ],
                }
            }
        }

    def _load_dummy_db(self) -> dict[str, Any]:
        if self._dummy_db_cache is not None:
            return self._dummy_db_cache

        path = self._dummy_db_path()
        if path.exists():
            try:
                self._dummy_db_cache = json.loads(path.read_text(encoding="utf-8-sig"))
                logger.info("Loaded dummy business database", path=str(path))
                return self._dummy_db_cache
            except Exception:  # noqa: BLE001
                logger.exception("Failed to parse dummy business database; using built-in defaults")

        self._dummy_db_cache = self._default_dummy_db()
        return self._dummy_db_cache

    def _fallback_agent_config(self, agent_id: str | None = None) -> AgentConfig:
        db = self._load_dummy_db()
        agents = db.get("agents", {}) if isinstance(db, dict) else {}
        key = agent_id or "default"
        payload = agents.get(key) or agents.get("default") or self._default_dummy_db()["agents"]["default"]
        return AgentConfig(**payload)

    def _find_service_in_catalog(self, answer: str, agent: AgentConfig) -> dict[str, Any] | None:
        text = (answer or "").lower()
        for service in agent.service_catalog:
            candidates = [service.name, service.service_id, *service.keywords]
            if any(token.lower() in text for token in candidates if token):
                return service.model_dump()
        return None

    @staticmethod
    def _agent_to_ui_context(agent: AgentConfig) -> dict[str, Any]:
        return {
            "agent_id": agent.agent_id,
            "business_name": agent.business_name,
            "greeting": agent.greeting,
            "language": agent.language,
            "coverage_country": agent.coverage_country,
            "coverage_areas": list(agent.coverage_areas),
            "booking_required_fields": list(agent.booking_required_fields),
            "fallback_phone": agent.fallback_phone,
            "services": [service.model_dump() for service in agent.service_catalog],
            "faqs": dict(agent.faqs),
            "notes": list(agent.notes),
        }

    async def fetch_agent_ui_context(self, agent_id: str | None = None) -> dict[str, Any]:
        if self._local_fallback_enabled():
            agent = self._fallback_agent_config(agent_id)
            return self._agent_to_ui_context(agent)

        url = f"{self.base_url}/agents/{agent_id or 'default'}/ui-context"
        try:
            payload = await self._get(url)
            if isinstance(payload, dict):
                return payload
            raise BackendCommunicationError("Invalid UI context payload")
        except Exception as exc:  # noqa: BLE001
            raise BackendCommunicationError(str(exc)) from exc

    async def answer_business_query(self, query: str, agent_id: str | None = None) -> dict[str, Any]:
        agent = await self.fetch_agent_config(agent_id)
        lowered = query.lower()

        suggested_services = [service.name for service in agent.service_catalog[:4]]

        faq_matches: list[str] = []
        wants_services = any(k in lowered for k in ["service", "services", "provide", "offer", "business"])
        wants_pricing = any(k in lowered for k in ["price", "pricing", "cost", "quote", "rate"])
        for key, value in agent.faqs.items():
            if key.lower() in lowered:
                faq_matches.append(value)
        if wants_pricing:
            for service in agent.service_catalog[:4]:
                faq_matches.append(f"{service.name}: {service.base_price_bhd}. {service.price_note}".strip())
        if wants_services:
            service_names = ", ".join(suggested_services) or "AC Repair, Cleaning, and Maintenance"
            faq_matches.append(
                f"{agent.business_name} provides {service_names}. Coverage is {agent.coverage_country}."
            )

        if faq_matches:
            answer = " ".join(faq_matches[:3]).strip()
        else:
            answer = (
                f"{agent.business_name} can help with booking, pricing guidance, and service availability in "
                f"{agent.coverage_country}. Tell me the exact service to continue."
            )

        return {"answer": answer, "suggested_services": suggested_services}

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
            agent = self._fallback_agent_config(agent_id)
            service_answer = answers.get("q0", "")
            location_answer = answers.get("q1", "")
            time_answer = answers.get("q2", "")
            matched = self._find_service_in_catalog(service_answer, agent)
            booking_ref = f"DUMMY-{int(time.time())}"

            if not matched:
                return {
                    "status": "unsupported_service",
                    "booking_ref": booking_ref,
                    "message": (
                        "We currently support AC repair, home cleaning, at-home salon, and general maintenance. "
                        "Please choose one of these services and we can continue."
                    ),
                    "answers": answers,
                }

            payload = {
                "agent_id": agent.agent_id,
                "status": "mock_confirmed",
                "booking_ref": booking_ref,
                "service_id": matched["service_id"],
                "service_name": matched["name"],
                "location": location_answer,
                "preferred_time": time_answer,
                "message": (
                    f"Booking {booking_ref} is recorded for {matched['name']} in {location_answer} at {time_answer}. "
                    "Operations team will confirm final timing and quote shortly."
                ),
                "price_range": matched.get("base_price_bhd", "") or "",
                "price_note": matched.get("price_note", "") or "",
                "answers": answers,
            }
            self._runtime_bookings[booking_ref] = payload
            return payload

        url = f"{self.base_url}/agents/{agent_id}/bookings"
        try:
            payload = await self._post(url, json={"answers": answers})
            logger.info("Booking created", agent_id=agent_id)
            return payload
        except Exception as exc:  # noqa: BLE001
            raise BackendCommunicationError(str(exc)) from exc

    async def track_booking(self, booking_id: str, agent_id: str | None = None) -> dict[str, Any]:
        normalized_id = booking_id.strip()
        if not normalized_id:
            raise BackendCommunicationError("booking_id is required")

        if self._local_fallback_enabled():
            if normalized_id in self._runtime_bookings:
                return self._runtime_bookings[normalized_id]

            db = self._load_dummy_db()
            bookings = db.get("bookings", {}) if isinstance(db, dict) else {}
            payload = bookings.get(normalized_id)
            if payload:
                return payload

            return {
                "status": "not_found",
                "booking_ref": normalized_id,
                "message": "Booking ID was not found in the current dummy dataset.",
            }

        url = f"{self.base_url}/agents/{agent_id or 'default'}/bookings/{normalized_id}"
        try:
            payload = await self._get(url)
            if isinstance(payload, dict):
                return payload
            raise BackendCommunicationError("Invalid booking status payload")
        except Exception as exc:  # noqa: BLE001
            raise BackendCommunicationError(str(exc)) from exc

    def _normalize_agent_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "agent_id": payload.get("id") or payload.get("agent_id", "default"),
            "business_name": payload.get("business_name", "Local Home Services Bahrain"),
            "greeting": payload.get("greeting") or payload.get("opening_line", "Hello!"),
            "intake_questions": payload.get("intake_questions", []),
            "language": payload.get("language", "en"),
            "fallback_phone": payload.get("fallback_phone"),
            "coverage_country": payload.get("coverage_country", "Bahrain"),
            "coverage_areas": payload.get("coverage_areas", []),
            "service_catalog": payload.get("service_catalog", []),
            "booking_required_fields": payload.get("booking_required_fields", []),
            "faqs": payload.get("faqs", {}),
            "notes": payload.get("notes", []),
        }
