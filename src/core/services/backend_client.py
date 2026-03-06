from __future__ import annotations

import json
import os
import re
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
        return exc.response.status_code == 429 or exc.response.status_code >= 500
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
            raise BackendCommunicationError("Backend circuit is open")
        if self.circuit_open_until and time.monotonic() >= self.circuit_open_until:
            self.failure_count = 0
            self.circuit_open_until = None

    def _record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= 3:
            self.circuit_open_until = time.monotonic() + 30
            logger.warning("Backend circuit opened for 30 seconds")

    def _record_success(self) -> None:
        self.failure_count = 0
        self.circuit_open_until = None

    def _local_fallback_enabled(self) -> bool:
        placeholder_hosts = {
            "api.your-backend.com",
            "example.com",
        }
        return self.local_test_mode or any(host in self.base_url for host in placeholder_hosts)

    def _dummy_db_path(self) -> Path:
        configured = os.getenv("DUMMY_DB_PATH", "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        project_root = Path(__file__).resolve().parents[3]
        return project_root / "tests" / "dummy_business_db.json"

    def _default_dummy_db(self) -> dict[str, Any]:
        return {
            "phone_routing": {
                "+97317000000": "default",
                "+97317111111": "premium",
            },
            "agents": {
                "default": {
                    "agent_id": "default",
                    "business_name": "Bahrain HomeCare Group",
                    "greeting": "Hello, this is Bahrain HomeCare Group. How can I help you today?",
                    "language": "en",
                    "multilingual_enabled": True,
                    "supported_languages": ["en", "ar", "hi"],
                    "default_greeting_language": "en",
                    "language_voice_map": {
                        "en": "elevenlabs-en-default",
                        "ar": "elevenlabs-ar-default",
                        "hi": "elevenlabs-hi-default",
                    },
                    "coverage_country": "Bahrain",
                    "coverage_areas": ["Manama", "Riffa", "Muharraq", "Saar", "Seef", "Juffair"],
                    "fallback_phone": "+97317000000",
                    "max_call_duration_minutes": 5,
                    "intake_questions": [
                        {
                            "key": "service_type",
                            "question": "What service do you need today?",
                            "answer_type": "text",
                            "required": True,
                        },
                        {
                            "key": "location",
                            "question": "Which area are you located in?",
                            "answer_type": "text",
                            "required": True,
                        },
                        {
                            "key": "preferred_time",
                            "question": "What time works best for a visit?",
                            "answer_type": "text",
                            "required": True,
                        },
                        {
                            "key": "allergy_sensitive",
                            "question": "Do you have any allergy or chemical sensitivity we should note?",
                            "answer_type": "yes_no",
                            "required": False,
                            "ask_when": "specific_services",
                            "service_tags": ["cleaning", "salon"],
                        },
                    ],
                    "booking_required_fields": ["service_type", "location", "preferred_time"],
                    "service_catalog": [
                        {
                            "service_id": "ac_repair",
                            "name": "AC Repair and Maintenance",
                            "description": "Diagnosis, refrigerant check, filter cleaning, cooling optimization.",
                            "base_price_bhd": "10-35 BHD",
                            "price_note": "Final quote depends on spare parts and compressor condition.",
                            "keywords": ["ac", "air conditioner", "cooling", "hvac", "split unit"],
                        },
                        {
                            "service_id": "deep_cleaning",
                            "name": "Home Deep Cleaning",
                            "description": "Kitchen, bathroom, floors, and optional sanitization package.",
                            "base_price_bhd": "12-45 BHD",
                            "price_note": "Price varies by home size and selected add-ons.",
                            "keywords": ["cleaning", "deep clean", "home cleaning", "sanitize"],
                        },
                        {
                            "service_id": "salon_home",
                            "name": "At-Home Salon",
                            "description": "Haircut, beard trim, and basic grooming at your location.",
                            "base_price_bhd": "8-25 BHD",
                            "price_note": "Premium stylist and late-night slots cost extra.",
                            "keywords": ["salon", "haircut", "beard", "stylist"],
                        },
                        {
                            "service_id": "general_maintenance",
                            "name": "General Maintenance",
                            "description": "Minor electrical, plumbing, and fixture fixes.",
                            "base_price_bhd": "9-30 BHD",
                            "price_note": "Materials are billed separately when needed.",
                            "keywords": ["maintenance", "plumbing", "electric", "repair"],
                        },
                    ],
                    "business_description": "Professional home services for homeowners and rental properties in Bahrain.",
                    "business_hours": "Mon-Sat 9:00 AM to 9:00 PM",
                    "cancellation_policy": "Cancellations within 2 hours may include a small visit fee.",
                    "payment_policy": "Cash, card, and BenefitPay are accepted.",
                    "deposit_policy": "Deposit required only for large jobs above 50 BHD.",
                    "faqs": {
                        "services": "We offer AC repair, deep cleaning, at-home salon, and general maintenance.",
                        "pricing": "Pricing depends on service type, area, urgency, and parts.",
                        "coverage": "We currently serve Bahrain areas only.",
                    },
                    "additional_information": [
                        "Same-day slots are available based on team capacity.",
                        "An operations specialist confirms final quote and timing.",
                    ],
                    "notes": ["Dummy data for local AI route development."],
                }
            },
            "bookings": {
                "DUMMY-10001": {
                    "status": "scheduled",
                    "booking_ref": "DUMMY-10001",
                    "message": "Your AC technician is scheduled today between 6:00 PM and 7:00 PM.",
                    "service_name": "AC Repair and Maintenance",
                    "location": "Manama",
                    "preferred_time": "6:00 PM",
                }
            },
        }

    def _load_dummy_db(self) -> dict[str, Any]:
        if self._dummy_db_cache is not None:
            return self._dummy_db_cache

        db_path = self._dummy_db_path()
        if db_path.exists():
            try:
                self._dummy_db_cache = json.loads(db_path.read_text(encoding="utf-8-sig"))
                logger.info("Loaded dummy business database", path=str(db_path))
                return self._dummy_db_cache
            except Exception:  # noqa: BLE001
                logger.exception("Failed to parse dummy DB; using built-in defaults")

        self._dummy_db_cache = self._default_dummy_db()
        return self._dummy_db_cache

    def _persist_dummy_db(self) -> None:
        if self._dummy_db_cache is None:
            return
        path = self._dummy_db_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._dummy_db_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to persist dummy DB updates", path=str(path))

    def _persist_booking_to_dummy_db(self, payload: dict[str, Any]) -> None:
        db = self._load_dummy_db()
        bookings = db.setdefault("bookings", {})
        booking_ref = str(payload.get("booking_ref", "")).upper()
        if booking_ref:
            bookings[booking_ref] = payload
        short_booking_id = str(payload.get("short_booking_id", "")).strip()
        if short_booking_id:
            bookings[short_booking_id] = payload
        self._dummy_db_cache = db
        self._persist_dummy_db()

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
            logger.exception("Backend GET failed", url=url)
            raise

    @retry(
        reraise=True,
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_circuit()
        try:
            response = await self.client.post(url, headers=self._headers, json=payload)
            response.raise_for_status()
            self._record_success()
            return response.json()
        except HTTPError:
            self._record_failure()
            logger.exception("Backend POST failed", url=url)
            raise

    def _fallback_agent_config(self, agent_id: str | None = None) -> AgentConfig:
        db = self._load_dummy_db()
        agents = db.get("agents", {}) if isinstance(db, dict) else {}
        key = agent_id or "default"
        raw = agents.get(key) or agents.get("default") or self._default_dummy_db()["agents"]["default"]
        payload = self._normalize_agent_config(raw)
        return AgentConfig(**payload)

    def _normalize_agent_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        service_catalog = payload.get("service_catalog") or payload.get("services") or []
        if service_catalog and isinstance(service_catalog, dict):
            service_catalog = [service_catalog]

        intake_questions = payload.get("intake_questions") or payload.get("intakeQuestions") or []
        language_settings = payload.get("language_settings") or {}
        transfer_settings = payload.get("live_transfer") or {}

        return {
            "agent_id": payload.get("id") or payload.get("agent_id") or "default",
            "business_name": payload.get("business_name") or "Local Home Services Bahrain",
            "greeting": payload.get("greeting") or payload.get("greeting_message") or "Hello, how can I help you today?",
            "intake_questions": intake_questions,
            "language": payload.get("language") or language_settings.get("default_language") or "en",
            "multilingual_enabled": payload.get("multilingual_enabled", language_settings.get("enabled", True)),
            "supported_languages": payload.get("supported_languages") or language_settings.get("supported_languages") or ["en", "ar"],
            "default_greeting_language": payload.get("default_greeting_language")
            or language_settings.get("default_greeting_language")
            or payload.get("language")
            or "en",
            "language_voice_map": payload.get("language_voice_map") or language_settings.get("language_voice_map") or {},
            "fallback_phone": payload.get("fallback_phone") or transfer_settings.get("transfer_number"),
            "max_call_duration_minutes": payload.get("max_call_duration_minutes") or transfer_settings.get("max_duration_minutes") or 5,
            "coverage_country": payload.get("coverage_country") or "Bahrain",
            "coverage_areas": payload.get("coverage_areas") or [],
            "excluded_areas": payload.get("excluded_areas") or [],
            "service_catalog": service_catalog,
            "booking_required_fields": payload.get("booking_required_fields") or ["service_type", "location", "preferred_time"],
            "faqs": payload.get("faqs") or {},
            "business_description": payload.get("business_description") or payload.get("description") or "",
            "business_hours": payload.get("business_hours") or "",
            "cancellation_policy": payload.get("cancellation_policy") or "",
            "payment_policy": payload.get("payment_policy") or "",
            "deposit_policy": payload.get("deposit_policy") or "",
            "additional_information": payload.get("additional_information") or [],
            "notes": payload.get("notes") or [],
        }

    def _agent_to_ui_context(self, agent: AgentConfig) -> dict[str, Any]:
        return {
            "agent_id": agent.agent_id,
            "business_name": agent.business_name,
            "greeting": agent.greeting,
            "language": agent.language,
            "multilingual_enabled": agent.multilingual_enabled,
            "supported_languages": agent.supported_languages,
            "default_greeting_language": agent.default_greeting_language,
            "fallback_phone": agent.fallback_phone,
            "max_call_duration_minutes": agent.max_call_duration_minutes,
            "coverage_country": agent.coverage_country,
            "coverage_areas": list(agent.coverage_areas),
            "booking_required_fields": list(agent.booking_required_fields),
            "services": [service.model_dump() for service in agent.service_catalog],
            "faqs": dict(agent.faqs),
            "business_description": agent.business_description,
            "business_hours": agent.business_hours,
            "cancellation_policy": agent.cancellation_policy,
            "payment_policy": agent.payment_policy,
            "deposit_policy": agent.deposit_policy,
            "additional_information": list(agent.additional_information),
            "intake_questions": [question.model_dump() for question in agent.intake_questions],
            "notes": list(agent.notes),
        }

    async def fetch_agent_config(self, agent_id: str | None = None) -> AgentConfig:
        if self._local_fallback_enabled():
            return self._fallback_agent_config(agent_id)

        url = f"{self.base_url}/agents/{agent_id or 'default'}/voice-config"
        try:
            payload = await self._get(url)
            return AgentConfig(**self._normalize_agent_config(payload))
        except Exception:  # noqa: BLE001
            logger.warning("Falling back to local dummy agent config", agent_id=agent_id)
            return self._fallback_agent_config(agent_id)

    async def fetch_intake_questions(self, agent_id: str | None = None) -> list[str]:
        config = await self.fetch_agent_config(agent_id)
        return [item.question for item in config.intake_questions]

    async def fetch_agent_ui_context(self, agent_id: str | None = None) -> dict[str, Any]:
        if self._local_fallback_enabled():
            agent = self._fallback_agent_config(agent_id)
            return self._agent_to_ui_context(agent)

        url = f"{self.base_url}/agents/{agent_id or 'default'}/ui-context"
        try:
            payload = await self._get(url)
            if isinstance(payload, dict):
                return payload
        except Exception:  # noqa: BLE001
            logger.warning("Falling back to local UI context", agent_id=agent_id)

        agent = self._fallback_agent_config(agent_id)
        return self._agent_to_ui_context(agent)

    async def resolve_agent_id_for_inbound(self, to_number: str | None) -> str | None:
        if not to_number:
            return None

        if self._local_fallback_enabled():
            db = self._load_dummy_db()
            routing = db.get("phone_routing", {}) if isinstance(db, dict) else {}
            mapped = routing.get(to_number)
            return str(mapped) if mapped else None

        url = f"{self.base_url}/telephony/resolve-agent"
        try:
            payload = await self._post(url, {"to_number": to_number})
            value = payload.get("agent_id") if isinstance(payload, dict) else None
            return str(value) if value else None
        except Exception:  # noqa: BLE001
            logger.warning("Agent resolution failed; using default mapping")
            db = self._load_dummy_db()
            routing = db.get("phone_routing", {}) if isinstance(db, dict) else {}
            mapped = routing.get(to_number)
            return str(mapped) if mapped else None

    async def answer_business_query(self, query: str, agent_id: str | None = None) -> dict[str, Any]:
        agent = await self.fetch_agent_config(agent_id)
        text = query.lower()

        service_lines = [
            f"{service.name} ({service.base_price_bhd})"
            for service in agent.service_catalog[:6]
        ]

        snippets: list[str] = []
        if any(token in text for token in ["service", "services", "provide", "offer", "business", "company"]):
            snippets.append(
                f"{agent.business_name} offers: {', '.join(service_lines)}."
                if service_lines
                else f"{agent.business_name} can share service options based on your needs."
            )

        if any(token in text for token in ["price", "pricing", "cost", "rate", "quote"]):
            snippets.append("Pricing depends on service type, location, urgency, and required materials.")

        if any(token in text for token in ["coverage", "area", "location", "where"]):
            if agent.coverage_areas:
                snippets.append(
                    f"Coverage is currently {agent.coverage_country}, including {', '.join(agent.coverage_areas[:8])}."
                )
            else:
                snippets.append(f"Coverage is currently {agent.coverage_country}.")

        if any(token in text for token in ["payment", "pay", "deposit"]):
            if agent.payment_policy:
                snippets.append(agent.payment_policy)
            if agent.deposit_policy:
                snippets.append(agent.deposit_policy)

        for key, value in agent.faqs.items():
            if key.lower() in text and value:
                snippets.append(value)

        if not snippets:
            snippets.append(
                f"{agent.business_name} can help with bookings, service details, pricing guidance, and booking status updates."
            )

        return {
            "answer": " ".join(snippets[:4]).strip(),
            "suggested_services": [service.name for service in agent.service_catalog[:5]],
        }

    def _find_service_in_catalog(self, answer: str, agent: AgentConfig) -> dict[str, Any] | None:
        lowered = (answer or "").lower()
        for service in agent.service_catalog:
            tokens = [service.name, service.service_id, *service.keywords]
            if any(token.lower() in lowered for token in tokens if token):
                return service.model_dump()
        return None

    async def book_service(self, agent_id: str, answers: dict[str, str]) -> dict[str, Any]:
        if self._local_fallback_enabled():
            return self._book_local(agent_id, answers)

        url = f"{self.base_url}/agents/{agent_id}/bookings"
        try:
            return await self._post(url, {"answers": answers})
        except Exception:  # noqa: BLE001
            logger.warning("Remote booking failed; using local booking fallback", agent_id=agent_id)
            return self._book_local(agent_id, answers)

    def _book_local(self, agent_id: str, answers: dict[str, str]) -> dict[str, Any]:
        agent = self._fallback_agent_config(agent_id)
        service_answer = (
            answers.get("service_type")
            or answers.get("service")
            or answers.get("service_name")
            or answers.get("q0")
            or ""
        )
        location_answer = answers.get("location") or answers.get("q1") or ""
        time_answer = answers.get("preferred_time") or answers.get("q2") or ""

        matched = self._find_service_in_catalog(service_answer, agent)
        booking_ref = f"DUMMY-{int(time.time())}"

        if not matched:
            return {
                "status": "unsupported_service",
                "booking_ref": booking_ref,
                "message": (
                    "I can currently book AC repair, deep cleaning, at-home salon, or general maintenance. "
                    "Please tell me which one you need."
                ),
                "answers": answers,
            }

        short_booking_id = re.sub(r"\D", "", booking_ref)[-5:] or re.sub(r"\D", "", booking_ref)

        payload = {
            "status": "mock_confirmed",
            "booking_ref": booking_ref,
            "short_booking_id": short_booking_id,
            "service_name": matched.get("name"),
            "service_id": matched.get("service_id"),
            "location": location_answer,
            "preferred_time": time_answer,
            "price_range": matched.get("base_price_bhd"),
            "message": (
                f"Booking {booking_ref} is created for {matched.get('name')} in {location_answer} at {time_answer}. "
                f"Short ID is {short_booking_id}. Our operations team will confirm final timing and quote shortly."
            ),
            "answers": answers,
        }
        self._runtime_bookings[booking_ref] = payload
        if short_booking_id:
            self._runtime_bookings[short_booking_id] = payload
        self._persist_booking_to_dummy_db(payload)
        return payload

    async def track_booking(self, booking_id: str, agent_id: str | None = None) -> dict[str, Any]:
        normalized = booking_id.strip().upper()
        if not normalized:
            raise BackendCommunicationError("booking_id is required")

        if self._local_fallback_enabled():
            return self._track_local(normalized)

        url = f"{self.base_url}/agents/{agent_id or 'default'}/bookings/{normalized}"
        try:
            payload = await self._get(url)
            if isinstance(payload, dict):
                return payload
        except Exception:  # noqa: BLE001
            logger.warning("Remote track booking failed; using local fallback", booking_id=normalized)

        return self._track_local(normalized)

    def _track_local(self, booking_id: str) -> dict[str, Any]:
        if booking_id in self._runtime_bookings:
            return self._runtime_bookings[booking_id]

        db = self._load_dummy_db()
        bookings = db.get("bookings", {}) if isinstance(db, dict) else {}
        payload = bookings.get(booking_id)
        if payload:
            return payload

        # Try matching by numeric-only variant (for spoken booking IDs like "12614").
        requested_digits = re.sub(r"\D", "", booking_id)
        if requested_digits:
            for key, value in self._runtime_bookings.items():
                candidate_digits = re.sub(r"\D", "", key)
                if candidate_digits.endswith(requested_digits) or requested_digits.endswith(candidate_digits):
                    return value
            for key, value in bookings.items():
                candidate_digits = re.sub(r"\D", "", key)
                if candidate_digits and (candidate_digits.endswith(requested_digits) or requested_digits.endswith(candidate_digits)):
                    return value

        return {
            "status": "not_found",
            "booking_ref": booking_id,
            "message": "Booking ID was not found in current records.",
        }
