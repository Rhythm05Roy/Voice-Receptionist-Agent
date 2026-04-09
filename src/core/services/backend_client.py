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
from src.core.types import AgentConfig, ServiceInfo


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, TimeoutException):
        return True
    if isinstance(exc, HTTPStatusError) and exc.response is not None:
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class BackendClient:
    _runtime_phone_routing: dict[str, str] = {}
    _runtime_agent_numbers: dict[str, dict[str, str]] = {}
    _runtime_forwarding_numbers: dict[str, str] = {}

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
        api_key = (self.api_key or "").strip()
        if not api_key or api_key in {"change-me", "dev-backend-key"}:
            return {}
        return {"Authorization": f"Bearer {api_key}"}

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

    def _is_business_catalog_endpoint(self) -> bool:
        return "/api/business/businesses" in self.base_url.lower()

    def _business_catalog_url(self) -> str:
        return self.base_url if self.base_url.endswith("/") else f"{self.base_url}/"

    def _dummy_db_path(self) -> Path:
        configured = os.getenv("DUMMY_DB_PATH", "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        project_root = Path(__file__).resolve().parents[3]
        return project_root / "tests" / "dummy_business_db.json"

    def _persist_dummy_db(self) -> None:
        if not self._dummy_db_cache:
            return
        try:
            db_path = self._dummy_db_path()
            db_path.write_text(json.dumps(self._dummy_db_cache, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            logger.warning("Failed to persist dummy business database")

    def _apply_runtime_overrides(self, agent: AgentConfig) -> AgentConfig:
        forwarding_number = self._runtime_forwarding_numbers.get(agent.agent_id)
        if forwarding_number:
            agent = agent.model_copy(update={"fallback_phone": forwarding_number})
        assigned_number = self._runtime_agent_numbers.get(agent.agent_id, {}).get("phone_number")
        if assigned_number:
            notes = list(agent.notes)
            binding_note = f"Assigned number: {assigned_number}"
            if binding_note not in notes:
                notes.append(binding_note)
            agent = agent.model_copy(update={"notes": notes})
        return agent

    def bind_phone_number(
        self,
        *,
        agent_id: str,
        phone_number: str,
        phone_number_sid: str = "",
        friendly_name: str = "",
    ) -> dict[str, str]:
        agent_key = str(agent_id)
        self._runtime_phone_routing[phone_number] = agent_key
        self._runtime_agent_numbers[agent_key] = {
            "phone_number": phone_number,
            "phone_number_sid": phone_number_sid,
            "friendly_name": friendly_name,
        }

        db = self._load_dummy_db()
        if isinstance(db, dict):
            routing = db.setdefault("phone_routing", {})
            if isinstance(routing, dict):
                routing[phone_number] = agent_key
            self._persist_dummy_db()

        return self._runtime_agent_numbers[agent_key]

    def set_call_forwarding(self, *, agent_id: str, forwarding_number: str) -> dict[str, str]:
        agent_key = str(agent_id)
        self._runtime_forwarding_numbers[agent_key] = forwarding_number

        db = self._load_dummy_db()
        if isinstance(db, dict):
            agents = db.setdefault("agents", {})
            if isinstance(agents, dict):
                raw = agents.get(agent_key)
                if isinstance(raw, dict):
                    raw["fallback_phone"] = forwarding_number
            self._persist_dummy_db()

        return {"agent_id": agent_key, "forwarding_number": forwarding_number}

    def get_phone_assignment(self, agent_id: str) -> dict[str, str] | None:
        assignment = self._runtime_agent_numbers.get(str(agent_id))
        if not assignment:
            return None
        payload = dict(assignment)
        forwarding_number = self._runtime_forwarding_numbers.get(str(agent_id))
        if forwarding_number:
            payload["forwarding_number"] = forwarding_number
        return payload

    def rebind_phone_number(
        self,
        *,
        agent_id: str,
        phone_number: str,
        phone_number_sid: str = "",
        friendly_name: str = "",
    ) -> dict[str, str]:
        agent_key = str(agent_id)
        previous_agent_id = self._runtime_phone_routing.get(phone_number)
        if previous_agent_id and previous_agent_id != agent_key:
            self._runtime_agent_numbers.pop(previous_agent_id, None)
        return self.bind_phone_number(
            agent_id=agent_key,
            phone_number=phone_number,
            phone_number_sid=phone_number_sid,
            friendly_name=friendly_name,
        )

    def release_phone_number(
        self,
        *,
        agent_id: str | None = None,
        phone_number: str | None = None,
        phone_number_sid: str | None = None,
    ) -> dict[str, str] | None:
        target_agent_id = str(agent_id) if agent_id else None
        assignment: dict[str, str] | None = None

        if target_agent_id:
            assignment = self._runtime_agent_numbers.pop(target_agent_id, None)
        else:
            for current_agent_id, current_assignment in list(self._runtime_agent_numbers.items()):
                if (
                    (phone_number and current_assignment.get("phone_number") == phone_number)
                    or (phone_number_sid and current_assignment.get("phone_number_sid") == phone_number_sid)
                ):
                    target_agent_id = current_agent_id
                    assignment = self._runtime_agent_numbers.pop(current_agent_id, None)
                    break

        if not assignment or not target_agent_id:
            return None

        assigned_number = assignment.get("phone_number")
        if assigned_number:
            self._runtime_phone_routing.pop(assigned_number, None)

        db = self._load_dummy_db()
        if isinstance(db, dict):
            routing = db.get("phone_routing", {})
            if isinstance(routing, dict) and assigned_number:
                routing.pop(assigned_number, None)
            self._persist_dummy_db()

        payload = dict(assignment)
        forwarding_number = self._runtime_forwarding_numbers.get(target_agent_id)
        if forwarding_number:
            payload["forwarding_number"] = forwarding_number
        payload["agent_id"] = target_agent_id
        return payload

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
    async def _get(self, url: str) -> Any:
        self._ensure_circuit()
        try:
            response = await self.client.get(url, headers=self._headers, follow_redirects=True)
            response.raise_for_status()
            self._record_success()
            return response.json()
        except HTTPError:
            self._record_failure()
            logger.exception("Backend GET failed", url=url)
            raise

    @staticmethod
    def _slug(value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
        return cleaned.strip("_") or "service"

    @staticmethod
    def _day_name(day_index: int) -> str:
        mapping = {
            0: "Mon",
            1: "Tue",
            2: "Wed",
            3: "Thu",
            4: "Fri",
            5: "Sat",
            6: "Sun",
        }
        return mapping.get(day_index, f"Day {day_index}")

    @staticmethod
    def _compact_time(value: str) -> str:
        if not value:
            return ""
        parts = value.split(":")
        if len(parts) >= 2:
            return f"{parts[0]}:{parts[1]}"
        return value

    def _default_questions_for_business(self, business_type: str) -> list[dict[str, Any]]:
        is_salon = "salon" in business_type.lower() or "spa" in business_type.lower()
        questions: list[dict[str, Any]] = [
            {
                "key": "service_type",
                "question": "Which service would you like to book today?",
                "answer_type": "text",
                "required": True,
            },
            {
                "key": "location",
                "question": "What area are you located in?",
                "answer_type": "text",
                "required": True,
            },
            {
                "key": "preferred_time",
                "question": "What time works best for your appointment?",
                "answer_type": "text",
                "required": True,
            },
        ]
        if is_salon:
            questions.append(
                {
                    "key": "stylist_preference",
                    "question": "Do you have any stylist or treatment preference?",
                    "answer_type": "text",
                    "required": False,
                }
            )
        return questions

    def _normalize_business_services(self, services: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for service in services:
            name = str(service.get("name") or "Service").strip()
            category = str(service.get("category_name") or "").strip()
            price = str(service.get("price") or "").strip()
            duration = service.get("duration")
            allow_booking = bool(service.get("allow_booking", True))
            description = str(service.get("description") or "").strip()
            price_note = []
            if duration:
                price_note.append(f"Duration: {duration} minutes")
            if not allow_booking:
                price_note.append("Currently inquiry only")
            keywords = [self._slug(name).replace("_", " ")]
            if category:
                keywords.append(category)
            normalized.append(
                {
                    "service_id": str(service.get("id") or self._slug(name)),
                    "name": name,
                    "description": description or f"{name} service",
                    "base_price": price,
                    "base_price_bhd": price,
                    "currency": "",
                    "price_note": ". ".join(price_note),
                    "keywords": keywords,
                }
            )
        return normalized

    def _normalize_business_faqs(self, faqs: list[dict[str, Any]]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for item in faqs:
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
            if question and answer:
                normalized[question] = answer
        return normalized

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        stopwords = {
            "what", "which", "when", "where", "who", "does", "do", "did", "your", "you",
            "have", "with", "about", "need", "please", "tell", "know", "kind", "actually",
            "can", "could", "would", "the", "and", "for", "are", "is", "any", "more",
            "detail", "details", "general", "policy", "policies", "service", "services",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z]{3,}", text.lower())
            if token not in stopwords
        }

    @staticmethod
    def _sentences(value: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\r?\n+", value or "") if part.strip()]

    def _payment_context_snippets(self, agent: AgentConfig, text: str) -> list[str]:
        snippets: list[str] = []
        payment_keywords = {"payment", "payments", "pay", "card", "cash", "digital", "gateway", "refund", "split", "bill"}

        if agent.payment_policy:
            snippets.append(agent.payment_policy)
        for source in [agent.deposit_policy, agent.cancellation_policy]:
            for sentence in self._sentences(source):
                lowered = sentence.lower()
                if any(keyword in lowered for keyword in payment_keywords):
                    snippets.append(sentence)

        for question, answer in agent.faqs.items():
            q_text = f"{question} {answer}".lower()
            if any(keyword in q_text for keyword in payment_keywords):
                snippets.append(answer)

        unique: list[str] = []
        seen: set[str] = set()
        for item in snippets:
            if item and item not in seen:
                unique.append(item)
                seen.add(item)
        return unique[:3]

    def _relevant_faq_answers(self, agent: AgentConfig, query: str) -> list[str]:
        query_terms = self._query_terms(query)
        if not query_terms:
            return []

        matches: list[tuple[int, str]] = []
        for question, answer in agent.faqs.items():
            faq_terms = self._query_terms(question)
            overlap = query_terms.intersection(faq_terms)
            if len(overlap) >= 2:
                matches.append((len(overlap), answer))
                continue
            combined = f"{question} {answer}".lower()
            if any(term in combined for term in query_terms if len(term) >= 5):
                matches.append((1, answer))

        matches.sort(key=lambda item: item[0], reverse=True)
        unique_answers: list[str] = []
        seen: set[str] = set()
        for _, answer in matches:
            if answer not in seen:
                unique_answers.append(answer)
                seen.add(answer)
        return unique_answers[:2]

    def _normalize_business_hours(self, hours: list[dict[str, Any]]) -> str:
        segments: list[str] = []
        for item in sorted(hours, key=lambda row: int(row.get("day", 99))):
            day_name = self._day_name(int(item.get("day", 99)))
            if item.get("is_closed"):
                segments.append(f"{day_name}: Closed")
                continue
            open_time = self._compact_time(str(item.get("open_time") or ""))
            close_time = self._compact_time(str(item.get("close_time") or ""))
            if open_time and close_time:
                segments.append(f"{day_name}: {open_time}-{close_time}")
        return "; ".join(segments)

    def _policy_content(self, policies: list[dict[str, Any]], policy_type: str) -> str:
        for item in policies:
            if str(item.get("policy_type") or "").strip().lower() == policy_type:
                return str(item.get("content") or "").strip()
        return ""

    def _normalize_business_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        business_name = str(payload.get("name") or payload.get("business_name") or "Business").strip()
        business_type = str(payload.get("business_type") or "").strip()
        description = str(payload.get("description") or "").strip()
        address = str(payload.get("address") or "").strip()
        timezone = str(payload.get("timezone") or "").strip()
        services = self._normalize_business_services(payload.get("services") or [])
        faqs = self._normalize_business_faqs(payload.get("faqs") or [])
        additional_info = [
            str(item.get("content") or "").strip()
            for item in payload.get("additional_info") or []
            if str(item.get("content") or "").strip()
        ]
        if address:
            additional_info.insert(0, f"Business address: {address}")
        if timezone:
            additional_info.append(f"Business timezone: {timezone}")
        if payload.get("business_website"):
            additional_info.append(f"Website: {payload['business_website']}")
        if payload.get("business_email"):
            additional_info.append(f"Contact email: {payload['business_email']}")

        greeting = (
            f"Hello, thank you for calling {business_name}. "
            "How can I help you today?"
        )
        return {
            "agent_id": str(payload.get("id") or payload.get("agent_id") or "default"),
            "business_name": business_name,
            "greeting": greeting,
            "intake_questions": self._default_questions_for_business(business_type),
            "language": "en",
            "multilingual_enabled": True,
            "supported_languages": ["en", "ar", "hi", "fr", "es"],
            "default_greeting_language": "en",
            "language_voice_map": {},
            "fallback_phone": None,
            "max_call_duration_minutes": 15,
            "coverage_country": "",
            "coverage_areas": [address] if address else [],
            "excluded_areas": [],
            "service_catalog": services,
            "booking_required_fields": ["service_type", "location", "preferred_time"],
            "faqs": faqs,
            "business_description": description or business_type,
            "business_hours": self._normalize_business_hours(payload.get("hours") or []),
            "cancellation_policy": self._policy_content(payload.get("policies") or [], "cancellation"),
            "payment_policy": self._policy_content(payload.get("policies") or [], "payment"),
            "deposit_policy": self._policy_content(payload.get("policies") or [], "deposit"),
            "additional_information": additional_info,
            "notes": [f"Business type: {business_type}"] if business_type else [],
        }

    def _select_business_payload(self, payload: Any, agent_id: str | None) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if not isinstance(payload, list):
            raise BackendCommunicationError("Unsupported business payload shape")

        if not payload:
            raise BackendCommunicationError("No businesses returned by backend")

        if agent_id is not None:
            for item in payload:
                if str(item.get("id")) == str(agent_id):
                    return item

        for item in payload:
            if bool(item.get("is_active", True)):
                return item

        return payload[0]

    async def _fetch_business_catalog_record(self, agent_id: str | None = None) -> dict[str, Any]:
        payload = await self._get(self._business_catalog_url())
        return self._select_business_payload(payload, agent_id)

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
            response = await self.client.post(url, headers=self._headers, json=payload, follow_redirects=True)
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
        return self._apply_runtime_overrides(AgentConfig(**payload))

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

        if self._is_business_catalog_endpoint():
            try:
                payload = await self._fetch_business_catalog_record(agent_id)
                return self._apply_runtime_overrides(AgentConfig(**self._normalize_business_record(payload)))
            except Exception:  # noqa: BLE001
                logger.warning("Falling back to local dummy agent config", agent_id=agent_id)
                return self._fallback_agent_config(agent_id)

        url = f"{self.base_url}/agents/{agent_id or 'default'}/voice-config"
        try:
            payload = await self._get(url)
            normalized = self._normalize_agent_config(payload)
            intake_questions = await self.fetch_intake_question_configs(str(normalized.get("agent_id") or agent_id or "default"))
            if intake_questions:
                normalized["intake_questions"] = intake_questions
            return self._apply_runtime_overrides(AgentConfig(**normalized))
        except Exception:  # noqa: BLE001
            logger.warning("Falling back to local dummy agent config", agent_id=agent_id)
            return self._fallback_agent_config(agent_id)

    async def fetch_intake_questions(self, agent_id: str | None = None) -> list[str]:
        config = await self.fetch_agent_config(agent_id)
        return [item.question for item in config.intake_questions]

    async def fetch_intake_question_configs(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        if not agent_id or self._local_fallback_enabled() or self._is_business_catalog_endpoint():
            config = await self.fetch_agent_config(agent_id)
            return [item.model_dump() for item in config.intake_questions]

        url = f"{self.base_url}/agents/{agent_id}/intake-questions"
        try:
            payload = await self._get(url)
            questions = payload.get("questions") if isinstance(payload, dict) else payload
            if isinstance(questions, list):
                return [dict(item) if isinstance(item, dict) else {"question": str(item)} for item in questions]
        except Exception:  # noqa: BLE001
            logger.warning("Failed to fetch remote intake questions; falling back to agent config", agent_id=agent_id)
        return [item.model_dump() for item in self._fallback_agent_config(agent_id).intake_questions]

    async def fetch_agent_ui_context(self, agent_id: str | None = None) -> dict[str, Any]:
        if self._local_fallback_enabled():
            agent = self._fallback_agent_config(agent_id)
            return self._agent_to_ui_context(agent)

        if self._is_business_catalog_endpoint():
            try:
                payload = await self._fetch_business_catalog_record(agent_id)
                agent = self._apply_runtime_overrides(AgentConfig(**self._normalize_business_record(payload)))
                return self._agent_to_ui_context(agent)
            except Exception:  # noqa: BLE001
                logger.warning("Falling back to local UI context", agent_id=agent_id)
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

        runtime_match = self._runtime_phone_routing.get(to_number)
        if runtime_match:
            return runtime_match

        if self._local_fallback_enabled():
            db = self._load_dummy_db()
            routing = db.get("phone_routing", {}) if isinstance(db, dict) else {}
            mapped = routing.get(to_number)
            return str(mapped) if mapped else None

        if self._is_business_catalog_endpoint():
            try:
                payload = await self._fetch_business_catalog_record()
                value = payload.get("id")
                return str(value) if value is not None else None
            except Exception:  # noqa: BLE001
                logger.warning("Business catalog agent resolution failed; using default mapping")
                return "default"

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
        return self.build_business_query_answer(agent, query)

    def _find_matching_services(self, query: str, agent: AgentConfig) -> list[ServiceInfo]:
        text = query.lower()
        matches: list[ServiceInfo] = []
        seen_ids: set[str] = set()

        for service in agent.service_catalog:
            candidates = [service.name, service.service_id, *service.keywords]
            if any(candidate and candidate.lower() in text for candidate in candidates):
                if service.service_id not in seen_ids:
                    matches.append(service)
                    seen_ids.add(service.service_id)
                    continue

            query_tokens = set(re.findall(r"[a-zA-Z]{3,}", text))
            service_tokens = {
                token
                for token in re.findall(r"[a-zA-Z]{3,}", " ".join([service.name, service.service_id, *service.keywords]).lower())
            }
            if query_tokens and service_tokens and query_tokens.intersection(service_tokens):
                if service.service_id not in seen_ids:
                    matches.append(service)
                    seen_ids.add(service.service_id)

        return matches

    def build_business_query_answer(self, agent: AgentConfig, query: str) -> dict[str, Any]:
        text = query.lower()
        matched_services = self._find_matching_services(query, agent)

        service_lines: list[str] = []
        services_to_describe = matched_services or agent.service_catalog[:6]
        for service in services_to_describe[:6]:
            price = service.base_price or service.base_price_bhd or "pricing varies"
            description = service.description.strip() or service.name
            line = f"{service.name}: {description} ({price})"
            if service.price_note:
                line += f". {service.price_note}"
            service_lines.append(line)

        snippets: list[str] = []
        service_query_tokens = ["service", "services", "provide", "offer", "detail", "details", "about", "explain"]

        if any(token in text for token in [*service_query_tokens, "business", "company"]):
            if matched_services and service_lines:
                snippets.append(f"For {matched_services[0].name}, {service_lines[0]}")
            elif service_lines:
                featured = [service.name for service in agent.service_catalog[:4]]
                remainder = max(0, len(agent.service_catalog) - len(featured))
                summary = f"{agent.business_name} mainly offers {', '.join(featured)}"
                if remainder:
                    summary += f", and {remainder} more options"
                snippets.append(summary + ".")
            elif agent.business_description:
                snippets.append(agent.business_description)

        if any(token in text for token in ["price", "pricing", "cost", "rate", "quote", "fee"]):
            if service_lines:
                if matched_services:
                    snippets.append(f"Pricing for {matched_services[0].name} starts from {matched_services[0].base_price or matched_services[0].base_price_bhd or 'pricing varies'}.")
                else:
                    snippets.append("Current pricing includes " + "; ".join(service_lines[:3]) + ".")
            else:
                snippets.append("Pricing depends on the selected service and appointment details.")

        if any(token in text for token in ["hour", "open", "close", "timing", "time"]):
            if agent.business_hours:
                snippets.append(f"Our business hours are {agent.business_hours}.")
            else:
                snippets.append("Business hours vary by day, and our team can confirm the latest schedule for you.")

        if any(token in text for token in ["payment", "pay", "deposit", "card", "cash"]):
            payment_snippets = self._payment_context_snippets(agent, text)
            if payment_snippets:
                snippets.extend(payment_snippets)
            else:
                snippets.append("Payment details depend on the selected service, and we can confirm the exact policy for you.")

        if any(token in text for token in ["cancel", "cancellation", "refund", "reschedule"]):
            if agent.cancellation_policy:
                snippets.append(agent.cancellation_policy)
            else:
                snippets.append("Cancellation terms depend on the appointment type, and we can confirm the latest policy for you.")

        if "polic" in text and not any(token in text for token in ["payment", "pay", "deposit", "cancel", "cancellation", "refund"]):
            policy_bits = [agent.cancellation_policy, agent.payment_policy, agent.deposit_policy]
            snippets.extend([bit for bit in policy_bits if bit][:3])

        if any(token in text for token in ["where", "address", "location", "located"]):
            location_added = False
            if agent.coverage_areas:
                snippets.append(f"We are located at or serve {', '.join(agent.coverage_areas[:5])}.")
                location_added = True
            else:
                for info in agent.additional_information:
                    if "address:" in info.lower():
                        snippets.append(info)
                        location_added = True
                        break
            if not location_added:
                snippets.append("We can confirm the exact service area and address details for you.")

        snippets.extend(self._relevant_faq_answers(agent, query))

        if not snippets:
            if agent.business_description:
                snippets.append(agent.business_description)
            elif service_lines:
                snippets.append(f"{agent.business_name} offers {', '.join(service_lines)}.")
            else:
                snippets.append(
                    f"{agent.business_name} can help with bookings, service details, pricing guidance, and booking status updates."
                )

        return {
            "answer": " ".join(snippets[:3]).strip(),
            "suggested_services": [service.name for service in (matched_services or agent.service_catalog[:5])],
            "matched_services": [service.name for service in matched_services],
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
        date_answer = answers.get("preferred_date") or ""

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

        # Build comprehensive booking record
        payload: dict[str, Any] = {
            "status": "confirmed",
            "booking_ref": booking_ref,
            "short_booking_id": short_booking_id,
            "agent_id": agent_id,
            "service_name": matched.get("name"),
            "service_id": matched.get("service_id"),
            "location": location_answer,
            "preferred_date": date_answer,
            "preferred_time": time_answer,
            "price_range": matched.get("base_price_bhd"),
            "created_at": time.time(),
        }

        # Customer info
        if answers.get("customer_name"):
            payload["customer_name"] = answers["customer_name"]
        if answers.get("customer_phone"):
            payload["customer_phone"] = answers["customer_phone"]

        # Service-specific details
        if answers.get("property_type"):
            payload["property_type"] = answers["property_type"]
        if answers.get("num_rooms"):
            payload["num_rooms"] = answers["num_rooms"]
        if answers.get("specific_areas"):
            payload["specific_areas"] = answers["specific_areas"]
        if answers.get("allergy_info"):
            payload["allergy_info"] = answers["allergy_info"]
        if answers.get("issue_description"):
            payload["issue_description"] = answers["issue_description"]
        if answers.get("urgency"):
            payload["urgency"] = answers["urgency"]
        if answers.get("special_instructions"):
            payload["special_instructions"] = answers["special_instructions"]

        # Build confirmation message
        date_str = f" on {date_answer}" if date_answer else ""
        time_str = f" at {time_answer}" if time_answer else ""
        location_str = f" in {location_answer}" if location_answer else ""

        payload["message"] = (
            f"Booking {booking_ref} is confirmed for {matched.get('name')}"
            f"{location_str}{date_str}{time_str}. "
            f"Your booking reference is {short_booking_id}. "
            f"Our operations team will confirm final timing and quote shortly."
        )
        payload["answers"] = answers

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
