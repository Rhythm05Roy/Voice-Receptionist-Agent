from __future__ import annotations

import re
import time
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.api.exceptions import ConversationEngineError
from src.core.services.backend_client import BackendClient
from src.core.services.elevenlabs import ElevenLabsClient
from src.core.services.openai import OpenAIClient
from src.core.types import AgentConfig

from . import prompts
from .handlers import handle_booking, handle_intake

State = Literal["greeting", "intent", "intake", "booking", "assist", "transfer", "end"]
TaskIntent = Literal["undecided", "new_booking", "track_booking", "business_info"]


class CallSession(BaseModel):
    call_id: str
    agent_config: AgentConfig
    current_state: State = "greeting"
    task_intent: TaskIntent = "undecided"
    collected_answers: dict[str, str] = Field(default_factory=dict)
    current_question_index: int = 0
    awaiting_answer: bool = False
    preferred_language: str | None = None
    selected_service: str | None = None
    last_booking_ref: str | None = None
    last_booking_payload: dict[str, str | None] = Field(default_factory=dict)
    last_intent: str | None = None
    intent_confidence: str = "unknown"
    post_booking_stage: str | None = None
    unknown_answer_retries: int = 0
    out_of_coverage: bool = False
    booking_completed: bool = False
    created_at: float = Field(default_factory=time.monotonic)
    last_activity: float = Field(default_factory=time.monotonic)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def touch(self) -> None:
        self.last_activity = time.monotonic()


class CallSessionManager:
    """In-memory session storage with TTL.

    Session key format: f"call:{call_uuid}".

    Redis migration sketch:
      - Replace _store dict with redis.asyncio.Redis.
      - On save: redis.setex(key, ttl, session.model_dump_json()).
      - On get: payload = redis.get(key); CallSession.model_validate_json(payload).
      - Use same key format for compatibility.
    """

    def __init__(self, ttl_seconds: int = 1800):
        self.ttl = ttl_seconds
        self._store: dict[str, CallSession] = {}

    @staticmethod
    def _key(call_id: str) -> str:
        return f"call:{call_id}"

    def get(self, call_id: str) -> CallSession | None:
        session = self._store.get(self._key(call_id))
        if not session:
            return None

        now = time.monotonic()
        if now - session.last_activity > self.ttl:
            self.delete(call_id)
            return None

        session.touch()
        return session

    def save(self, session: CallSession) -> None:
        session.touch()
        self._store[self._key(session.call_id)] = session

    def delete(self, call_id: str) -> None:
        self._store.pop(self._key(call_id), None)

    def prune(self) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, session in self._store.items()
            if now - session.last_activity > self.ttl
        ]
        for key in expired:
            self._store.pop(key, None)


class ConversationEngine:
    def __init__(
        self,
        backend_client: BackendClient,
        llm_client: OpenAIClient,
        tts_client: ElevenLabsClient,
        environment: str = "development",
        session_manager: CallSessionManager | None = None,
    ):
        self.backend_client = backend_client
        self.llm_client = llm_client
        self.tts_client = tts_client
        self.environment = environment
        self.sessions = session_manager or CallSessionManager()

    def has_session(self, call_id: str) -> bool:
        return self.sessions.get(call_id) is not None

    @staticmethod
    def _is_farewell(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        if not normalized:
            return False
        keywords = {
            "bye",
            "goodbye",
            "thanks bye",
            "thank you bye",
            "thats all",
            "that's all",
            "no thanks",
            "end call",
            "hang up",
        }
        return any(token in normalized for token in keywords)

    @staticmethod
    def _extract_booking_id(text: str) -> str | None:
        if not text:
            return None

        raw = text.strip()
        upper = raw.upper()

        # Preferred ID format (e.g., DUMMY-10001).
        match = re.search(r"\b([A-Z]{2,}-\d{3,})\b", upper)
        if match:
            return match.group(1)

        # Spoken/numeric form: "my booking id is 12614".
        numeric_context = re.search(r"(booking\s*(?:id|number|reference)[^0-9]*)([0-9][0-9,\s-]{2,})", upper)
        if numeric_context:
            digits = re.sub(r"\D", "", numeric_context.group(2))
            if len(digits) >= 3:
                return digits

        # Fallback: any standalone 4+ digit token.
        fallback = re.search(r"\b(\d{4,})\b", upper)
        if fallback:
            return fallback.group(1)

        return None

    @staticmethod
    def _looks_like_tracking_request(text: str) -> bool:
        lowered = (text or "").lower()
        cues = (
            "track",
            "booking status",
            "status of booking",
            "booking id",
            "reschedule",
            "cancel booking",
            "my booking",
            "previous booking",
        )
        return any(cue in lowered for cue in cues)

    @staticmethod
    def _looks_like_info_request(text: str) -> bool:
        lowered = (text or "").lower()
        cues = (
            "service",
            "services",
            "what do you provide",
            "what kind",
            "business",
            "company",
            "pricing",
            "price",
            "cost",
            "rate",
            "offer",
        )
        return any(cue in lowered for cue in cues)

    @staticmethod
    def _looks_like_new_booking_request(text: str) -> bool:
        lowered = (text or "").lower()
        if any(token in lowered for token in ("thank", "thanks", "shukran", "no need", "don't need", "dont need")):
            return False
        cues = (
            "book",
            "booking",
            "appointment",
            "need service",
            "i need",
            "visit",
        )
        return any(cue in lowered for cue in cues)

    @staticmethod
    def _looks_like_gratitude(text: str) -> bool:
        lowered = (text or "").lower()
        return any(token in lowered for token in ("thank you", "thanks", "shukran", "much appreciated"))

    @staticmethod
    def _looks_like_no_more_help(text: str) -> bool:
        lowered = (text or "").lower()
        cues = (
            "no more",
            "nothing else",
            "anything else",
            "that's all",
            "thats all",
            "i don't need any other",
            "i dont need any other",
            "no i don't need",
            "no i dont need",
            "no need",
            "all good",
        )
        return any(cue in lowered for cue in cues)

    @staticmethod
    def _looks_like_acknowledgement(text: str) -> bool:
        lowered = (text or "").lower().strip()
        compact = re.sub(r"[^a-z0-9\s]+", " ", lowered)
        compact = re.sub(r"\s+", " ", compact).strip()
        if compact in {"ok", "okay", "alright", "cool", "got it", "great", "sounds great"}:
            return True
        return "sounds great" in compact

    @staticmethod
    def _looks_like_booking_id_request(text: str) -> bool:
        lowered = (text or "").lower()
        cues = (
            "provide booking id",
            "give booking id",
            "my booking id",
            "booking id so",
            "booking reference",
            "reference number",
        )
        return any(cue in lowered for cue in cues)

    @staticmethod
    def _normalize_confidence(raw: str | None) -> str:
        value = (raw or "").strip().lower()
        if value in {"high", "medium", "low"}:
            return value
        return "unknown"

    def _disambiguation_prompt(self) -> str:
        return (
            "I can help in three ways: new booking, booking status by booking ID, or service and pricing questions. "
            "Which one do you want now?"
        )

    def _resolve_intent(
        self,
        text: str,
        llm_intent: str,
        llm_confidence: str,
    ) -> tuple[str, str | None, str]:
        booking_id = self._extract_booking_id(text)

        strong_tracking = bool(booking_id or self._looks_like_tracking_request(text))
        strong_info = self._looks_like_info_request(text) and not self._looks_like_new_booking_request(text)
        strong_new_booking = self._looks_like_new_booking_request(text)

        if strong_tracking:
            return "track_booking", booking_id, "high"
        if strong_info:
            return "business_info", booking_id, "high"
        if strong_new_booking:
            return "new_booking", booking_id, "high"

        resolved_intent = llm_intent
        confidence = self._normalize_confidence(llm_confidence)

        if resolved_intent == "track_booking" and not booking_id:
            resolved_intent = "unclear"
            confidence = "low"

        if confidence == "low" and resolved_intent in {"new_booking", "track_booking", "business_info", "pricing_info"}:
            resolved_intent = "unclear"

        return resolved_intent, booking_id, confidence

    async def start_session(self, call_id: str, agent_id: str | None = None) -> dict[str, str]:
        agent_config = await self.backend_client.fetch_agent_config(agent_id)
        preferred_language = agent_config.default_greeting_language or agent_config.language or "en"
        session = CallSession(
            call_id=call_id,
            agent_config=agent_config,
            preferred_language=preferred_language.lower(),
        )
        self.sessions.save(session)

        greeting = agent_config.greeting or prompts.GREETING_TEMPLATE
        audio_url = ""
        try:
            voice_id = agent_config.language_voice_map.get(session.preferred_language or "")
            audio_url = await self.tts_client.synthesize_text(greeting, voice_id=voice_id)
        except Exception:  # noqa: BLE001
            logger.warning("Greeting TTS failed, proceeding with text response", call_id=call_id)

        logger.bind(call_id=call_id).info("Conversation session started")
        return {"text": greeting, "audio_url": audio_url}

    async def process_user_input(
        self,
        call_id: str,
        transcribed_text: str,
        agent_id: str | None = None,
    ) -> dict[str, str | None]:
        session = await self._get_or_create_session(call_id=call_id, agent_id=agent_id)
        self.sessions.prune()
        start = time.perf_counter()
        previous_state = session.current_state

        text = (transcribed_text or "").strip()
        logger.bind(call_id=call_id).debug(
            "Processing user turn",
            state=session.current_state,
            task_intent=session.task_intent,
            has_text=bool(text),
        )
        response: dict[str, str | None]
        try:
            if self._is_farewell(text):
                session.current_state = "end"
                response = {
                    "action": "hangup",
                    "text_to_speak": "Thank you for calling. Goodbye.",
                    "transfer_number": None,
                }
            else:
                if text and session.agent_config.multilingual_enabled:
                    detected_lang = await self.llm_client.detect_language_preference(
                        user_input=text,
                        supported_languages=session.agent_config.supported_languages,
                        default_language=session.preferred_language or session.agent_config.default_greeting_language or "en",
                    )
                    session.preferred_language = detected_lang

                if session.current_state in {"greeting", "intent"}:
                    response = await self._handle_intent_entry(session, text)
                elif session.current_state == "intake":
                    response = await self._handle_intake_turn(session, text)
                elif session.current_state == "booking":
                    response = await self._complete_booking(session)
                elif session.current_state == "assist":
                    response = await self._handle_assist_turn(session, text)
                elif session.current_state == "transfer":
                    response = {
                        "action": "transfer",
                        "text_to_speak": "Please hold while I transfer you.",
                        "transfer_number": session.agent_config.fallback_phone,
                    }
                else:
                    response = {
                        "action": "hangup",
                        "text_to_speak": "Thank you for calling. Goodbye.",
                        "transfer_number": None,
                    }
        finally:
            self.sessions.save(session)
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.bind(call_id=call_id).info(
                "Turn processed",
                latency_ms=latency_ms,
                state_from=previous_state,
                state_to=session.current_state,
                intent=session.last_intent or "n/a",
                confidence=session.intent_confidence,
                action=(response.get("action") if "response" in locals() else "error"),
            )

        return response

    async def _handle_intent_entry(self, session: CallSession, text: str) -> dict[str, str | None]:
        if not text:
            session.current_state = "intent"
            session.last_intent = "unclear"
            session.intent_confidence = "unknown"
            return {
                "action": "speak",
                "text_to_speak": self._disambiguation_prompt(),
                "transfer_number": None,
            }

        intent_payload = await self.llm_client.detect_call_intent(
            user_input=text,
            context={
                "state": session.current_state,
                "business_name": session.agent_config.business_name,
                "services": [service.name for service in session.agent_config.service_catalog],
            },
        )
        intent, booking_id, confidence = self._resolve_intent(
            text=text,
            llm_intent=str(intent_payload.get("intent") or "unclear"),
            llm_confidence=str(intent_payload.get("confidence") or "unknown"),
        )
        session.last_intent = intent
        session.intent_confidence = confidence

        logger.bind(call_id=session.call_id).info(
            "Intent resolution",
            resolved_intent=intent,
            booking_id=booking_id,
            llm_intent=intent_payload.get("intent"),
            raw_text=text,
        )

        if intent == "end_call":
            session.current_state = "end"
            return {
                "action": "hangup",
                "text_to_speak": "Thank you for calling. Goodbye.",
                "transfer_number": None,
            }

        if intent == "transfer_request" and session.agent_config.fallback_phone:
            session.current_state = "transfer"
            return {
                "action": "transfer",
                "text_to_speak": "Sure, I will transfer you to a human agent now.",
                "transfer_number": session.agent_config.fallback_phone,
            }

        if intent == "track_booking" or booking_id:
            session.current_state = "assist"
            session.task_intent = "track_booking"
            if booking_id:
                return await self._handle_track_booking(session, booking_id)
            return {
                "action": "speak",
                "text_to_speak": "Sure. Please share your booking ID and I will check the latest status.",
                "transfer_number": None,
            }

        if intent in {"business_info", "pricing_info"}:
            session.current_state = "assist"
            session.task_intent = "business_info"
            info = await self.backend_client.answer_business_query(text, agent_id=session.agent_config.agent_id)
            return {
                "action": "speak",
                "text_to_speak": f"{info.get('answer', '')} If you want, I can help you book now.",
                "transfer_number": None,
            }

        if intent == "new_booking":
            session.current_state = "intake"
            session.task_intent = "new_booking"
            return await self._handle_intake_turn(session, text)

        session.current_state = "intent"
        return {"action": "speak", "text_to_speak": self._disambiguation_prompt(), "transfer_number": None}

    async def _handle_intake_turn(self, session: CallSession, text: str) -> dict[str, str | None]:
        result = await handle_intake(session, text, self.llm_client)
        action = result.get("action")

        if action == "next":
            return {
                "action": "speak",
                "text_to_speak": str(result.get("text") or prompts.INTAKE_FALLBACK),
                "transfer_number": None,
            }

        if action == "info_then_reask":
            info = await self.backend_client.answer_business_query(text, agent_id=session.agent_config.agent_id)
            session.awaiting_answer = True
            return {
                "action": "speak",
                "text_to_speak": (
                    f"{info.get('answer', '')} "
                    "To continue your booking, please answer the current booking question."
                ),
                "transfer_number": None,
            }

        if action == "out_of_coverage":
            session.current_state = "assist"
            session.task_intent = "business_info"
            return {
                "action": "speak",
                "text_to_speak": str(result.get("text") or "Currently we only serve Bahrain."),
                "transfer_number": None,
            }

        if action == "transfer":
            session.current_state = "transfer"
            return {
                "action": "transfer",
                "text_to_speak": str(result.get("text") or "I will transfer you to a human agent."),
                "transfer_number": str(result.get("transfer_number") or session.agent_config.fallback_phone),
            }

        if action == "disqualify":
            session.current_state = "end"
            return {
                "action": "hangup",
                "text_to_speak": str(result.get("text") or "Thank you for calling."),
                "transfer_number": None,
            }

        if action == "complete":
            session.current_state = "booking"
            return await self._complete_booking(session)

        session.awaiting_answer = True
        return {
            "action": "speak",
            "text_to_speak": prompts.INTAKE_FALLBACK,
            "transfer_number": None,
        }

    async def _handle_assist_turn(self, session: CallSession, text: str) -> dict[str, str | None]:
        if not text:
            return {
                "action": "speak",
                "text_to_speak": "Is there anything else I can help you with?",
                "transfer_number": None,
            }

        if session.booking_completed and self._looks_like_no_more_help(text):
            session.current_state = "end"
            session.task_intent = "undecided"
            return {
                "action": "hangup",
                "text_to_speak": "Perfect. Thanks for calling. Have a great day.",
                "transfer_number": None,
            }

        if session.booking_completed and self._looks_like_gratitude(text):
            session.task_intent = "undecided"
            return {
                "action": "speak",
                "text_to_speak": "You're welcome. If you need anything else, I can help now.",
                "transfer_number": None,
            }

        if session.booking_completed and self._looks_like_acknowledgement(text):
            booking_hint = (
                f"Your booking ID is {session.last_booking_ref}. "
                if session.last_booking_ref
                else ""
            )
            session.task_intent = "undecided"
            return {
                "action": "speak",
                "text_to_speak": (
                    f"Great. {booking_hint}"
                    "If you want, I can also check booking status or answer service questions."
                ),
                "transfer_number": None,
            }

        if session.booking_completed and self._looks_like_booking_id_request(text):
            if session.last_booking_ref:
                return {
                    "action": "speak",
                    "text_to_speak": (
                        f"Your latest booking ID is {session.last_booking_ref}. "
                        "You can say track booking and share this ID anytime."
                    ),
                    "transfer_number": None,
                }
            return {
                "action": "speak",
                "text_to_speak": "Your booking is confirmed. I can fetch details if you share the booking ID.",
                "transfer_number": None,
            }

        intent_payload = await self.llm_client.detect_call_intent(
            user_input=text,
            context={
                "state": "assist",
                "task_intent": session.task_intent,
                "booking_completed": session.booking_completed,
            },
        )
        intent, booking_id, confidence = self._resolve_intent(
            text=text,
            llm_intent=str(intent_payload.get("intent") or "unclear"),
            llm_confidence=str(intent_payload.get("confidence") or "unknown"),
        )
        session.last_intent = intent
        session.intent_confidence = confidence

        if intent == "end_call" or self._is_farewell(text):
            session.current_state = "end"
            return {
                "action": "hangup",
                "text_to_speak": "Thank you for calling. Goodbye.",
                "transfer_number": None,
            }

        if intent == "transfer_request" and session.agent_config.fallback_phone:
            session.current_state = "transfer"
            return {
                "action": "transfer",
                "text_to_speak": "Sure, I will transfer you now.",
                "transfer_number": session.agent_config.fallback_phone,
            }

        wants_tracking = (
            intent == "track_booking"
            or (session.task_intent == "track_booking" and (booking_id or self._looks_like_tracking_request(text)))
        )
        if wants_tracking:
            session.task_intent = "track_booking"
            if not booking_id:
                if session.last_booking_ref:
                    return await self._handle_track_booking(session, session.last_booking_ref)
                return {
                    "action": "speak",
                    "text_to_speak": "Please share your booking ID to continue tracking.",
                    "transfer_number": None,
                }
            return await self._handle_track_booking(session, booking_id)

        if intent == "new_booking":
            session.task_intent = "new_booking"
            session.current_state = "intake"
            session.current_question_index = 0
            session.awaiting_answer = False
            session.collected_answers.clear()
            session.booking_completed = False
            session.post_booking_stage = None
            return await self._handle_intake_turn(session, text)

        if intent in {"business_info", "pricing_info"}:
            info = await self.backend_client.answer_business_query(text, agent_id=session.agent_config.agent_id)
            follow_up = " Would you like to make a booking now or ask another question?"
            return {
                "action": "speak",
                "text_to_speak": f"{info.get('answer', '')}{follow_up}",
                "transfer_number": None,
            }
        return {
            "action": "speak",
            "text_to_speak": self._disambiguation_prompt(),
            "transfer_number": None,
        }

    async def _handle_track_booking(self, session: CallSession, booking_id: str) -> dict[str, str | None]:
        payload = await self.backend_client.track_booking(booking_id, agent_id=session.agent_config.agent_id)
        status = str(payload.get("status", "")).lower()
        message = str(payload.get("message") or "I checked the booking status.").strip()
        session.task_intent = "undecided"
        session.last_booking_ref = booking_id

        if status == "not_found":
            return {
                "action": "speak",
                "text_to_speak": f"{message} You can share another booking ID or create a new booking now.",
                "transfer_number": None,
            }

        detail_parts: list[str] = [message]
        for key, prefix in (("service_name", "Service"), ("location", "Location"), ("preferred_time", "Time")):
            value = payload.get(key)
            if value:
                detail_parts.append(f"{prefix}: {value}.")

        detail_parts.append("Do you need any other help?")
        return {
            "action": "speak",
            "text_to_speak": " ".join(detail_parts),
            "transfer_number": None,
        }

    async def _complete_booking(self, session: CallSession) -> dict[str, str | None]:
        if session.booking_completed:
            session.current_state = "assist"
            session.task_intent = "undecided"
            return {
                "action": "speak",
                "text_to_speak": "Your booking is already recorded. Anything else I can help with?",
                "transfer_number": None,
            }

        booking_action = await handle_booking(session, self.backend_client, self.llm_client)
        action = booking_action.get("action")

        if action == "speak":
            booking_ref = booking_action.get("booking_ref")
            if booking_ref:
                session.last_booking_ref = str(booking_ref)
            session.last_booking_payload = {
                "service_type": session.collected_answers.get("service_type"),
                "location": session.collected_answers.get("location"),
                "preferred_time": session.collected_answers.get("preferred_time"),
            }
            session.booking_completed = True
            session.current_state = "assist"
            session.task_intent = "undecided"
            session.post_booking_stage = "completed"
            return {
                "action": "speak",
                "text_to_speak": str(booking_action.get("text_to_speak") or "Booking recorded."),
                "transfer_number": None,
            }

        if action == "transfer":
            session.current_state = "transfer"
            return {
                "action": "transfer",
                "text_to_speak": str(booking_action.get("text_to_speak") or "Please hold for transfer."),
                "transfer_number": str(booking_action.get("transfer_number") or session.agent_config.fallback_phone),
            }

        session.current_state = "end"
        return {
            "action": "hangup",
            "text_to_speak": str(booking_action.get("text_to_speak") or "Thank you for calling."),
            "transfer_number": None,
        }

    def end_call(self, call_id: str) -> None:
        self.sessions.delete(call_id)
        logger.bind(call_id=call_id).info("Conversation session ended")

    async def _get_or_create_session(self, call_id: str, agent_id: str | None = None) -> CallSession:
        existing = self.sessions.get(call_id)
        if existing:
            return existing

        await self.start_session(call_id=call_id, agent_id=agent_id)
        created = self.sessions.get(call_id)
        if created is None:
            raise ConversationEngineError(f"Unable to create session for call_id={call_id}")
        return created
