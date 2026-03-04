from __future__ import annotations

import re
import time
from typing import Literal, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.api.exceptions import ConversationEngineError
from src.core.services.backend_client import BackendClient
from src.core.services.elevenlabs import ElevenLabsClient
from src.core.services.openai import OpenAIClient
from src.core.types import AgentConfig

from . import prompts
from .handlers import build_system_messages, handle_booking, handle_intake

State = Literal["greeting", "intake", "booking", "assist", "transfer", "end"]
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
    out_of_coverage: bool = False
    booking_completed: bool = False
    service_prompt_retries: int = 0
    created_at: float = Field(default_factory=time.monotonic)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def has_more_questions(self) -> bool:
        return self.current_question_index < len(self.agent_config.intake_questions)

    @property
    def next_question(self) -> str | None:
        if self.has_more_questions:
            return self.agent_config.intake_questions[self.current_question_index]
        return None


class CallSessionManager:
    """In-memory session store with TTL.

    Session keys: f"call:{call_uuid}". TTL set to 30 minutes for MVP.
    To swap to Redis later, replace the dict with `redis.asyncio.Redis` and
    serialize CallSession using model_dump()/model_validate().
    """

    def __init__(self, ttl_seconds: int = 1800):
        self.ttl = ttl_seconds
        self._store: dict[str, CallSession] = {}

    def _key(self, call_id: str) -> str:
        return f"call:{call_id}"

    def get(self, call_id: str) -> CallSession | None:
        key = self._key(call_id)
        session = self._store.get(key)
        if not session:
            return None
        if time.monotonic() - session.created_at > self.ttl:
            self.delete(call_id)
            return None
        return session

    def save(self, session: CallSession) -> None:
        self._store[self._key(session.call_id)] = session

    def delete(self, call_id: str) -> None:
        self._store.pop(self._key(call_id), None)

    def prune(self) -> None:
        now = time.monotonic()
        expired = [k for k, s in self._store.items() if now - s.created_at > self.ttl]
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
        keywords = [
            "bye",
            "goodbye",
            "thanks bye",
            "thank you bye",
            "no thanks",
            "that's all",
            "thats all",
            "khuda hafiz",
            "allah hafiz",
            "ok bye",
            "okay bye",
            "bye bye",
            "no no no bye",
        ]
        return any(k in normalized for k in keywords)

    @staticmethod
    def _extract_booking_id(text: str) -> str | None:
        if not text:
            return None
        match = re.search(r"\b([A-Za-z]{2,}-\d{4,})\b", text)
        return match.group(1) if match else None

    @staticmethod
    def _looks_like_pricing_question(text: str) -> bool:
        normalized = text.lower()
        return any(k in normalized for k in ["price", "pricing", "cost", "charge", "rate", "quotation", "quote"])

    @staticmethod
    def _looks_like_service_question(text: str) -> bool:
        normalized = text.lower()
        return any(k in normalized for k in ["service", "services", "provide", "offer", "available"])

    @staticmethod
    def _looks_like_booking_detail_question(text: str) -> bool:
        normalized = text.lower()
        return any(
            k in normalized
            for k in ["what details", "what information", "what else", "what do you need", "required", "require"]
        )

    @staticmethod
    def _looks_like_tracking_request(text: str) -> bool:
        normalized = text.lower()
        return any(
            k in normalized
            for k in ["track", "booking status", "status", "booking id", "already booked", "previously booked", "my booking"]
        )

    @staticmethod
    def _looks_like_new_booking_request(text: str) -> bool:
        normalized = text.lower()
        return any(k in normalized for k in ["book", "appointment", "visit", "need service", "new booking"])

    def _mentions_known_service(self, session: CallSession, text: str) -> bool:
        normalized = text.lower()
        for service in session.agent_config.service_catalog:
            tokens = [service.name, service.service_id, *service.keywords]
            if any(token.lower() in normalized for token in tokens if token):
                return True
        return False

    async def start_session(self, call_id: str, agent_id: str | None = None) -> dict[str, str]:
        agent_config = await self.backend_client.fetch_agent_config(agent_id)
        session = CallSession(
            call_id=call_id,
            agent_config=agent_config,
            preferred_language=agent_config.language or "en",
        )
        self.sessions.save(session)
        greeting = agent_config.greeting or prompts.GREETING_TEMPLATE
        audio_url = ""
        try:
            audio_url = await self.tts_client.synthesize_text(greeting)
        except Exception:  # noqa: BLE001
            logger.warning("Greeting TTS failed, continuing with text only", call_id=call_id)
        logger.info("Conversation started", call_id=call_id)
        return {"text": greeting, "audio_url": audio_url}

    async def process_user_input(
        self,
        call_id: str,
        transcribed_text: str,
        agent_id: str | None = None,
    ) -> dict[str, str | None]:
        session = await self._get_or_create_session(call_id, agent_id)
        self.sessions.prune()
        logger.bind(call_id=call_id).debug("Processing user input", state=session.current_state, task=session.task_intent)

        text = (transcribed_text or "").strip()
        if text.lower() in {"cancel", "stop", "hang up"}:
            session.current_state = "end"
            self.sessions.save(session)
            return {
                "action": "hangup",
                "text_to_speak": "Request cancelled. Thank you for contacting us.",
                "transfer_number": None,
            }

        if session.current_state == "greeting":
            response = await self._handle_greeting_turn(session, text)
            self.sessions.save(session)
            return response

        if session.current_state == "intake":
            result = await handle_intake(session, text, self.llm_client)
            self.sessions.save(session)
            if result["action"] == "next":
                return {"action": "speak", "text_to_speak": result["text"], "transfer_number": None}
            if result["action"] == "disqualify":
                session.current_state = "end"
                self.sessions.save(session)
                return {"action": "hangup", "text_to_speak": result["text"], "transfer_number": None}
            if result["action"] == "out_of_coverage":
                session.current_state = "assist"
                session.task_intent = "business_info"
                self.sessions.save(session)
                return {"action": "speak", "text_to_speak": result["text"], "transfer_number": None}

            session.current_state = "booking"
            self.sessions.save(session)
            return await self._complete_booking(session)

        if session.current_state == "booking":
            return await self._complete_booking(session)

        if session.current_state == "assist":
            response = await self._handle_assist_turn(session, text)
            self.sessions.save(session)
            return response

        if session.current_state == "transfer":
            transfer_number = session.agent_config.fallback_phone
            return {
                "action": "transfer",
                "text_to_speak": "I will transfer you to a specialist now. Please hold.",
                "transfer_number": transfer_number,
            }

        return {"action": "hangup", "text_to_speak": "Thank you for calling. Goodbye!", "transfer_number": None}

    async def _handle_greeting_turn(self, session: CallSession, text: str) -> dict[str, str | None]:
        if not text:
            return {
                "action": "speak",
                "text_to_speak": (
                    "Tell me what you want to do: new booking, track existing booking by booking ID, "
                    "or know about services and pricing."
                ),
                "transfer_number": None,
            }

        intent_payload = await self._detect_call_intent(text)
        intent = intent_payload.get("intent", "unclear")
        extracted_booking_id = intent_payload.get("booking_id", "") or self._extract_booking_id(text) or ""

        if intent == "track_booking":
            session.current_state = "assist"
            session.task_intent = "track_booking"
            if extracted_booking_id:
                return await self._handle_track_booking(session, extracted_booking_id)
            return {
                "action": "speak",
                "text_to_speak": "Sure. Please share your booking ID so I can track it.",
                "transfer_number": None,
            }

        if intent in {"business_info", "pricing_info"}:
            session.current_state = "assist"
            session.task_intent = "business_info"
            info = await self.backend_client.answer_business_query(text, agent_id=session.agent_config.agent_id)
            follow_up = " If you want to book now, tell me the service name."
            return {
                "action": "speak",
                "text_to_speak": f"{info.get('answer', '')}{follow_up}",
                "transfer_number": None,
            }

        if intent == "new_booking":
            session.current_state = "intake"
            session.task_intent = "new_booking"
            result = await handle_intake(session, text, self.llm_client)
            if result["action"] == "next":
                return {"action": "speak", "text_to_speak": result["text"], "transfer_number": None}
            if result["action"] == "disqualify":
                session.current_state = "end"
                return {"action": "hangup", "text_to_speak": result["text"], "transfer_number": None}
            if result["action"] == "out_of_coverage":
                session.current_state = "assist"
                session.task_intent = "business_info"
                return {"action": "speak", "text_to_speak": result["text"], "transfer_number": None}
            session.current_state = "booking"
            return await self._complete_booking(session)

        return {
            "action": "speak",
            "text_to_speak": (
                "I can help with new booking, booking tracking by booking ID, or service information. "
                "Tell me what you want to do."
            ),
            "transfer_number": None,
        }

    async def _handle_assist_turn(self, session: CallSession, text: str) -> dict[str, str | None]:
        if self._is_farewell(text):
            session.current_state = "end"
            return {"action": "hangup", "text_to_speak": "Thank you for calling. Goodbye.", "transfer_number": None}

        if not text:
            return {"action": "speak", "text_to_speak": "Is there anything else I can help you with?", "transfer_number": None}

        # track-booking flow first if caller asked for tracking
        if session.task_intent == "track_booking" or self._looks_like_tracking_request(text):
            session.task_intent = "track_booking"
            booking_id = self._extract_booking_id(text)
            if booking_id:
                return await self._handle_track_booking(session, booking_id)

            # Let caller switch from tracking to booking naturally
            if self._looks_like_new_booking_request(text):
                session.task_intent = "new_booking"
                session.current_state = "intake"
                result = await handle_intake(session, text, self.llm_client)
                if result["action"] == "next":
                    return {"action": "speak", "text_to_speak": result["text"], "transfer_number": None}
                return {
                    "action": "speak",
                    "text_to_speak": "Please share the service you want to book.",
                    "transfer_number": None,
                }

            return {
                "action": "speak",
                "text_to_speak": "Please share your booking ID (example DUMMY-10001) so I can track it.",
                "transfer_number": None,
            }

        # if caller asks for new booking from assist mode, switch to booking flow
        if self._looks_like_new_booking_request(text) or self._mentions_known_service(session, text):
            session.task_intent = "new_booking"
            session.current_state = "intake"
            result = await handle_intake(session, text, self.llm_client)
            if result["action"] == "next":
                return {"action": "speak", "text_to_speak": result["text"], "transfer_number": None}
            if result["action"] == "disqualify":
                session.current_state = "end"
                return {"action": "hangup", "text_to_speak": result["text"], "transfer_number": None}
            return await self._complete_booking(session)

        if self._looks_like_service_question(text) or self._looks_like_pricing_question(text):
            info = await self.backend_client.answer_business_query(text, agent_id=session.agent_config.agent_id)
            return {"action": "speak", "text_to_speak": info.get("answer", ""), "transfer_number": None}

        if self._looks_like_booking_detail_question(text):
            required = session.agent_config.booking_required_fields or [
                "service type",
                "location",
                "preferred time",
                "contact number",
            ]
            return {
                "action": "speak",
                "text_to_speak": f"To complete booking we need: {', '.join(required)}.",
                "transfer_number": None,
            }

        messages = list(build_system_messages(session.agent_config, session.preferred_language))
        messages.extend(prompts.FEW_SHOT_EXAMPLES)
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are in assist mode. Be helpful, concise, and practical. "
                    "If user asks business questions, answer clearly then ask what they want next."
                ),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Collected answers: {session.collected_answers}\n"
                    f"Task intent: {session.task_intent}\n"
                    f"Caller: {text}"
                ),
            }
        )
        reply = await self.llm_client.generate_reply(messages)
        return {"action": "speak", "text_to_speak": reply, "transfer_number": None}

    async def _handle_track_booking(self, session: CallSession, booking_id: str) -> dict[str, str | None]:
        payload = await self.backend_client.track_booking(booking_id, agent_id=session.agent_config.agent_id)
        status = str(payload.get("status", "")).lower()
        message = str(payload.get("message", "")).strip() or "I checked the booking status."

        if status == "not_found":
            return {
                "action": "speak",
                "text_to_speak": f"{message} You can share another booking ID or start a new booking now.",
                "transfer_number": None,
            }

        extra = []
        service_name = payload.get("service_name")
        location = payload.get("location")
        preferred_time = payload.get("preferred_time")
        if service_name:
            extra.append(f"Service: {service_name}.")
        if location:
            extra.append(f"Location: {location}.")
        if preferred_time:
            extra.append(f"Time: {preferred_time}.")

        text = " ".join([message, *extra, "Do you need anything else?"]).strip()
        return {"action": "speak", "text_to_speak": text, "transfer_number": None}

    async def _detect_call_intent(self, text: str) -> dict[str, str]:
        if hasattr(self.llm_client, "detect_call_intent"):
            try:
                payload = await self.llm_client.detect_call_intent(text)
                if isinstance(payload, dict):
                    return {
                        "intent": str(payload.get("intent", "unclear")),
                        "booking_id": str(payload.get("booking_id", "")),
                    }
            except Exception:  # noqa: BLE001
                logger.exception("Intent detection via LLM failed; using heuristic fallback")

        if self._looks_like_tracking_request(text):
            return {"intent": "track_booking", "booking_id": self._extract_booking_id(text) or ""}
        if self._looks_like_pricing_question(text):
            return {"intent": "pricing_info", "booking_id": ""}
        if self._looks_like_service_question(text):
            return {"intent": "business_info", "booking_id": ""}
        if self._looks_like_new_booking_request(text):
            return {"intent": "new_booking", "booking_id": ""}
        return {"intent": "unclear", "booking_id": ""}

    def end_call(self, call_id: str) -> None:
        self.sessions.delete(call_id)
        logger.info("Conversation ended", call_id=call_id)

    async def _get_or_create_session(self, call_id: str, agent_id: str | None = None) -> CallSession:
        session = self.sessions.get(call_id)
        if session:
            return session
        await self.start_session(call_id, agent_id)
        session = self.sessions.get(call_id)
        if not session:
            raise ConversationEngineError(f"Unable to create session for {call_id}")
        return session

    async def _complete_booking(self, session: CallSession) -> dict[str, str | None]:
        if session.out_of_coverage:
            session.current_state = "assist"
            session.task_intent = "business_info"
            self.sessions.save(session)
            return {
                "action": "speak",
                "text_to_speak": (
                    "Your location appears outside Bahrain. I can still answer questions and arrange human follow-up if needed."
                ),
                "transfer_number": None,
            }

        if session.booking_completed:
            session.current_state = "assist"
            session.task_intent = "business_info"
            self.sessions.save(session)
            return {
                "action": "speak",
                "text_to_speak": "Your booking is already recorded. What else can I help you with?",
                "transfer_number": None,
            }

        booking = await handle_booking(session, self.backend_client, self.llm_client)
        if booking.get("action") == "speak":
            session.booking_completed = True
            session.current_state = "assist"
            session.task_intent = "business_info"
        elif booking.get("action") == "transfer":
            session.current_state = "transfer"
        else:
            session.current_state = "end"
        self.sessions.save(session)
        return booking

    def _compose_messages(self, session: CallSession) -> Sequence[dict[str, str]]:
        messages = list(build_system_messages(session.agent_config, session.preferred_language))
        messages.extend(prompts.FEW_SHOT_EXAMPLES)

        summary_lines = [
            f"Question {idx + 1}: {session.agent_config.intake_questions[idx]} -> {session.collected_answers.get(f'q{idx}', 'N/A')}"
            for idx in range(len(session.agent_config.intake_questions))
        ]
        intake_summary = "\n".join(summary_lines)
        messages.append({"role": "assistant", "content": "Here is what we captured so far:"})
        messages.append({"role": "user", "content": intake_summary})
        return messages
