from __future__ import annotations

import time
from typing import Literal, Sequence

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from src.api.exceptions import ConversationEngineError
from src.core.types import AgentConfig
from src.core.services.backend_client import BackendClient
from src.core.services.openai import OpenAIClient
from src.core.services.elevenlabs import ElevenLabsClient
from . import prompts
from .handlers import build_system_messages, handle_intake, handle_booking

State = Literal["greeting", "intake", "booking", "transfer", "end"]


class CallSession(BaseModel):
    call_id: str
    agent_config: AgentConfig
    current_state: State = "greeting"
    collected_answers: dict[str, str] = Field(default_factory=dict)
    current_question_index: int = 0
    awaiting_answer: bool = False
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
        for k in expired:
            self._store.pop(k, None)


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

    async def start_session(self, call_id: str, agent_id: str | None = None) -> dict[str, str]:
        agent_config = await self.backend_client.fetch_agent_config(agent_id)
        session = CallSession(call_id=call_id, agent_config=agent_config)
        self.sessions.save(session)
        greeting = agent_config.greeting or prompts.GREETING_TEMPLATE
        audio_url = await self.tts_client.synthesize_text(greeting)
        logger.info("Conversation started", call_id=call_id)
        return {"text": greeting, "audio_url": audio_url}

    async def process_user_input(self, call_id: str, transcribed_text: str, agent_id: str | None = None) -> dict[str, str | None]:
        session = await self._get_or_create_session(call_id, agent_id)
        self.sessions.prune()
        logger.bind(call_id=call_id).debug("Processing user input", state=session.current_state)

        if transcribed_text and transcribed_text.lower() in {"cancel", "stop", "hang up"}:
            session.current_state = "end"
            self.sessions.save(session)
            return {
                "action": "hangup",
                "text_to_speak": "تم الإلغاء، شكرًا لتواصلك.",
                "transfer_number": None,
            }

        if session.current_state == "greeting":
            session.current_state = "intake"
            self.sessions.save(session)
            result = await handle_intake(session, transcribed_text)
            if result["action"] == "next":
                return {
                    "action": "speak",
                    "text_to_speak": result["text"],
                    "transfer_number": None,
                }
            if result["action"] == "disqualify":
                session.current_state = "end"
                self.sessions.save(session)
                return {
                    "action": "hangup",
                    "text_to_speak": result["text"],
                    "transfer_number": None,
                }
            session.current_state = "booking"
            self.sessions.save(session)
            return await self._booking_or_decision(session)

        if session.current_state == "intake":
            result = await handle_intake(session, transcribed_text)
            self.sessions.save(session)
            if result["action"] == "next":
                return {
                    "action": "speak",
                    "text_to_speak": result["text"],
                    "transfer_number": None,
                }
            if result["action"] == "disqualify":
                session.current_state = "end"
                self.sessions.save(session)
                return {
                    "action": "hangup",
                    "text_to_speak": result["text"],
                    "transfer_number": None,
                }
            # intake finished
            session.current_state = "booking"
            self.sessions.save(session)
            return await self._booking_or_decision(session)

        if session.current_state == "booking":
            booking = await handle_booking(session, self.backend_client)
            session.current_state = "end"
            self.sessions.save(session)
            return booking

        if session.current_state == "transfer":
            transfer_number = session.agent_config.fallback_phone
            return {
                "action": "transfer",
                "text_to_speak": "سأحوّلك الآن إلى موظف مختص. Please hold while I connect you.",
                "transfer_number": transfer_number,
            }

        return {"action": "hangup", "text_to_speak": "Thank you for calling. Goodbye!", "transfer_number": None}

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

    async def _booking_or_decision(self, session: CallSession) -> dict[str, str | None]:
        if session.agent_config.fallback_phone:
            session.current_state = "transfer"
            self.sessions.save(session)
            return {
                "action": "transfer",
                "text_to_speak": "سأحوّلك الآن إلى زميلي لإتمام الطلب.",
                "transfer_number": session.agent_config.fallback_phone,
            }

        messages = self._compose_messages(session)
        reply = await self.llm_client.generate_reply(messages)
        return {
            "action": "speak",
            "text_to_speak": reply,
            "transfer_number": None,
        }

    def _compose_messages(self, session: CallSession) -> Sequence[dict[str, str]]:
        messages = list(build_system_messages(session.agent_config))
        messages.extend(prompts.FEW_SHOT_EXAMPLES)

        summary_lines = [
            f"Question {idx+1}: {session.agent_config.intake_questions[idx]} -> {session.collected_answers.get(f'q{idx}', 'N/A')}"
            for idx in range(len(session.agent_config.intake_questions))
        ]
        intake_summary = "\n".join(summary_lines)
        messages.append({"role": "assistant", "content": "Here is what we captured so far:"})
        messages.append({"role": "user", "content": intake_summary})
        return messages
