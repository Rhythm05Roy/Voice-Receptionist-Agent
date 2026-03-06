"""Conversation engine — LLM-driven dialogue with function calling.

The engine no longer uses a rigid state machine or question-by-question
intake.  Instead, it:

1. Builds a rich system prompt from AgentConfig (services, policies, FAQs).
2. Passes full conversation history + user message to GPT-4o.
3. GPT decides what to say AND when to call tools (book, track, transfer).
4. Tool results are sent back to GPT for a natural follow-up response.

This lets the agent handle corrections, side-questions, frustration,
and multi-intent turns naturally.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.api.exceptions import ConversationEngineError
from src.core.services.backend_client import BackendClient
from src.core.services.elevenlabs import ElevenLabsClient
from src.core.services.openai import OpenAIClient

from . import prompts
from .handlers import execute_tool_call


class CallSession(BaseModel):
    call_id: str
    agent_config: Any  # AgentConfig
    preferred_language: str | None = None
    created_at: float = Field(default_factory=time.monotonic)
    last_activity: float = Field(default_factory=time.monotonic)

    # Full conversation history (role-tagged messages for LLM)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    # Turn counter for analytics
    turn_count: int = 0
    # Session-level state (booking refs, flags)
    state: dict[str, Any] = Field(default_factory=dict)
    # Compiled system prompt (cached per session)
    system_prompt: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def add_message(self, role: str, content: str) -> None:
        """Append a message to conversation history with sliding window."""
        if content:
            self.conversation_history.append({"role": role, "content": content})
            # Keep last 30 messages to stay within token limits
            if len(self.conversation_history) > 30:
                self.conversation_history = self.conversation_history[-30:]
        if role == "user":
            self.turn_count += 1

    @property
    def call_duration_seconds(self) -> float:
        return time.monotonic() - self.created_at


class SessionManagerProtocol(Protocol):
    async def get(self, call_id: str) -> CallSession | None: ...
    async def save(self, session: CallSession) -> None: ...
    async def delete(self, call_id: str) -> None: ...
    async def prune(self) -> None: ...
    async def ping(self) -> bool: ...


class CallSessionManager:
    """In-memory async session manager."""

    def __init__(self, ttl_seconds: int = 1800):
        self.ttl = ttl_seconds
        self._store: dict[str, CallSession] = {}

    async def get(self, call_id: str) -> CallSession | None:
        key = f"call:{call_id}"
        session = self._store.get(key)
        if not session:
            return None
        if time.monotonic() - session.last_activity > self.ttl:
            await self.delete(call_id)
            return None
        session.touch()
        return session

    async def save(self, session: CallSession) -> None:
        session.touch()
        self._store[f"call:{session.call_id}"] = session

    async def delete(self, call_id: str) -> None:
        self._store.pop(f"call:{call_id}", None)

    async def prune(self) -> None:
        now = time.monotonic()
        expired = [k for k, s in self._store.items() if now - s.last_activity > self.ttl]
        for k in expired:
            self._store.pop(k, None)

    async def ping(self) -> bool:
        return True


class ConversationEngine:
    """LLM-driven conversation engine with function calling."""

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

    async def has_session(self, call_id: str) -> bool:
        return (await self.sessions.get(call_id)) is not None

    async def start_session(self, call_id: str, agent_id: str | None = None) -> dict[str, str]:
        """Create a new call session and return the greeting."""
        agent_config = await self.backend_client.fetch_agent_config(agent_id)
        preferred_language = agent_config.default_greeting_language or agent_config.language or "en"

        # Build the system prompt once from agent config
        system_prompt = prompts.build_system_prompt(agent_config)

        session = CallSession(
            call_id=call_id,
            agent_config=agent_config,
            preferred_language=preferred_language.lower(),
            system_prompt=system_prompt,
        )

        # Greeting
        greeting = agent_config.greeting or prompts.GREETING_TEMPLATE
        session.add_message("assistant", greeting)
        await self.sessions.save(session)

        # TTS for greeting
        audio_url = ""
        try:
            voice_id = agent_config.language_voice_map.get(preferred_language, None)
            audio_url = await self.tts_client.synthesize_text(greeting, voice_id=voice_id)
        except Exception:  # noqa: BLE001
            logger.warning("Greeting TTS failed", call_id=call_id)

        logger.bind(call_id=call_id).info("Session started", agent_id=agent_config.agent_id)
        return {"text": greeting, "audio_url": audio_url}

    async def process_user_input(
        self,
        call_id: str,
        transcribed_text: str,
        agent_id: str | None = None,
    ) -> dict[str, str | None]:
        """Process one turn of conversation using GPT with full history."""
        session = await self._get_or_create_session(call_id, agent_id)
        await self.sessions.prune()
        start = time.perf_counter()

        text = (transcribed_text or "").strip()

        # Record user turn
        session.add_message("user", text)

        logger.bind(call_id=call_id).debug(
            "Processing turn",
            turn_count=session.turn_count,
            history_len=len(session.conversation_history),
        )

        # Check call duration
        max_minutes = getattr(session.agent_config, "max_call_duration_minutes", 15) or 15
        if session.call_duration_seconds > max_minutes * 60:
            session.add_message(
                "assistant",
                "We've reached the maximum call duration. Please call back for further assistance. Goodbye.",
            )
            await self.sessions.save(session)
            return {
                "action": "hangup",
                "text_to_speak": "We've reached the maximum call duration. Please call back for further assistance. Goodbye.",
                "transfer_number": None,
            }

        try:
            response = await self._llm_turn(session, text)
        except Exception:  # noqa: BLE001
            logger.bind(call_id=call_id).exception("LLM turn failed")
            response = {
                "action": "speak",
                "text_to_speak": "I'm sorry, I'm having a brief issue. Could you repeat that?",
                "transfer_number": None,
            }

        # Record assistant response
        if response.get("text_to_speak"):
            session.add_message("assistant", response["text_to_speak"])

        await self.sessions.save(session)

        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.bind(call_id=call_id).info(
            "Turn processed",
            latency_ms=latency_ms,
            action=response.get("action"),
            turn_count=session.turn_count,
        )

        return response

    async def _llm_turn(self, session: CallSession, user_text: str) -> dict[str, str | None]:
        """Core LLM interaction — handles text responses and tool calls."""

        # Call GPT with full conversation + tools
        result = await self.llm_client.conversation_turn(
            system_prompt=session.system_prompt,
            conversation_history=session.conversation_history[:-1],  # exclude the latest user msg (already in call)
            user_message=user_text,
        )

        response_text = result.get("response_text", "")
        tool_calls = result.get("tool_calls", [])

        # If no tool calls, just return the text response
        if not tool_calls:
            return {
                "action": "speak",
                "text_to_speak": response_text or "Is there anything else I can help you with?",
                "transfer_number": None,
            }

        # Process tool calls
        tool_results: list[dict[str, Any]] = []
        final_action = "speak"
        transfer_number = None

        for tc in tool_calls:
            tool_output = await execute_tool_call(
                tool_name=tc["name"],
                arguments=tc["arguments"],
                backend_client=self.backend_client,
                agent_config=session.agent_config,
                session_state=session.state,
            )
            tool_results.append({
                "tool_call_id": tc["id"],
                "output": tool_output,
            })

            # Check for special actions
            if tc["name"] == "end_call":
                final_action = "hangup"
            elif tc["name"] == "transfer_to_human":
                final_action = "transfer"
                transfer_number = tool_output.get("transfer_number")

        # Send tool results back to GPT for natural follow-up
        try:
            raw_message = result.get("raw_message")
            # Serialize assistant message with tool calls
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if response_text:
                assistant_msg["content"] = response_text
            else:
                assistant_msg["content"] = None

            if raw_message and hasattr(raw_message, "tool_calls") and raw_message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc_raw.id,
                        "type": "function",
                        "function": {
                            "name": tc_raw.function.name,
                            "arguments": tc_raw.function.arguments,
                        },
                    }
                    for tc_raw in raw_message.tool_calls
                ]

            follow_up = await self.llm_client.continue_after_tool(
                system_prompt=session.system_prompt,
                conversation_history=session.conversation_history[:-1],
                tool_call_message=assistant_msg,
                tool_results=tool_results,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to get follow-up after tools, using default")
            # Fallback: construct response from tool outputs
            messages = []
            for tr in tool_results:
                msg = tr["output"].get("message", "")
                if msg:
                    messages.append(msg)
            follow_up = " ".join(messages) if messages else "Done. Is there anything else?"

        return {
            "action": final_action,
            "text_to_speak": follow_up,
            "transfer_number": transfer_number,
        }

    async def end_call(self, call_id: str) -> None:
        await self.sessions.delete(call_id)
        logger.bind(call_id=call_id).info("Session ended")

    async def _get_or_create_session(self, call_id: str, agent_id: str | None = None) -> CallSession:
        existing = await self.sessions.get(call_id)
        if existing:
            return existing

        await self.start_session(call_id, agent_id)
        created = await self.sessions.get(call_id)
        if created is None:
            raise ConversationEngineError(f"Unable to create session for {call_id}")
        return created
