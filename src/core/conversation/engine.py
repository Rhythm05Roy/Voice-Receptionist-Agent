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
import re
import time
import uuid
from typing import Any, Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.api.exceptions import ConversationEngineError
from src.core.services.backend_client import BackendClient
from src.core.services.call_store import CallStore
from src.core.services.elevenlabs import ElevenLabsClient
from src.core.services.openai import OpenAIClient

from . import prompts
from .handlers import execute_tool_call


class CallSession(BaseModel):
    call_id: str
    agent_config: Any  # AgentConfig
    agent_id: str = ""
    preferred_language: str | None = None
    created_at: float = Field(default_factory=time.monotonic)
    last_activity: float = Field(default_factory=time.monotonic)
    context_updated_at: float = Field(default_factory=time.monotonic)
    is_test: bool = False

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
        call_store: CallStore | None = None,
        context_refresh_ttl_seconds: int = 60,
    ):
        self.backend_client = backend_client
        self.llm_client = llm_client
        self.tts_client = tts_client
        self.call_store = call_store or CallStore()
        self.environment = environment
        self.sessions = session_manager or CallSessionManager()
        self.context_refresh_ttl_seconds = max(1, context_refresh_ttl_seconds)

    async def has_session(self, call_id: str) -> bool:
        return (await self.sessions.get(call_id)) is not None

    async def start_session(
        self,
        call_id: str,
        agent_id: str | None = None,
        is_test: bool = False,
        caller_number: str | None = None,
        called_number: str | None = None,
    ) -> dict[str, str]:
        """Create a new call session and return the greeting."""
        agent_config = await self.backend_client.fetch_agent_config(agent_id)
        preferred_language = agent_config.default_greeting_language or agent_config.language or "en"

        # Build the system prompt once from agent config
        system_prompt = prompts.build_system_prompt(agent_config)

        session = CallSession(
            call_id=call_id,
            agent_config=agent_config,
            agent_id=agent_config.agent_id,
            is_test=is_test,
            preferred_language=preferred_language.lower(),
            system_prompt=system_prompt,
        )
        session.state["customer_details"] = {
            "phone_number": caller_number or "",
        }
        session.state["call_context"] = {
            "called_number": called_number or "",
        }

        # Greeting
        greeting = agent_config.greeting or prompts.GREETING_TEMPLATE
        session.add_message("assistant", greeting)
        await self.sessions.save(session)
        self.call_store.start_call(
            call_id=call_id,
            agent_id=agent_config.agent_id,
            caller_number=caller_number or "",
            called_number=called_number or "",
            is_test=is_test,
        )

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
        await self._refresh_session_context_if_stale(session)

        text = (transcribed_text or "").strip()

        # Record user turn
        session.add_message("user", text)
        self._update_active_intent_from_user_text(session, text)
        await self._update_language_preference(session, text)

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

        business_context = self._get_business_context_for_turn(session, text)
        deterministic_response = self._try_business_info_response(session, text, business_context)
        if deterministic_response is not None:
            deterministic_response["text_to_speak"] = await self._render_response_for_language(
                session,
                deterministic_response["text_to_speak"] or "",
            )
            session.state["interaction_type"] = session.state.get("interaction_type") or "query"
            queries = session.state.setdefault("queries", [])
            queries.append({"text": text, "type": "business_info"})
            session.add_message("assistant", deterministic_response["text_to_speak"])
            await self.sessions.save(session)
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.bind(call_id=call_id).info(
                "Turn processed via business-info fast path",
                latency_ms=latency_ms,
                action=deterministic_response.get("action"),
                turn_count=session.turn_count,
            )
            return deterministic_response

        try:
            response = await self._llm_turn(session, text, business_context=business_context)
        except Exception:  # noqa: BLE001
            logger.bind(call_id=call_id).exception("LLM turn failed")
            response = {
                "action": "speak",
                "text_to_speak": "I'm sorry, I'm having a brief issue. Could you repeat that?",
                "transfer_number": None,
            }

        # Record assistant response
        if response.get("text_to_speak"):
            response["text_to_speak"] = await self._render_response_for_language(
                session,
                response["text_to_speak"] or "",
            )
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

    async def _update_language_preference(self, session: CallSession, user_text: str) -> None:
        text = (user_text or "").strip()
        if not text:
            return

        supported_languages = list(session.agent_config.supported_languages or [])
        default_language = (
            session.preferred_language
            or session.agent_config.default_greeting_language
            or session.agent_config.language
            or "en"
        ).lower()
        if not supported_languages:
            supported_languages = [default_language]
        if len(supported_languages) == 1 and supported_languages[0] == default_language:
            return

        detector = getattr(self.llm_client, "detect_language_preference", None)
        if not callable(detector):
            return

        try:
            detected = await detector(text, supported_languages, default_language)
        except Exception:  # noqa: BLE001
            logger.bind(call_id=session.call_id).warning("Language detection failed")
            return

        detected = (detected or default_language).lower()
        if detected not in supported_languages:
            return
        if detected != session.preferred_language:
            session.preferred_language = detected
            customer_details = session.state.setdefault("customer_details", {})
            customer_details["language"] = detected

    async def _render_response_for_language(self, session: CallSession, text: str) -> str:
        if not text:
            return text

        preferred_language = (session.preferred_language or "").lower()
        default_language = (
            session.agent_config.default_greeting_language
            or session.agent_config.language
            or "en"
        ).lower()
        if not preferred_language or preferred_language == default_language:
            return text

        rewriter = getattr(self.llm_client, "rewrite_confirmation", None)
        if not callable(rewriter):
            return text

        try:
            return await rewriter(text, caller_language_hint=preferred_language)
        except Exception:  # noqa: BLE001
            logger.bind(call_id=session.call_id).warning("Language rewrite failed")
            return text

    async def _refresh_session_context_if_stale(self, session: CallSession) -> None:
        now = time.monotonic()
        if now - session.context_updated_at < self.context_refresh_ttl_seconds:
            return

        refreshed_config = await self.backend_client.fetch_agent_config(session.agent_id or None)
        session.agent_config = refreshed_config
        session.agent_id = refreshed_config.agent_id
        if not session.preferred_language:
            preferred_language = refreshed_config.default_greeting_language or refreshed_config.language or "en"
            session.preferred_language = preferred_language.lower()
        session.system_prompt = prompts.build_system_prompt(refreshed_config)
        session.context_updated_at = now

        logger.bind(call_id=session.call_id).info(
            "Session context refreshed",
            agent_id=session.agent_id,
            ttl_seconds=self.context_refresh_ttl_seconds,
        )

    def _update_active_intent_from_user_text(self, session: CallSession, user_text: str) -> None:
        text = user_text.lower()
        booking_tokens = {"book", "booking", "appointment", "reserve", "schedule"}
        tracking_tokens = {"track", "status", "booking id", "reference"}
        transfer_tokens = {"transfer", "human", "agent"}

        if any(token in text for token in booking_tokens):
            session.state["active_intent"] = "booking"
            return
        if any(token in text for token in tracking_tokens):
            session.state["active_intent"] = "tracking"
            return
        if any(token in text for token in transfer_tokens):
            session.state["active_intent"] = "transfer"

    def _booking_flow_in_progress(self, session: CallSession, user_text: str) -> bool:
        if session.state.get("active_intent") == "booking" and not session.state.get("booking_completed"):
            return True

        interaction_type = session.state.get("interaction_type")
        if interaction_type == "booking" and not session.state.get("booking_completed"):
            return True

        last_assistant_message = next(
            (
                item.get("content", "")
                for item in reversed(session.conversation_history[:-1])
                if item.get("role") == "assistant" and item.get("content")
            ),
            "",
        ).lower()
        booking_slot_prompts = {
            "where are you located",
            "which area are you located",
            "what is your location",
            "what area are you in",
            "when should we visit",
            "what time works best",
            "preferred time",
            "preferred date",
            "what date works",
            "your name",
            "phone number",
            "contact number",
        }
        if any(prompt in last_assistant_message for prompt in booking_slot_prompts):
            return True

        text = user_text.lower()
        booking_answer_patterns = {
            "my location is",
            "i am in",
            "i'm in",
            "located in",
            "my area is",
            "my city is",
            "for tomorrow",
            "for today",
            "at 12",
            "at 1",
            "at 2",
            "at 3",
            "at 4",
            "at 5",
            "at 6",
            "in the morning",
            "in the evening",
        }
        return any(pattern in text for pattern in booking_answer_patterns)

    def _try_business_info_response(
        self,
        session: CallSession,
        user_text: str,
        business_context: dict[str, Any] | None = None,
    ) -> dict[str, str | None] | None:
        text = user_text.lower().strip()
        if not text:
            return None

        if self._booking_flow_in_progress(session, user_text):
            return None

        booking_tokens = {
            "book",
            "booking",
            "appointment",
            "reserve",
            "schedule",
            "track",
            "status",
            "transfer",
            "human",
            "agent",
            "bye",
            "goodbye",
        }
        if any(token in text for token in booking_tokens):
            return None

        business_info_tokens = {
            "service",
            "services",
            "provide",
            "offer",
            "price",
            "pricing",
            "cost",
            "rate",
            "quote",
            "hour",
            "open",
            "close",
            "timing",
            "payment",
            "deposit",
            "cancel",
            "cancellation",
            "refund",
            "where",
            "address",
            "location",
            "located",
            "business",
            "company",
            "website",
            "email",
        }
        if not any(token in text for token in business_info_tokens):
            return None

        result = business_context or self._get_business_context_for_turn(session, user_text)
        if self._should_route_business_query_to_llm(session, user_text, result):
            return None
        answer = (result.get("answer") or "").strip()
        if not answer:
            return None

        return {
            "action": "speak",
            "text_to_speak": answer,
            "transfer_number": None,
        }

    def _get_business_context_for_turn(self, session: CallSession, user_text: str) -> dict[str, Any] | None:
        builder = getattr(self.backend_client, "build_business_query_answer", None)
        if callable(builder):
            result = builder(session.agent_config, user_text)
        else:
            result = self._build_business_info_response(session, user_text.lower())
        return result if result and result.get("answer") else None

    def _should_route_business_query_to_llm(
        self,
        session: CallSession,
        user_text: str,
        business_context: dict[str, Any] | None,
    ) -> bool:
        if not business_context:
            return False

        text = user_text.lower()
        conversational_markers = {
            "tell me",
            "can you tell",
            "about yourself",
            "about it",
            "in detail",
            "details",
            "explain",
            "more about",
            "sorry",
            "interrupt",
            "help me understand",
        }
        if any(marker in text for marker in conversational_markers):
            return True

        if business_context.get("matched_services"):
            return True

        if session.state.get("active_intent") in {"booking", "tracking", "transfer"}:
            return True

        words = [token for token in re.findall(r"[a-zA-Z]{2,}", text)]
        terse_fact_tokens = {
            "price", "pricing", "hours", "hour", "open", "close", "timing", "services", "service",
            "address", "location", "where", "payment", "payments", "deposit", "refund",
            "cancellation", "cancel", "policy", "policies", "business", "email", "website", "and",
        }
        if len(words) <= 4 and set(words).issubset(terse_fact_tokens):
            return False

        if any(text.startswith(prefix) for prefix in ("what", "how", "can", "do", "does", "is", "are", "tell", "explain")):
            return True

        if len(words) > 5:
            return True

        return False

    def _build_business_info_response(self, session: CallSession, text: str) -> dict[str, str]:
        agent = session.agent_config
        service_lines: list[str] = []
        for service in agent.service_catalog[:6]:
            price = service.base_price or service.base_price_bhd or "pricing varies"
            description = (service.description or service.name).strip()
            service_lines.append(f"{service.name}: {description} ({price})")

        snippets: list[str] = []

        if any(token in text for token in {"service", "services", "provide", "offer", "business", "company"}):
            if service_lines:
                snippets.append(f"{agent.business_name} offers {', '.join(service_lines)}.")
            elif agent.business_description:
                snippets.append(agent.business_description)

        if any(token in text for token in {"price", "pricing", "cost", "rate", "quote"}):
            if service_lines:
                snippets.append("Pricing depends on the selected service. Current options include " + "; ".join(service_lines[:3]) + ".")

        if any(token in text for token in {"hour", "open", "close", "timing"}):
            if agent.business_hours:
                snippets.append(f"Our business hours are {agent.business_hours}.")
            else:
                snippets.append("Business hours vary by day, and our team can confirm the latest schedule for you.")

        if any(token in text for token in {"payment", "deposit"}):
            if agent.payment_policy:
                snippets.append(agent.payment_policy)
            if agent.deposit_policy:
                snippets.append(agent.deposit_policy)
            if not agent.payment_policy and not agent.deposit_policy:
                snippets.append("Payment details depend on the selected service, and we can confirm the exact policy for you.")

        if any(token in text for token in {"cancel", "cancellation", "refund"}):
            if agent.cancellation_policy:
                snippets.append(agent.cancellation_policy)
            else:
                snippets.append("Cancellation terms depend on the appointment type, and we can confirm the latest policy for you.")

        if any(token in text for token in {"where", "address", "location", "located"}):
            if agent.coverage_areas:
                snippets.append(f"We serve {', '.join(agent.coverage_areas[:5])}.")
            else:
                snippets.append("We can confirm the exact service area and address details for you.")

        for question, answer in agent.faqs.items():
            question_tokens = {part for part in question.lower().split() if len(part) > 2}
            if question_tokens and any(token in question_tokens for token in text.split()):
                snippets.append(answer)

        if not snippets:
            if agent.business_description:
                snippets.append(agent.business_description)
            elif service_lines:
                snippets.append(f"{agent.business_name} offers {', '.join(service_lines)}.")
            else:
                snippets.append(f"{agent.business_name} can help with service details, pricing, and bookings.")

        return {"answer": " ".join(snippets[:4]).strip()}

    async def _llm_turn(
        self,
        session: CallSession,
        user_text: str,
        business_context: dict[str, Any] | None = None,
    ) -> dict[str, str | None]:
        """Core LLM interaction for text responses and tool calls."""

        system_prompt = session.system_prompt
        preferred_language = session.preferred_language or session.agent_config.default_greeting_language or session.agent_config.language or "en"
        system_prompt = (
            f"{system_prompt}\n\n"
            f"CURRENT CALLER LANGUAGE: {preferred_language}.\n"
            "Reply in the caller's current language unless they explicitly ask to switch."
        )
        if business_context and business_context.get("answer"):
            factual_lines = [
                "TURN-SPECIFIC FACTS:",
                f"- Use these business facts for this reply: {business_context['answer']}",
                "- Use the facts above, but answer naturally and conversationally.",
            ]
            matched_services = business_context.get("matched_services") or []
            if matched_services:
                factual_lines.append(
                    f"- The caller is asking specifically about: {', '.join(matched_services)}. Focus on that service instead of listing the full catalog."
                )
            system_prompt = f"{system_prompt}\n\n" + "\n".join(factual_lines)

        result = await self.llm_client.conversation_turn(
            system_prompt=system_prompt,
            conversation_history=session.conversation_history[:-1],
            user_message=user_text,
        )

        response_text = result.get("response_text", "")
        tool_calls = result.get("tool_calls", [])

        if not tool_calls:
            session.state["interaction_type"] = session.state.get("interaction_type") or "query"
            return {
                "action": "speak",
                "text_to_speak": response_text or "Is there anything else I can help you with?",
                "transfer_number": None,
            }

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

            if tc["name"] == "end_call":
                final_action = "hangup"
            elif tc["name"] == "transfer_to_human":
                final_action = "transfer"
                transfer_number = tool_output.get("transfer_number")

        raw_message = result.get("raw_message")
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

        session.conversation_history.append(assistant_msg)

        for tr in tool_results:
            import json as _json
            session.conversation_history.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": _json.dumps(tr["output"]),
            })

        try:
            follow_up = await self.llm_client.continue_after_tool(
                system_prompt=system_prompt,
                conversation_history=session.conversation_history[:-len(tool_results) - 1],
                tool_call_message=assistant_msg,
                tool_results=tool_results,
            )
        except Exception:
            logger.warning("Failed to get follow-up after tools, using default")
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
        session = await self.sessions.get(call_id)
        if session:
            # Determine outcome from last action
            outcome = "completed"
            last_state = session.state
            if last_state.get("transferred"):
                outcome = "transferred"

            record = self.call_store.end_call(
                call_id=call_id,
                outcome=outcome,
                transcript=[
                    {"role": m.get("role", ""), "content": m.get("content", "")}
                    for m in session.conversation_history
                    if m.get("content")
                ],
                turn_count=session.turn_count,
            )
            if record is not None:
                record.report_payload = self._build_call_report_payload(session)

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

    async def build_call_report(self, call_id: str) -> dict[str, Any] | None:
        session = await self.sessions.get(call_id)
        if session is not None:
            return self._build_call_report_payload(session)

        record = self.call_store.get_call(call_id)
        if record is None:
            return None
        if record.report_payload:
            return record.report_payload

        transcript = list(record.transcript)
        interaction_type = self._infer_interaction_type({}, transcript)
        booking_status = self._derive_booking_status({}, interaction_type)
        final_disposition = self._derive_final_disposition({}, interaction_type, transferred=record.outcome == "transferred")
        return {
            "call_id": record.call_id,
            "agent_id": record.agent_id,
            "business_name": None,
            "summary": self._build_report_summary({}, transcript, interaction_type, business_name=None),
            "action_required": self._derive_action_required({}, interaction_type, booking_status),
            "action_type": self._derive_action_type({}, interaction_type, transferred=record.outcome == "transferred"),
            "booking_status": booking_status,
            "final_disposition": final_disposition,
            "customer_details": {
                "phone_number": record.caller_number,
            },
            "order_or_booked_service": {
                "interaction_type": interaction_type,
            },
            "call_analytics": {
                "call_id": record.call_id,
                "agent_id": record.agent_id,
                "duration_seconds": round(record.duration_seconds, 2),
                "turn_count": record.turn_count,
                "outcome": record.outcome,
                "started_at": record.started_at,
                "ended_at": record.ended_at,
                "is_test": record.is_test,
                "called_number": record.called_number,
            },
            "transcript": list(record.transcript),
        }

    def _build_call_report_payload(self, session: CallSession) -> dict[str, Any]:
        record = self.call_store.get_call(session.call_id)
        customer_details = dict(session.state.get("customer_details") or {})
        if record and record.caller_number and not customer_details.get("phone_number"):
            customer_details["phone_number"] = record.caller_number
        if session.preferred_language:
            customer_details.setdefault("language", session.preferred_language)

        order_or_booking = dict(session.state.get("order_or_booking") or {})
        interaction_type = self._infer_interaction_type(session.state, session.conversation_history)
        if session.state.get("queries"):
            order_or_booking.setdefault("queries", list(session.state.get("queries", [])))
        if session.state.get("intake_answers"):
            order_or_booking.setdefault("intake_answers", dict(session.state.get("intake_answers", {})))
        if session.state.get("last_booking_ref"):
            order_or_booking.setdefault("booking_ref", session.state.get("last_booking_ref"))
        if session.state.get("last_short_id"):
            order_or_booking.setdefault("short_booking_id", session.state.get("last_short_id"))
        order_or_booking.setdefault("interaction_type", interaction_type)

        started_at = record.started_at if record else None
        ended_at = record.ended_at if record else None
        duration_seconds = round(record.duration_seconds, 2) if record else round(session.call_duration_seconds, 2)
        caller_number = record.caller_number if record else customer_details.get("phone_number", "")
        called_number = record.called_number if record else (session.state.get("call_context") or {}).get("called_number", "")
        transferred = bool(session.state.get("transferred"))
        booking_status = self._derive_booking_status(order_or_booking, interaction_type)
        transcript = [
            {"role": item.get("role", ""), "content": item.get("content", "")}
            for item in session.conversation_history
            if item.get("content")
        ]
        final_disposition = self._derive_final_disposition(order_or_booking, interaction_type, transferred=transferred)

        return {
            "call_id": session.call_id,
            "agent_id": session.agent_id,
            "business_name": session.agent_config.business_name,
            "summary": self._build_report_summary(
                order_or_booking,
                transcript,
                interaction_type,
                business_name=session.agent_config.business_name,
            ),
            "action_required": self._derive_action_required(order_or_booking, interaction_type, booking_status),
            "action_type": self._derive_action_type(order_or_booking, interaction_type, transferred=transferred),
            "booking_status": booking_status,
            "final_disposition": final_disposition,
            "customer_details": customer_details,
            "order_or_booked_service": order_or_booking,
            "call_analytics": {
                "call_id": session.call_id,
                "agent_id": session.agent_id,
                "business_name": session.agent_config.business_name,
                "caller_number": caller_number,
                "called_number": called_number,
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_seconds": duration_seconds,
                "turn_count": session.turn_count,
                "outcome": "transferred" if transferred else "completed",
                "was_transferred": transferred,
                "transfer_number": session.agent_config.fallback_phone,
                "is_test": session.is_test,
            },
            "transcript": transcript,
        }

    def _infer_interaction_type(self, state: dict[str, Any], transcript: list[dict[str, Any]]) -> str:
        explicit = state.get("interaction_type") or (state.get("order_or_booking") or {}).get("type")
        active_intent = state.get("active_intent")
        if explicit in {"booking", "tracking", "transfer"}:
            return explicit
        if active_intent in {"booking", "tracking", "transfer"}:
            return active_intent

        joined_user_text = " ".join(
            item.get("content", "").lower()
            for item in transcript
            if item.get("role") == "user" and item.get("content")
        )
        if any(token in joined_user_text for token in {"book", "booking", "reservation", "reserve", "appointment"}):
            return "booking"
        if any(token in joined_user_text for token in {"track", "status", "booking id", "reference"}):
            return "tracking"
        if any(token in joined_user_text for token in {"human", "agent", "representative", "transfer"}):
            return "transfer"
        if explicit:
            return explicit
        return "query"

    def _derive_booking_status(self, order_or_booking: dict[str, Any], interaction_type: str) -> str | None:
        status = order_or_booking.get("status")
        if status:
            return str(status)
        if interaction_type == "booking":
            if order_or_booking.get("booking_ref") or order_or_booking.get("short_booking_id"):
                return "confirmed"
            return "pending"
        if interaction_type == "tracking":
            return "inquiry"
        return None

    def _derive_action_required(
        self,
        order_or_booking: dict[str, Any],
        interaction_type: str,
        booking_status: str | None,
    ) -> bool:
        if interaction_type == "booking":
            return booking_status not in {"confirmed", "completed", "cancelled"}
        if interaction_type in {"tracking", "transfer"}:
            return True
        return False

    def _derive_action_type(
        self,
        order_or_booking: dict[str, Any],
        interaction_type: str,
        *,
        transferred: bool,
    ) -> str | None:
        if transferred:
            return "human_transfer"
        if interaction_type == "booking":
            status = self._derive_booking_status(order_or_booking, interaction_type)
            return "booking_followup" if status != "confirmed" else "booking_completed"
        if interaction_type == "tracking":
            return "tracking_followup"
        if interaction_type == "transfer":
            return "human_transfer"
        return None

    def _derive_final_disposition(
        self,
        order_or_booking: dict[str, Any],
        interaction_type: str,
        *,
        transferred: bool,
    ) -> str:
        if transferred:
            return "transferred_to_human"
        if interaction_type == "booking":
            status = self._derive_booking_status(order_or_booking, interaction_type)
            return "booking_confirmed" if status == "confirmed" else "booking_request"
        if interaction_type == "tracking":
            return "booking_tracking"
        if interaction_type == "transfer":
            return "transfer_request"
        return "general_query"

    def _build_report_summary(
        self,
        order_or_booking: dict[str, Any],
        transcript: list[dict[str, Any]],
        interaction_type: str,
        *,
        business_name: str | None,
    ) -> str:
        service_name = (
            order_or_booking.get("service_type")
            or order_or_booking.get("service_name")
            or "a service"
        )
        if interaction_type == "booking":
            if order_or_booking.get("booking_ref") or order_or_booking.get("short_booking_id"):
                return f"Customer completed a booking for {service_name}."
            return f"Customer wants to make a reservation for {service_name}."
        if interaction_type == "tracking":
            booking_ref = order_or_booking.get("booking_ref") or "an existing booking"
            return f"Customer wants to check the status of {booking_ref}."
        if interaction_type == "transfer":
            return "Customer requested escalation to a human agent."

        last_user_message = next(
            (
                item.get("content", "").strip()
                for item in reversed(transcript)
                if item.get("role") == "user" and item.get("content")
            ),
            "",
        )
        if last_user_message:
            return f"Customer asked: {last_user_message}"
        if business_name:
            return f"Customer contacted {business_name}."
        return "Customer contacted the business."
