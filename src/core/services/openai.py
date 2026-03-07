"""OpenAI LLM client — function-calling & conversation-history aware.

This is the 'brain' of the voice agent.  Instead of rigid per-question
classification, the LLM receives the **full conversation history** and
a set of **tools** (book, track, transfer, end_call) so it can:

* Gather booking details naturally across turns.
* Handle answer corrections ("change my location to X").
* Answer side-questions without losing context.
* Detect frustration and offer human transfer.
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator, Sequence

from loguru import logger
from openai import AsyncOpenAI, OpenAIError

from src.api.exceptions import ConversationEngineError


# ── Tool / Function schemas ──────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "submit_booking",
            "description": (
                "Submit a booking request. Call ONLY when ALL required fields "
                "have been naturally collected and confirmed with the caller. "
                "Required: service_type, location, preferred_date, preferred_time. "
                "Also collect customer_name, customer_phone, and any service-specific details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service_type": {
                        "type": "string",
                        "description": "The service the caller wants (must match an available service).",
                    },
                    "location": {
                        "type": "string",
                        "description": "Caller's area / city for the service visit.",
                    },
                    "preferred_date": {
                        "type": "string",
                        "description": "Date for the visit (e.g. 'tomorrow', 'March 10', 'next Monday').",
                    },
                    "preferred_time": {
                        "type": "string",
                        "description": "Preferred time window (e.g. '2 PM - 4 PM', 'morning', '12pm to 2pm').",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Caller's full name.",
                    },
                    "customer_phone": {
                        "type": "string",
                        "description": "Caller's phone number for confirmation callbacks.",
                    },
                    "property_type": {
                        "type": "string",
                        "enum": ["apartment", "villa", "office", "shop", "other"],
                        "description": "Type of property (for cleaning, maintenance, AC services).",
                    },
                    "num_rooms": {
                        "type": "integer",
                        "description": "Number of rooms (for cleaning/maintenance services).",
                    },
                    "specific_areas": {
                        "type": "string",
                        "description": "Specific areas to focus on (e.g. 'kitchen and bathrooms', 'living room AC unit').",
                    },
                    "allergy_info": {
                        "type": "string",
                        "description": "Allergies or chemical sensitivities to consider (e.g. 'dust allergy', 'no bleach').",
                    },
                    "issue_description": {
                        "type": "string",
                        "description": "Detailed description of issue for repair/maintenance (e.g. 'AC not cooling', 'leaking faucet').",
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["normal", "urgent", "emergency"],
                        "description": "Urgency level. Use 'urgent' if same-day, 'emergency' for immediate needs.",
                    },
                    "special_instructions": {
                        "type": "string",
                        "description": "Any special instructions (e.g. 'ring doorbell twice', 'call on arrival', 'bring eco-friendly products').",
                    },
                },
                "required": ["service_type", "location", "preferred_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_booking",
            "description": "Look up the status of an existing booking by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "The booking reference ID.",
                    },
                },
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": (
                "Transfer the caller to a human agent.  Use when: "
                "caller explicitly asks, or shows clear frustration, "
                "or the request is outside your capabilities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for the transfer.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the call politely when the caller says goodbye or has no more needs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "farewell_message": {
                        "type": "string",
                        "description": "A polite goodbye message.",
                    },
                },
                "required": ["farewell_message"],
            },
        },
    },
]


class OpenAIClient:
    """Conversation-aware LLM client with function-calling support."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.fast_model = "gpt-4o-mini"  # ~60% faster for simple turns

    def _select_model(self, needs_tools: bool = True) -> str:
        """Smart model routing — fast model for simple turns, full model for tool calls."""
        return self.model if needs_tools else self.fast_model

    # ── Primary conversation method ──────────────────────────────

    async def conversation_turn(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, Any]],
        user_message: str,
    ) -> dict[str, Any]:
        """Process one conversation turn with full context.

        Returns dict with keys:
          - response_text: str  (what to say to the caller)
          - tool_calls: list[dict]  (any function calls to execute)
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history (already role-tagged)
        for turn in conversation_history:
            messages.append(turn)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.6,
                max_tokens=400,
            )

            choice = completion.choices[0]
            response_text = (choice.message.content or "").strip()
            tool_calls: list[dict[str, Any]] = []

            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": args,
                    })

            logger.debug(
                "LLM turn completed",
                model=self.model,
                has_response=bool(response_text),
                tool_count=len(tool_calls),
                usage=getattr(completion, "usage", None),
            )

            return {
                "response_text": response_text,
                "tool_calls": tool_calls,
                "raw_message": choice.message,
            }

        except OpenAIError as exc:
            logger.exception("OpenAI conversation_turn failed")
            raise ConversationEngineError(str(exc)) from exc

    # ── Streaming variant for low-latency TTS ────────────────────

    async def conversation_turn_stream(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, Any]],
        user_message: str,
    ) -> AsyncIterator[str]:
        """Stream text tokens for immediate TTS forwarding.

        Note: streaming does not support tool calls in the same turn.
        Use conversation_turn() for turns that may require tool calls,
        and this method for pure conversational responses.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        for turn in conversation_history:
            messages.append(turn)
        messages.append({"role": "user", "content": user_message})

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
                max_tokens=400,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

        except OpenAIError as exc:
            logger.exception("OpenAI streaming failed")
            raise ConversationEngineError(str(exc)) from exc

    # ── Continue conversation after tool results ─────────────────

    async def continue_after_tool(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, Any]],
        tool_call_message: Any,
        tool_results: list[dict[str, Any]],
    ) -> str:
        """Send tool results back to GPT and get the follow-up response."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        for turn in conversation_history:
            messages.append(turn)

        # Add the assistant message that contained the tool call
        messages.append(tool_call_message)

        # Add each tool result
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": json.dumps(result["output"], ensure_ascii=False),
            })

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
                max_tokens=400,
            )
            return (completion.choices[0].message.content or "").strip()

        except OpenAIError as exc:
            logger.exception("OpenAI continue_after_tool failed")
            raise ConversationEngineError(str(exc)) from exc

    # ── Simple generation (for greeting rewriting, etc.) ─────────

    async def generate_reply(self, messages: Sequence[dict[str, str]]) -> str:
        """Simple one-shot generation without tools."""
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                temperature=0.55,
                max_tokens=320,
            )
            reply = completion.choices[0].message.content or ""
            return reply.strip()
        except OpenAIError as exc:
            logger.exception("OpenAI call failed")
            raise ConversationEngineError(str(exc)) from exc

    # ── Language detection ───────────────────────────────────────

    async def detect_language_preference(
        self,
        user_input: str,
        supported_languages: list[str],
        default_language: str,
    ) -> str:
        """Detect caller's language preference."""
        prompt = (
            "Return strict JSON with key language. Choose only one code from supported_languages. "
            "Detect language from user text and explicit requests like 'speak in English'."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "text": user_input,
                        "supported_languages": supported_languages,
                        "default_language": default_language,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=50,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            detected = str(parsed.get("language", default_language)).strip().lower()
            if detected in supported_languages:
                return detected
        except Exception:  # noqa: BLE001
            pass

        # Heuristic fallbacks
        if re.search(r"\benglish\b", user_input.lower()):
            return "en" if "en" in supported_languages else default_language
        if re.search(r"\barabic\b", user_input.lower()) and "ar" in supported_languages:
            return "ar"
        if any("\u0600" <= ch <= "\u06FF" for ch in user_input) and "ar" in supported_languages:
            return "ar"
        return default_language

    async def rewrite_confirmation(self, text: str, caller_language_hint: str = "en") -> str:
        """Rewrite a confirmation message for a natural phone call."""
        prompt = (
            "Rewrite this confirmation for phone call speech. Keep it short, clear, and human. "
            "Do not add any facts."
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({"language": caller_language_hint, "text": text}, ensure_ascii=False)},
        ]
        try:
            return await self.generate_reply(messages)
        except Exception:  # noqa: BLE001
            return text
