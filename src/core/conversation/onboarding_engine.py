"""Voice-based onboarding engine — SOW AI assistant.

Collects business information through natural conversation:
- Business name, type, description
- Services offered
- Business hours
- Booking system preference

Uses GPT function calling to extract structured data from the
conversation and build an AgentConfig.
"""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from src.core.services.openai import OpenAIClient


# ── Onboarding-specific tools ────────────────────────────────────

ONBOARDING_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "save_business_info",
            "description": (
                "Save the core business information. Call when the user has "
                "provided their business name, type, and/or description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "business_name": {"type": "string", "description": "Name of the business"},
                    "business_type": {
                        "type": "string",
                        "enum": ["salon_spa", "medical_practice", "home_services", "restaurant", "fitness", "other"],
                        "description": "Category of the business",
                    },
                    "business_description": {"type": "string", "description": "Brief description of the business"},
                },
                "required": ["business_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_services",
            "description": "Save the services the business offers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "services": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "price": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                        "description": "List of services offered",
                    },
                },
                "required": ["services"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_business_hours",
            "description": "Save the business operating hours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {"type": "string", "description": "Business hours as described by the user"},
                },
                "required": ["hours"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_booking_system",
            "description": "Save the booking/appointment system the business uses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "system": {
                        "type": "string",
                        "enum": ["mindbody", "calendly", "google_calendar", "square", "vagaro", "other", "none"],
                        "description": "The booking system used",
                    },
                    "notes": {"type": "string", "description": "Additional notes about booking handling"},
                },
                "required": ["system"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_onboarding",
            "description": (
                "Call when ALL business information has been collected and confirmed. "
                "This creates the AI agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmation_message": {"type": "string", "description": "Message confirming setup is complete"},
                },
                "required": ["confirmation_message"],
            },
        },
    },
]


ONBOARDING_SYSTEM_PROMPT = """\
You are SOW, a friendly AI onboarding assistant. You help business owners set up \
their AI phone receptionist through a natural conversation.

Your job is to collect:
1. Business name and type (salon/spa, medical, home services, restaurant, fitness, other)
2. A brief description of the business
3. Main services offered (1-5 services)
4. Business hours
5. Which booking system they use (MindBody, Calendly, Google Calendar, Square, Vagaro, other/none)

RULES:
- Be conversational and friendly. Don't interrogate.
- Ask ONE thing at a time.
- If the user provides multiple pieces of info at once, extract ALL of them.
- After each answer, call the appropriate save_* tool to store the data.
- Keep track of what's been collected. Don't re-ask for info already saved.
- When everything is collected, summarize what you have and ask for confirmation.
- After confirmation, call finalize_onboarding.
- Keep responses SHORT — this may be spoken aloud.
- If the user seems confused, explain what you need simply.

Start by introducing yourself and asking for the business name.\
"""


class OnboardingSession(BaseModel):
    session_id: str
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    collected_data: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.monotonic)
    is_complete: bool = False
    turn_count: int = 0

    def add_message(self, role: str, content: str) -> None:
        if content:
            self.conversation_history.append({"role": role, "content": content})
            if len(self.conversation_history) > 30:
                self.conversation_history = self.conversation_history[-30:]
        if role == "user":
            self.turn_count += 1


class OnboardingEngine:
    """Manages voice/text onboarding sessions."""

    def __init__(self, llm_client: OpenAIClient):
        self.llm = llm_client
        self._sessions: dict[str, OnboardingSession] = {}

    async def start_session(self, session_id: str) -> dict[str, str]:
        """Start a new onboarding session and return the greeting."""
        session = OnboardingSession(session_id=session_id)

        greeting = (
            "Hi! I'm SOW, your AI setup assistant. I'll help you create your "
            "AI phone receptionist in just a few minutes. Let's start — what's "
            "the name of your business?"
        )
        session.add_message("assistant", greeting)
        self._sessions[session_id] = session

        return {"text": greeting, "step": "business_name", "is_complete": False}

    async def process_turn(self, session_id: str, user_input: str) -> dict[str, Any]:
        """Process one turn of the onboarding conversation."""
        session = self._sessions.get(session_id)
        if not session:
            result = await self.start_session(session_id)
            session = self._sessions[session_id]
            if not user_input.strip():
                return result

        session.add_message("user", user_input)

        try:
            result = await self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": ONBOARDING_SYSTEM_PROMPT},
                    *session.conversation_history,
                ],
                tools=ONBOARDING_TOOLS,
                tool_choice="auto",
                temperature=0.6,
                max_tokens=300,
            )

            choice = result.choices[0]
            response_text = (choice.message.content or "").strip()
            tool_calls = choice.message.tool_calls or []

            # Process tool calls
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                tool_name = tc.function.name
                if tool_name == "save_business_info":
                    session.collected_data.update({
                        k: v for k, v in args.items() if v
                    })
                elif tool_name == "save_services":
                    session.collected_data["services"] = args.get("services", [])
                elif tool_name == "save_business_hours":
                    session.collected_data["business_hours"] = args.get("hours", "")
                elif tool_name == "save_booking_system":
                    session.collected_data["booking_system"] = args.get("system", "none")
                    if args.get("notes"):
                        session.collected_data["booking_notes"] = args["notes"]
                elif tool_name == "finalize_onboarding":
                    session.is_complete = True

            # If tool calls, get follow-up response
            if tool_calls and not response_text:
                tool_results_msgs: list[dict[str, Any]] = [
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in tool_calls
                    ]},
                ]
                for tc in tool_calls:
                    tool_results_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"status": "saved"}),
                    })

                follow_up = await self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=[
                        {"role": "system", "content": ONBOARDING_SYSTEM_PROMPT},
                        *session.conversation_history,
                        *tool_results_msgs,
                    ],
                    temperature=0.6,
                    max_tokens=300,
                )
                response_text = (follow_up.choices[0].message.content or "").strip()

            if response_text:
                session.add_message("assistant", response_text)

            # Determine current step
            step = self._get_current_step(session)

            return {
                "text": response_text or "Got it! What else can you tell me?",
                "step": step,
                "is_complete": session.is_complete,
                "collected_data": session.collected_data,
            }

        except Exception:
            logger.exception("Onboarding turn failed", session_id=session_id)
            fallback = "Sorry, I had a brief hiccup. Could you repeat that?"
            session.add_message("assistant", fallback)
            return {
                "text": fallback,
                "step": self._get_current_step(session),
                "is_complete": False,
                "collected_data": session.collected_data,
            }

    def get_collected_data(self, session_id: str) -> dict[str, Any]:
        """Return all collected data for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return {}
        return session.collected_data

    def build_agent_config(self, session_id: str) -> dict[str, Any]:
        """Convert collected onboarding data into an AgentConfig-compatible dict."""
        data = self.get_collected_data(session_id)
        services = data.get("services", [])
        service_catalog = []
        for i, svc in enumerate(services):
            service_catalog.append({
                "service_id": f"svc_{i}",
                "name": svc.get("name", ""),
                "description": svc.get("description", ""),
                "base_price": svc.get("price", ""),
            })

        return {
            "business_name": data.get("business_name", ""),
            "business_type": data.get("business_type", "other"),
            "business_description": data.get("business_description", ""),
            "business_hours": data.get("business_hours", ""),
            "service_catalog": service_catalog,
            "booking_system": data.get("booking_system", "none"),
            "language": "en",
            "default_greeting_language": "en",
            "supported_languages": ["en"],
            "max_call_duration_minutes": 15,
        }

    def end_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _get_current_step(self, session: OnboardingSession) -> str:
        d = session.collected_data
        if not d.get("business_name"):
            return "business_name"
        if not d.get("business_type"):
            return "business_type"
        if not d.get("services"):
            return "services"
        if not d.get("business_hours"):
            return "business_hours"
        if not d.get("booking_system"):
            return "booking_system"
        if not session.is_complete:
            return "confirmation"
        return "complete"
