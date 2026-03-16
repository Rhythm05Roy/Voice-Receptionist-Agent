"""Tests for the conversation engine with LLM-driven dialogue."""
import asyncio

from src.core.conversation.engine import CallSessionManager, ConversationEngine
from src.core.types import AgentConfig


class _Backend:
    def __init__(self):
        self.business_name = "Bahrain HomeCare Group"
        self.greeting = "Hello, this is Bahrain HomeCare Group. How can I help you today?"
        self._bookings: dict[str, dict[str, str]] = {
            "DUMMY-10001": {
                "status": "scheduled",
                "booking_ref": "DUMMY-10001",
                "message": "Technician is scheduled for this booking.",
                "service_name": "AC Repair and Maintenance",
                "location": "Manama",
                "preferred_time": "6:00 PM",
            },
        }
        self._booking_counter = 10001
        self.last_booking: dict | None = None

    async def fetch_agent_config(self, agent_id=None):
        return AgentConfig(
            agent_id="default",
            business_name=self.business_name,
            greeting=self.greeting,
            language="en",
            default_greeting_language="en",
            supported_languages=["en", "ar"],
            coverage_country="Bahrain",
            coverage_areas=["Manama", "Riffa", "Muharraq"],
            fallback_phone="+97317000000",
            max_call_duration_minutes=15,
            intake_questions=[
                {
                    "key": "service_type",
                    "question": "What service do you need?",
                    "answer_type": "text",
                    "required": True,
                },
                {
                    "key": "location",
                    "question": "Where are you located?",
                    "answer_type": "text",
                    "required": True,
                },
                {
                    "key": "preferred_time",
                    "question": "When should we visit?",
                    "answer_type": "text",
                    "required": True,
                },
            ],
            service_catalog=[
                {
                    "service_id": "ac_repair",
                    "name": "AC Repair and Maintenance",
                    "description": "AC support",
                    "base_price_bhd": "10-35 BHD",
                    "keywords": ["ac", "repair"],
                },
                {
                    "service_id": "deep_cleaning",
                    "name": "Home Deep Cleaning",
                    "description": "Cleaning support",
                    "base_price_bhd": "12-45 BHD",
                    "keywords": ["cleaning", "deep clean"],
                },
            ],
            faqs={
                "services": "We provide AC repair and deep cleaning.",
                "pricing": "Pricing depends on job scope and area.",
            },
        )

    async def resolve_agent_id_for_inbound(self, to_number: str | None):
        return "default"

    async def book_service(self, agent_id: str, answers: dict) -> dict:
        self._booking_counter += 1
        ref = f"DUMMY-{self._booking_counter}"
        short_id = str(self._booking_counter)
        self.last_booking = {**answers, "ref": ref}
        self._bookings[ref] = {
            "status": "created",
            "booking_ref": ref,
            "short_booking_id": short_id,
            "message": f"Booking {ref} confirmed.",
            "service_name": answers.get("service_type", ""),
            "location": answers.get("location", ""),
            "preferred_time": answers.get("preferred_time", ""),
        }
        return self._bookings[ref]

    async def track_booking(self, booking_id: str, agent_id: str | None = None):
        # Try lookup
        result = self._bookings.get(booking_id)
        if result:
            return result
        # Search by short ID suffix
        for key, val in self._bookings.items():
            if key.endswith(booking_id):
                return val
        return {
            "status": "not_found",
            "booking_ref": booking_id,
            "message": f"No booking found for ID {booking_id}.",
        }


class _LLM:
    """Fake LLM that returns predictable responses based on user text."""

    def __init__(self):
        self.call_count = 0

    async def conversation_turn(self, system_prompt, conversation_history, user_message):
        """Simulate GPT-4o responses with function calling."""
        self.call_count += 1
        text = user_message.lower().strip()

        # Booking-related: if user mentions booking + all required fields
        if "book" in text and "cleaning" in text and "manama" in text:
            return {
                "response_text": "",
                "tool_calls": [{
                    "id": f"call_{self.call_count}",
                    "name": "submit_booking",
                    "arguments": {
                        "service_type": "Home Deep Cleaning",
                        "location": "Manama",
                        "preferred_time": "tomorrow 2 PM",
                    },
                }],
                "raw_message": _FakeRawMessage("submit_booking", {
                    "service_type": "Home Deep Cleaning",
                    "location": "Manama",
                    "preferred_time": "tomorrow 2 PM",
                }),
            }

        # Track booking
        if "track" in text or "booking" in text and "status" in text:
            import re
            booking_id = ""
            match = re.search(r"(\d{4,})", text)
            if match:
                booking_id = f"DUMMY-{match.group(1)}"
            return {
                "response_text": "",
                "tool_calls": [{
                    "id": f"call_{self.call_count}",
                    "name": "track_booking",
                    "arguments": {"booking_id": booking_id or "DUMMY-10001"},
                }],
                "raw_message": _FakeRawMessage("track_booking", {"booking_id": booking_id or "DUMMY-10001"}),
            }

        # End call
        if any(w in text for w in ["bye", "goodbye", "end call"]):
            return {
                "response_text": "",
                "tool_calls": [{
                    "id": f"call_{self.call_count}",
                    "name": "end_call",
                    "arguments": {"farewell_message": "Thank you for calling. Goodbye!"},
                }],
                "raw_message": _FakeRawMessage("end_call", {"farewell_message": "Thank you for calling. Goodbye!"}),
            }

        # Transfer
        if "transfer" in text or "human" in text or "frustrated" in text:
            return {
                "response_text": "",
                "tool_calls": [{
                    "id": f"call_{self.call_count}",
                    "name": "transfer_to_human",
                    "arguments": {"reason": "Caller requested human agent"},
                }],
                "raw_message": _FakeRawMessage("transfer_to_human", {"reason": "Caller requested human agent"}),
            }

        # General conversation — no tool calls
        return {
            "response_text": f"I understand you said: {user_message}. How can I help?",
            "tool_calls": [],
            "raw_message": None,
        }

    async def continue_after_tool(self, system_prompt, conversation_history, tool_call_message, tool_results):
        """Simulate follow-up after tool execution."""
        for result in tool_results:
            output = result.get("output", {})
            status = output.get("status", "")
            if status == "confirmed":
                ref = output.get("booking_ref", "")
                return f"Your booking {ref} is confirmed. Is there anything else?"
            if status == "transferring":
                return "I'm transferring you to a human agent now."
            if status == "ended":
                return output.get("farewell_message", "Goodbye!")
            if status in ("scheduled", "created"):
                ref = output.get("booking_ref", "")
                return f"Booking {ref} is {status}. {output.get('message', '')}"
            if status == "not_found":
                return f"I couldn't find that booking. {output.get('message', '')}"
            if status == "out_of_coverage":
                return output.get("message", "That location is outside our service area.")
        return "Done. Is there anything else I can help with?"

    async def rewrite_confirmation(self, text: str, caller_language_hint: str = "en") -> str:
        return text


class _FakeRawMessage:
    def __init__(self, name: str, args: dict):
        self.tool_calls = [_FakeToolCall(name, args)]
        self.content = None


class _FakeToolCall:
    def __init__(self, name: str, args: dict):
        import json
        self.id = f"call_fake_{name}"
        self.function = _FakeFunction(name, json.dumps(args))


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _TTS:
    async def synthesize_text(self, text: str, voice_id: str | None = None) -> str:
        return "data:audio/mpeg;base64,AAA"


def _make_engine() -> tuple[ConversationEngine, _Backend, _LLM]:
    backend = _Backend()
    llm = _LLM()
    tts = _TTS()
    session_mgr = CallSessionManager()
    engine = ConversationEngine(
        backend_client=backend,
        llm_client=llm,
        tts_client=tts,
        environment="test",
        session_manager=session_mgr,
        context_refresh_ttl_seconds=60,
    )
    return engine, backend, llm


# ── Test: start session returns greeting ──────────────────────────

def test_start_session_returns_greeting():
    engine, _, _ = _make_engine()
    result = asyncio.run(engine.start_session("call-001"))
    assert "Bahrain HomeCare Group" in result["text"]
    assert result["audio_url"]


# ── Test: general conversation returns LLM text ──────────────────

def test_general_conversation():
    engine, _, _ = _make_engine()
    asyncio.run(engine.start_session("call-002"))
    result = asyncio.run(engine.process_user_input("call-002", "What services do you offer?"))
    assert result["action"] == "speak"
    assert result["text_to_speak"]  # Should have a response
    assert "AC Repair and Maintenance" in result["text_to_speak"]
    assert "Home Deep Cleaning" in result["text_to_speak"]


def test_business_info_query_uses_deterministic_context():
    engine, _, llm = _make_engine()
    asyncio.run(engine.start_session("call-002b"))
    result = asyncio.run(engine.process_user_input("call-002b", "Tell me about your pricing and business hours"))

    assert result["action"] == "speak"
    assert "Pricing depends" in result["text_to_speak"] or "Current pricing includes" in result["text_to_speak"]
    assert "hours" in result["text_to_speak"].lower()
    assert llm.call_count == 0


# ── Test: booking triggers submit_booking tool ───────────────────

def test_booking_flow():
    engine, backend, _ = _make_engine()
    asyncio.run(engine.start_session("call-003"))
    result = asyncio.run(engine.process_user_input(
        "call-003",
        "I want to book cleaning in Manama",
    ))
    # LLM should have called submit_booking, engine should have result
    assert result["text_to_speak"]
    assert backend.last_booking is not None
    assert "DUMMY-" in result["text_to_speak"]


# ── Test: track booking ──────────────────────────────────────────

def test_track_booking():
    engine, _, _ = _make_engine()
    asyncio.run(engine.start_session("call-004"))
    result = asyncio.run(engine.process_user_input(
        "call-004",
        "Can I track my booking 10001?",
    ))
    assert result["text_to_speak"]
    assert "scheduled" in result["text_to_speak"].lower() or "DUMMY" in result["text_to_speak"]


# ── Test: farewell triggers hangup ───────────────────────────────

def test_farewell_ends_call():
    engine, _, _ = _make_engine()
    asyncio.run(engine.start_session("call-005"))
    result = asyncio.run(engine.process_user_input("call-005", "Bye, thanks!"))
    assert result["action"] == "hangup"
    assert result["text_to_speak"]


# ── Test: transfer request ───────────────────────────────────────

def test_transfer_request():
    engine, _, _ = _make_engine()
    asyncio.run(engine.start_session("call-006"))
    result = asyncio.run(engine.process_user_input(
        "call-006",
        "I'm frustrated, transfer me to a human",
    ))
    assert result["action"] == "transfer"
    assert result["transfer_number"]


# ── Test: conversation history persists across turns ─────────────

def test_conversation_history_persists():
    engine, _, _ = _make_engine()
    asyncio.run(engine.start_session("call-007"))
    asyncio.run(engine.process_user_input("call-007", "I need AC repair"))
    asyncio.run(engine.process_user_input("call-007", "In Manama"))
    # Session should have history of all turns
    session = asyncio.run(engine.sessions.get("call-007"))
    assert session is not None
    assert session.turn_count == 2
    assert len(session.conversation_history) >= 4  # greeting + user + response + user


# ── Test: session end clears data ────────────────────────────────

def test_end_call_clears_session():
    engine, _, _ = _make_engine()
    asyncio.run(engine.start_session("call-008"))
    asyncio.run(engine.end_call("call-008"))
    assert not asyncio.run(engine.has_session("call-008"))


def test_session_context_refreshes_after_ttl():
    engine, backend, _ = _make_engine()
    engine.context_refresh_ttl_seconds = 1

    asyncio.run(engine.start_session("call-009"))
    session = asyncio.run(engine.sessions.get("call-009"))
    assert session is not None
    assert session.agent_config.business_name == "Bahrain HomeCare Group"

    backend.business_name = "Urban Glow Salon"
    backend.greeting = "Hello, thank you for calling Urban Glow Salon. How can I help you today?"
    session.context_updated_at -= 5
    asyncio.run(engine.sessions.save(session))

    asyncio.run(engine.process_user_input("call-009", "What services do you offer?"))

    refreshed = asyncio.run(engine.sessions.get("call-009"))
    assert refreshed is not None
    assert refreshed.agent_config.business_name == "Urban Glow Salon"
    assert "Urban Glow Salon" in refreshed.system_prompt
