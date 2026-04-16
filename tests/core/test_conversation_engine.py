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
        self.last_posted_report: dict | None = None

    async def fetch_agent_config(self, agent_id=None):
        return AgentConfig(
            agent_id="default",
            business_name=self.business_name,
            business_category="home services",
            greeting=self.greeting,
            language="en",
            default_greeting_language="en",
            supported_languages=["en", "ar"],
            selected_voice_id=14,
            selected_language_id=13,
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
                {
                    "service_id": "salon_home",
                    "name": "At-Home Salon",
                    "description": "Haircut, beard trim, and basic grooming at customer location.",
                    "base_price_bhd": "8-25 BHD",
                    "keywords": ["salon", "haircut", "grooming"],
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

    async def post_call_report(self, payload: dict[str, object]) -> dict[str, object]:
        self.last_posted_report = payload
        return {
            "detail": "Call event processed successfully.",
            "external_call_id": payload.get("call_id"),
            "status": "completed",
        }


class _LLM:
    """Fake LLM that returns predictable responses based on user text."""

    def __init__(self):
        self.call_count = 0

    async def conversation_turn(self, system_prompt, conversation_history, user_message):
        """Simulate GPT-4o responses with function calling."""
        self.call_count += 1
        text = user_message.lower().strip()

        if "about yourself" in text or "tell about yourself" in text:
            return {
                "response_text": "We are Bahrain HomeCare Group, and we help with AC repair, deep cleaning, at-home salon, and general maintenance across Bahrain.",
                "tool_calls": [],
                "raw_message": None,
            }

        if "at home salon" in text and any(token in text for token in ["about", "detail", "details"]):
            return {
                "response_text": "Our at-home salon service includes haircut, beard trim, and basic grooming at your location. Pricing starts from 8-25 BHD depending on the requested service.",
                "tool_calls": [],
                "raw_message": None,
            }

        if "book" in text and any(token in text for token in ["haircut", "salon"]):
            return {
                "response_text": "Great! Where are you located?",
                "tool_calls": [],
                "raw_message": None,
            }

        if "riffa" in text:
            return {
                "response_text": "Thanks. What time works best for your appointment?",
                "tool_calls": [],
                "raw_message": None,
            }

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

        if "TURN-SPECIFIC FACTS:" in system_prompt:
            marker = "- Use these business facts for this reply:"
            if marker in system_prompt:
                facts = system_prompt.split(marker, 1)[1].split("\n", 1)[0].strip()
                return {
                    "response_text": facts,
                    "tool_calls": [],
                    "raw_message": None,
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

    async def detect_language_preference(self, user_input: str, supported_languages: list[str], default_language: str) -> str:
        lowered = user_input.lower()
        if "arabic" in lowered or any("\u0600" <= ch <= "\u06FF" for ch in user_input):
            return "ar" if "ar" in supported_languages else default_language
        return "en" if "en" in supported_languages else default_language

    async def rewrite_confirmation(self, text: str, caller_language_hint: str = "en") -> str:
        if caller_language_hint == "ar":
            return f"[ar] {text}"
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
    result = asyncio.run(engine.process_user_input("call-002b", "pricing and business hours"))

    assert result["action"] == "speak"
    assert "Pricing depends" in result["text_to_speak"] or "Current pricing includes" in result["text_to_speak"]
    assert "hours" in result["text_to_speak"].lower()
    assert llm.call_count == 0


def test_service_specific_business_query_uses_llm_with_context():
    engine, _, llm = _make_engine()
    asyncio.run(engine.start_session("call-002bb"))
    result = asyncio.run(engine.process_user_input("call-002bb", "I want to know about your at home salon service in detail"))

    assert result["action"] == "speak"
    assert "at-home salon service" in result["text_to_speak"].lower()
    assert "ac repair" not in result["text_to_speak"].lower()
    assert llm.call_count == 1


def test_about_yourself_query_uses_llm_not_canned_fast_path():
    engine, _, llm = _make_engine()
    asyncio.run(engine.start_session("call-002bc"))
    result = asyncio.run(engine.process_user_input("call-002bc", "Can you tell about yourself?"))

    assert result["action"] == "speak"
    assert "bahrain homecare group" in result["text_to_speak"].lower()
    assert llm.call_count == 1


def test_booking_follow_up_answer_is_not_hijacked_by_business_info_fast_path():
    engine, _, llm = _make_engine()
    asyncio.run(engine.start_session("call-002c"))

    first_turn = asyncio.run(engine.process_user_input("call-002c", "I want to book a haircut at my place"))
    second_turn = asyncio.run(engine.process_user_input("call-002c", "My location is Bahrain and Riffa city"))

    assert "where are you located" in first_turn["text_to_speak"].lower()
    assert "what time works best" in second_turn["text_to_speak"].lower()
    assert "we are located at or serve" not in second_turn["text_to_speak"].lower()
    assert llm.call_count == 2


def test_language_preference_updates_from_supported_language_route_data():
    engine, _, _ = _make_engine()
    asyncio.run(engine.start_session("call-002d"))

    result = asyncio.run(engine.process_user_input("call-002d", "Please speak in Arabic"))
    session = asyncio.run(engine.sessions.get("call-002d"))

    assert session is not None
    assert session.preferred_language == "ar"
    assert result["text_to_speak"].startswith("[ar] ")


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
    engine, backend, _ = _make_engine()
    asyncio.run(engine.start_session("call-008"))
    asyncio.run(engine.end_call("call-008"))
    assert not asyncio.run(engine.has_session("call-008"))
    assert backend.last_posted_report is not None
    assert backend.last_posted_report["call_id"] == "call-008"


def test_call_report_contains_booking_details():
    engine, _, _ = _make_engine()
    asyncio.run(engine.start_session("call-008b", caller_number="+14165550101", called_number="+15145550111"))
    asyncio.run(engine.process_user_input("call-008b", "I want to book cleaning in Manama"))
    report = asyncio.run(engine.build_call_report("call-008b"))

    assert report is not None
    assert report["voice_id"] == 14
    assert report["language_id"] == 13
    assert report["call_analytics"]["business_category"] == "home services"
    assert report["customer_details"]["phone_number"] == "+14165550101"
    assert report["order_or_booked_service"]["interaction_type"] == "booking"
    assert report["order_or_booked_service"]["service_type"] == "Home Deep Cleaning"
    assert report["call_analytics"]["called_number"] == "+15145550111"


def test_call_report_keeps_incremental_customer_and_booking_details_before_submit():
    engine, _, _ = _make_engine()
    asyncio.run(engine.start_session("call-008c"))
    asyncio.run(engine.process_user_input("call-008c", "I want to book a haircut at my place"))
    asyncio.run(engine.process_user_input("call-008c", "My location is Bahrain and Riffa city"))
    asyncio.run(engine.process_user_input("call-008c", "15th April 2026"))
    asyncio.run(engine.process_user_input("call-008c", "9:30 PM"))
    asyncio.run(engine.process_user_input("call-008c", "My name is Mr. Copper"))
    asyncio.run(engine.process_user_input("call-008c", "My number is 01756552585"))

    report = asyncio.run(engine.build_call_report("call-008c"))

    assert report is not None
    assert report["customer_details"]["name"] == "Mr. Copper"
    assert report["customer_details"]["phone_number"] == "01756552585"
    assert report["customer_details"]["location"] == "Bahrain and Riffa city"
    assert report["order_or_booked_service"]["interaction_type"] == "booking"
    assert report["order_or_booked_service"]["service_type"] == "At-Home Salon"
    assert report["order_or_booked_service"]["preferred_date"] == "15th April 2026"
    assert report["order_or_booked_service"]["preferred_time"] == "9:30 PM"


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
