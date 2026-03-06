import asyncio

from src.core.conversation.engine import CallSessionManager, ConversationEngine
from src.core.types import AgentConfig


class _Backend:
    def __init__(self):
        self._bookings: dict[str, dict[str, str]] = {
            "DUMMY-10001": {
                "status": "scheduled",
                "booking_ref": "DUMMY-10001",
                "message": "Your booking is scheduled.",
                "service_name": "AC Repair and Maintenance",
                "location": "Manama",
                "preferred_time": "6:00 PM",
            },
            "12614": {
                "status": "scheduled",
                "booking_ref": "12614",
                "message": "Your booking is scheduled.",
                "service_name": "AC Repair and Maintenance",
                "location": "Manama",
                "preferred_time": "6:00 PM",
            },
        }

    async def fetch_agent_config(self, agent_id=None):
        return AgentConfig(
            agent_id="default",
            greeting="Hello, this is Bahrain HomeCare Group. How can I help you today?",
            language="en",
            default_greeting_language="en",
            supported_languages=["en", "ar"],
            coverage_country="Bahrain",
            coverage_areas=["Manama", "Riffa", "Muharraq"],
            fallback_phone="+97317000000",
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

    async def answer_business_query(self, query: str, agent_id: str | None = None):
        return {
            "answer": "We provide AC repair and deep cleaning in Bahrain.",
            "suggested_services": ["AC Repair and Maintenance", "Home Deep Cleaning"],
        }

    async def book_service(self, agent_id: str, answers: dict[str, str]):
        payload = {
            "status": "mock_confirmed",
            "booking_ref": "DUMMY-9001",
            "short_booking_id": "9001",
            "message": (
                f"Booking DUMMY-9001 is created for {answers.get('service_type', 'service')} "
                f"in {answers.get('location', 'Bahrain')} at {answers.get('preferred_time', 'soon')}."
            ),
            "service_name": answers.get("service_type", "AC Repair and Maintenance"),
            "location": answers.get("location", "Bahrain"),
            "preferred_time": answers.get("preferred_time", "soon"),
        }
        self._bookings["DUMMY-9001"] = payload
        self._bookings["9001"] = payload
        return {
            "status": payload["status"],
            "booking_ref": payload["booking_ref"],
            "short_booking_id": payload["short_booking_id"],
            "message": payload["message"],
        }

    async def track_booking(self, booking_id: str, agent_id: str | None = None):
        key = booking_id.upper() if booking_id.upper().startswith("DUMMY-") else booking_id
        if key in self._bookings:
            return self._bookings[key]
        return {
            "status": "not_found",
            "booking_ref": booking_id,
            "message": "Booking ID was not found.",
        }


class _LLM:
    async def generate_reply(self, messages):
        if messages:
            last = messages[-1].get("content", "")
            marker = "question="
            if marker in last:
                return last.split(marker, 1)[1].split("\n", 1)[0].strip()
        return "Could you confirm if you need a booking, tracking, or service details?"

    async def detect_language_preference(self, user_input, supported_languages, default_language):
        return default_language

    async def detect_call_intent(self, user_input: str, context=None):
        text = user_input.lower()
        if "booking id" in text or "track" in text:
            return {"intent": "track_booking", "booking_id": "", "confidence": "high"}
        if "service" in text or "provide" in text or "pricing" in text:
            return {"intent": "business_info", "booking_id": "", "confidence": "medium"}
        if "book" in text or "need" in text or "clean" in text or "ac" in text:
            return {"intent": "new_booking", "booking_id": "", "confidence": "medium"}
        if "bye" in text:
            return {"intent": "end_call", "booking_id": "", "confidence": "high"}
        return {"intent": "unclear", "booking_id": "", "confidence": "low"}

    async def analyze_turn(self, question, user_input, collected_answers, question_meta=None):
        return {
            "intent": "answer",
            "extracted_answer": user_input,
            "normalized_answer": user_input,
            "assistant_reply": "",
        }

    async def rewrite_confirmation(self, text: str, caller_language_hint: str = "en"):
        return text


class _TTS:
    async def synthesize_text(self, text, voice_id=None):
        return "data:audio/mpeg;base64,AAA"


class _WrongIntentLLM(_LLM):
    async def detect_call_intent(self, user_input: str, context=None):
        text = user_input.lower()
        if "service" in text or "provide" in text:
            # Simulate model misclassification observed in local runs.
            return {"intent": "track_booking", "booking_id": "", "confidence": "low"}
        return await super().detect_call_intent(user_input, context)


class _HallucinatedBookingIdLLM(_LLM):
    async def detect_call_intent(self, user_input: str, context=None):
        if "service" in user_input.lower():
            return {"intent": "track_booking", "booking_id": "DUMMY-99999", "confidence": "low"}
        return await super().detect_call_intent(user_input, context)


class _LowConfidenceBookingLLM(_LLM):
    async def detect_call_intent(self, user_input: str, context=None):
        return {"intent": "new_booking", "booking_id": "", "confidence": "low"}


def run(coro):
    return asyncio.run(coro)


def test_intent_business_info_then_new_booking_path():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-info-book"))

    info_turn = run(engine.process_user_input("call-info-book", "What services do you provide?"))
    assert info_turn["action"] == "speak"
    assert "provide" in info_turn["text_to_speak"].lower()

    booking_turn = run(engine.process_user_input("call-info-book", "I want to book deep cleaning"))
    assert booking_turn["action"] == "speak"
    assert "where" in booking_turn["text_to_speak"].lower()


def test_full_booking_flow_records_booking_and_stays_assistive():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-booking"))
    step1 = run(engine.process_user_input("call-booking", "I need AC repair"))
    step2 = run(engine.process_user_input("call-booking", "Manama"))
    step3 = run(engine.process_user_input("call-booking", "6 PM"))

    assert step1["action"] == "speak"
    assert step2["action"] == "speak"
    assert step3["action"] == "speak"
    assert "booking dummy-9001" in step3["text_to_speak"].lower()

    follow_up = run(engine.process_user_input("call-booking", "What is your pricing?"))
    assert follow_up["action"] == "speak"
    assert "would you like to make a booking now" in follow_up["text_to_speak"].lower()


def test_track_booking_flow_returns_status():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-track"))
    first = run(engine.process_user_input("call-track", "I want to track booking id DUMMY-10001"))

    assert first["action"] == "speak"
    assert "scheduled" in first["text_to_speak"].lower()
    assert "service" in first["text_to_speak"].lower()


def test_track_booking_with_numeric_booking_id():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-track-num"))
    turn = run(engine.process_user_input("call-track-num", "my booking id is 12614"))

    assert turn["action"] == "speak"
    assert "scheduled" in turn["text_to_speak"].lower()


def test_farewell_ends_call_without_forced_hangup_early():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-end"))
    start = run(engine.process_user_input("call-end", "I need cleaning"))
    end = run(engine.process_user_input("call-end", "bye"))

    assert start["action"] == "speak"
    assert end["action"] == "hangup"


def test_intent_guard_prevents_false_tracking_for_service_questions():
    engine = ConversationEngine(_Backend(), _WrongIntentLLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-guard-track"))
    turn = run(engine.process_user_input("call-guard-track", "I want to know about your services"))

    assert turn["action"] == "speak"
    assert "provide ac repair" in turn["text_to_speak"].lower()
    assert "booking id was not found" not in turn["text_to_speak"].lower()


def test_intent_guard_ignores_hallucinated_booking_id_for_service_questions():
    engine = ConversationEngine(_Backend(), _HallucinatedBookingIdLLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-hallucinated-id"))
    turn = run(engine.process_user_input("call-hallucinated-id", "Can you explain your services?"))

    assert turn["action"] == "speak"
    assert "provide ac repair" in turn["text_to_speak"].lower()
    assert "booking id was not found" not in turn["text_to_speak"].lower()


def test_post_booking_can_return_recent_booking_id():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-post-booking-id"))
    _ = run(engine.process_user_input("call-post-booking-id", "I need AC repair"))
    _ = run(engine.process_user_input("call-post-booking-id", "Manama"))
    _ = run(engine.process_user_input("call-post-booking-id", "6 PM"))
    follow_up = run(engine.process_user_input("call-post-booking-id", "Can you give me my booking ID?"))

    assert follow_up["action"] == "speak"
    assert "dummy-9001" in follow_up["text_to_speak"].lower()


def test_post_booking_gratitude_and_close_flow():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-thanks-close"))
    _ = run(engine.process_user_input("call-thanks-close", "I need AC repair"))
    _ = run(engine.process_user_input("call-thanks-close", "Manama"))
    _ = run(engine.process_user_input("call-thanks-close", "6 PM"))

    thanks = run(engine.process_user_input("call-thanks-close", "Thank you for the booking."))
    assert thanks["action"] == "speak"
    assert "you're welcome" in thanks["text_to_speak"].lower()

    close = run(engine.process_user_input("call-thanks-close", "No, I don't need anything else."))
    assert close["action"] == "hangup"


def test_post_booking_acknowledgement_does_not_force_hangup():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-post-book-ack"))
    _ = run(engine.process_user_input("call-post-book-ack", "I need AC repair"))
    _ = run(engine.process_user_input("call-post-book-ack", "Manama"))
    _ = run(engine.process_user_input("call-post-book-ack", "6 PM"))
    ack = run(engine.process_user_input("call-post-book-ack", "Okay sounds great"))

    assert ack["action"] == "speak"
    assert "booking id" in ack["text_to_speak"].lower()


def test_track_recent_booking_without_explicit_id_uses_last_booking_ref():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-track-last-id"))
    _ = run(engine.process_user_input("call-track-last-id", "I need AC repair"))
    _ = run(engine.process_user_input("call-track-last-id", "Manama"))
    _ = run(engine.process_user_input("call-track-last-id", "6 PM"))
    status = run(engine.process_user_input("call-track-last-id", "Can I know the status of my booking?"))

    assert status["action"] == "speak"
    assert "dummy-9001" in status["text_to_speak"].lower()
    assert "service:" in status["text_to_speak"].lower()


def test_low_confidence_intent_prompts_disambiguation():
    engine = ConversationEngine(_Backend(), _LowConfidenceBookingLLM(), _TTS(), session_manager=CallSessionManager())

    run(engine.start_session("call-low-confidence"))
    turn = run(engine.process_user_input("call-low-confidence", "hmm okay maybe"))

    assert turn["action"] == "speak"
    assert "three ways" in turn["text_to_speak"].lower()
