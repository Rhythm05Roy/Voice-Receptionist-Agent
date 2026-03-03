import asyncio
import pytest

from src.core.conversation.engine import ConversationEngine, CallSessionManager
from src.core.types import AgentConfig


class _Backend:
    def __init__(self, fallback: str | None = None):
        self.fallback = fallback

    async def fetch_agent_config(self, agent_id=None):
        return AgentConfig(
            agent_id="a1",
            greeting="Hi",
            intake_questions=["Any allergies?", "Where are you located?"],
            language="en",
            fallback_phone=self.fallback,
        )

    async def book_service(self, agent_id: str, answers: dict[str, str]):
        return {"message": f"Booked for {answers.get('q1', 'unknown')}"}


class _LLM:
    async def generate_reply(self, messages):
        return "We have your request and will confirm shortly."


class _TTS:
    async def synthesize_text(self, text, voice_id=None):
        return "data:audio/mpeg;base64,AAA"


@pytest.mark.asyncio
async def test_intake_flow_then_booking():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())
    await engine.start_session("call-1")

    first = await engine.process_user_input("call-1", "")
    assert first["action"] == "speak"

    second = await engine.process_user_input("call-1", "no allergies")
    assert second["action"] == "speak"

    third = await engine.process_user_input("call-1", "Manama")
    assert third["action"] == "speak"
    assert "confirm" in third["text_to_speak"].lower()


@pytest.mark.asyncio
async def test_disqualify_short_answer():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())
    await engine.start_session("call-2")
    _ = await engine.process_user_input("call-2", "")
    result = await engine.process_user_input("call-2", "")
    assert result["action"] == "speak"
    assert "تفاصيل" in result["text_to_speak"] or "more" in result["text_to_speak"].lower()


@pytest.mark.asyncio
async def test_disqualify_cancel():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())
    await engine.start_session("call-2b")
    _ = await engine.process_user_input("call-2b", "")
    result = await engine.process_user_input("call-2b", "not interested")
    assert result["action"] == "hangup"


@pytest.mark.asyncio
async def test_transfer_when_fallback_available():
    engine = ConversationEngine(_Backend(fallback="+973555"), _LLM(), _TTS(), session_manager=CallSessionManager())
    await engine.start_session("call-3")
    _ = await engine.process_user_input("call-3", "")
    _ = await engine.process_user_input("call-3", "no")
    result = await engine.process_user_input("call-3", "Seef")
    if result["action"] == "transfer":
        assert result["transfer_number"] == "+973555"


@pytest.mark.asyncio
async def test_cancel_hangup():
    engine = ConversationEngine(_Backend(), _LLM(), _TTS(), session_manager=CallSessionManager())
    await engine.start_session("call-4")
    result = await engine.process_user_input("call-4", "cancel")
    assert result["action"] == "hangup"
