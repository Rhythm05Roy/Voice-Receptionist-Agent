import uuid

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from src.api.deps import get_backend_client, get_conversation_engine
from src.core.conversation.engine import ConversationEngine
from src.core.services.backend_client import BackendClient
from src.schemas import (
    AgentBusinessQueryRequest,
    AgentBusinessQueryResponse,
    AgentPreviewRequest,
    AgentTrackBookingRequest,
    AgentTrackBookingResponse,
    AgentUIContextResponse,
    TTSResponse,
)

router = APIRouter(prefix="/agent", tags=["agent"])


# ── Existing Endpoints ───────────────────────────────────────────

@router.post("/preview", response_model=TTSResponse)
async def preview_agent_greeting(
    payload: AgentPreviewRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> TTSResponse:
    call_id = f"preview-{uuid.uuid4()}"
    result = await engine.start_session(call_id=call_id, agent_id=payload.agent_id)
    await engine.end_call(call_id)
    return TTSResponse(audio_url=result["audio_url"], text=result["text"])


@router.get("/context", response_model=AgentUIContextResponse)
async def get_agent_context(
    agent_id: str | None = None,
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentUIContextResponse:
    payload = await backend_client.fetch_agent_ui_context(agent_id=agent_id)
    return AgentUIContextResponse(**payload)


@router.post("/query", response_model=AgentBusinessQueryResponse)
async def answer_agent_business_query(
    payload: AgentBusinessQueryRequest,
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentBusinessQueryResponse:
    result = await backend_client.answer_business_query(query=payload.text, agent_id=payload.agent_id)
    return AgentBusinessQueryResponse(**result)


@router.post("/track-booking", response_model=AgentTrackBookingResponse)
async def track_booking(
    payload: AgentTrackBookingRequest,
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentTrackBookingResponse:
    result = await backend_client.track_booking(payload.booking_id, agent_id=payload.agent_id)
    return AgentTrackBookingResponse(**result)


# ── Agent Test Endpoints (Figma: "Talk Your Agent" button) ───────


class TestStartRequest(BaseModel):
    agent_id: str | None = None


class TestTurnRequest(BaseModel):
    session_id: str
    text: str
    agent_id: str | None = None


class TestResponse(BaseModel):
    session_id: str
    text: str
    action: str = "speak"
    is_active: bool = True


@router.post("/test-start", response_model=TestResponse)
async def test_start(
    payload: TestStartRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> TestResponse:
    """Start a test conversation with the agent. Returns greeting."""
    session_id = f"test-{uuid.uuid4()}"
    result = await engine.start_session(
        call_id=session_id,
        agent_id=payload.agent_id,
        is_test=True,
    )
    return TestResponse(
        session_id=session_id,
        text=result["text"],
        action="speak",
        is_active=True,
    )


@router.post("/test-turn", response_model=TestResponse)
async def test_turn(
    payload: TestTurnRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> TestResponse:
    """Send a message in a test conversation."""
    if not await engine.has_session(payload.session_id):
        # Auto-start if session expired
        await engine.start_session(
            call_id=payload.session_id,
            agent_id=payload.agent_id,
            is_test=True,
        )

    result = await engine.process_user_input(
        call_id=payload.session_id,
        transcribed_text=payload.text,
        agent_id=payload.agent_id,
    )

    is_active = result.get("action") != "hangup"

    return TestResponse(
        session_id=payload.session_id,
        text=result["text_to_speak"],
        action=result["action"],
        is_active=is_active,
    )


@router.post("/test-end")
async def test_end(
    payload: TestTurnRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> dict:
    """End a test conversation and return summary."""
    await engine.end_call(payload.session_id)
    return {
        "status": "ended",
        "session_id": payload.session_id,
    }
