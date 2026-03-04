import uuid

from fastapi import APIRouter, Depends

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


@router.post("/preview", response_model=TTSResponse)
async def preview_agent_greeting(
    payload: AgentPreviewRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> TTSResponse:
    call_id = f"preview-{uuid.uuid4()}"
    result = await engine.start_session(call_id=call_id, agent_id=payload.agent_id)
    engine.end_call(call_id)
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
