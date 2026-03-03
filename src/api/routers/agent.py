import uuid
from fastapi import APIRouter, Depends

from src.api.deps import get_conversation_engine
from src.core.conversation.engine import ConversationEngine
from src.schemas import AgentPreviewRequest, TTSResponse

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
