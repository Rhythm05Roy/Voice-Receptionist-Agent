from fastapi import APIRouter, Depends

from src.api.deps import get_elevenlabs_client
from src.core.services.elevenlabs import ElevenLabsClient
from src.schemas import TTSRequest, TTSResponse

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/tts", response_model=TTSResponse)
async def generate_tts(
    payload: TTSRequest,
    elevenlabs_client: ElevenLabsClient = Depends(get_elevenlabs_client),
) -> TTSResponse:
    audio_url = await elevenlabs_client.synthesize_text(payload.text, voice_id=payload.voice_id)
    return TTSResponse(audio_url=audio_url, text=payload.text)
