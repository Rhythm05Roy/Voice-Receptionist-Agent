from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

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


@router.get("/cache/{audio_id}")
async def get_cached_tts_audio(
    audio_id: str,
    _: ElevenLabsClient = Depends(get_elevenlabs_client),
) -> Response:
    audio_bytes = ElevenLabsClient.get_cached_audio_bytes(audio_id)
    if not audio_bytes:
        raise HTTPException(status_code=404, detail="Audio not found or expired.")
    return Response(content=audio_bytes, media_type="audio/mpeg")
