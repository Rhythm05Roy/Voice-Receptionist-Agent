from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice_id: str | None = Field(default=None)

    model_config = {"populate_by_name": True, "extra": "ignore"}


class TTSResponse(BaseModel):
    audio_url: str
    text: str
