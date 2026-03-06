from pydantic import BaseModel, Field
from typing import Any


class SpeechResult(BaseModel):
    text: str | None = None
    confidence: float | None = None


class SpeechPayload(BaseModel):
    timeout_reason: str | None = Field(default=None, alias="timeout_reason")
    results: list[SpeechResult] | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @property
    def best_text(self) -> str | None:
        if not self.results:
            return None
        for res in self.results:
            if res.text:
                return res.text
        return None


class TelephonyWebhookRequest(BaseModel):
    from_number: str = Field(alias="from")
    to_number: str = Field(alias="to")
    uuid: str
    conversation_uuid: str | None = None
    dtmf: str | None = None
    speech: SpeechPayload | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @property
    def speech_text(self) -> str | None:
        return self.speech.best_text if self.speech else None


class TelephonyEventRequest(BaseModel):
    uuid: str
    status: str | None = None
    conversation_uuid: str | None = None
    duration: int | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}


class TelephonyResponse(BaseModel):
    ncco: list[dict[str, Any]]
