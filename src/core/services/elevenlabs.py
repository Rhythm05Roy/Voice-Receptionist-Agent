import base64
from httpx import AsyncClient, HTTPError, HTTPStatusError
from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception, before_sleep_log

from src.api.exceptions import VoiceGenerationError


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, HTTPStatusError) and exc.response is not None:
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class ElevenLabsClient:
    def __init__(self, client: AsyncClient, api_key: str, default_voice_id: str):
        self.client = client
        self.api_key = api_key
        self.default_voice_id = default_voice_id
        self._voice_cache: set[str] = {default_voice_id}

    def _resolve_voice(self, voice_id: str | None) -> str:
        if voice_id:
            self._voice_cache.add(voice_id)
            return voice_id
        return self.default_voice_id

    @retry(
        reraise=True,
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
    async def synthesize_text(self, text: str, voice_id: str | None = None) -> str:
        voice = self._resolve_voice(voice_id)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        payload = {"text": text, "model_id": "eleven_multilingual_v2"}
        headers = {"xi-api-key": self.api_key, "Accept": "audio/mpeg"}
        try:
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            audio_bytes = response.content
            data_url = "data:audio/mpeg;base64," + base64.b64encode(audio_bytes).decode()
            logger.info("Generated TTS", voice_id=voice)
            return data_url
        except HTTPError as exc:  # includes HTTPStatusError
            logger.exception("TTS generation failed")
            raise VoiceGenerationError(str(exc))
