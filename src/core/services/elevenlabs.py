import base64
from typing import AsyncIterator

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
        """Batch TTS — returns complete audio as data URL."""
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

    async def stream_speech(
        self,
        text: str,
        voice_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Streaming TTS — yields audio chunks for low-latency playback.

        Uses ElevenLabs streaming endpoint to start receiving audio
        before the full text is synthesized. This dramatically reduces
        time-to-first-byte (TTFB).
        """
        voice = self._resolve_voice(voice_id)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "output_format": "mp3_44100_128",
        }
        headers = {
            "xi-api-key": self.api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        try:
            async with self.client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    yield chunk
        except HTTPError as exc:
            logger.exception("Streaming TTS failed")
            raise VoiceGenerationError(str(exc))

    async def synthesize_text_fast(self, text: str, voice_id: str | None = None) -> str:
        """Turbo TTS — uses faster model for lower latency at slight quality tradeoff."""
        voice = self._resolve_voice(voice_id)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",  # Faster model
            "output_format": "mp3_22050_32",   # Smaller output
        }
        headers = {"xi-api-key": self.api_key, "Accept": "audio/mpeg"}
        try:
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            audio_bytes = response.content
            data_url = "data:audio/mpeg;base64," + base64.b64encode(audio_bytes).decode()
            logger.info("Generated turbo TTS", voice_id=voice, text_len=len(text))
            return data_url
        except HTTPError as exc:
            logger.exception("Turbo TTS failed, falling back to standard")
            return await self.synthesize_text(text, voice_id)
