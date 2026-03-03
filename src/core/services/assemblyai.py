import asyncio
import uuid
from collections.abc import AsyncGenerator
from httpx import AsyncClient, HTTPError
from loguru import logger

from src.api.exceptions import BackendCommunicationError


class RealTimeSession:
    """Local stub simulating AssemblyAI real-time websocket.

    In production, replace this with `assemblyai.RealtimeClient` websocket:
    - open websocket to wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000
    - send binary audio chunks
    - receive partial/final transcripts
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self._buffer: list[bytes] = []

    async def stream_audio(self, chunks: AsyncGenerator[bytes, None]) -> AsyncGenerator[dict, None]:
        async for chunk in chunks:
            self._buffer.append(chunk)
            yield {"message_type": "PartialTranscript", "text": "..."}
        yield {"message_type": "FinalTranscript", "text": "(captured audio)"}

    async def close(self) -> dict:
        transcript = "" if not self._buffer else "(streamed audio captured)"
        return {"session_id": self.session_id, "transcript": transcript}


class AssemblyAIClient:
    def __init__(self, client: AsyncClient, api_key: str):
        self.client = client
        self.api_key = api_key
        self.base_url = "https://api.assemblyai.com/v2"

    @property
    def _headers(self) -> dict[str, str]:
        return {"authorization": self.api_key}

    async def start_transcription(self, audio_url: str) -> dict:
        url = f"{self.base_url}/transcript"
        try:
            resp = await self.client.post(url, headers=self._headers, json={"audio_url": audio_url})
            resp.raise_for_status()
            data = resp.json()
            logger.info("AssemblyAI transcription started", transcript_id=data.get("id"))
            return data
        except HTTPError as exc:
            logger.exception("AssemblyAI request failed")
            raise BackendCommunicationError(str(exc))

    async def start_realtime_session(self) -> RealTimeSession:
        session = RealTimeSession()
        logger.debug("Started pseudo real-time transcription session", session_id=session.session_id)
        return session
