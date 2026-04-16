from src.api import deps
from src.core.services.elevenlabs import ElevenLabsClient


class _FakeElevenLabs:
    async def synthesize_text(self, text: str, voice_id: str | None = None) -> str:
        return "data:audio/mpeg;base64,AAA"


def test_tts_endpoint(client):
    client.app.dependency_overrides[deps.get_elevenlabs_client] = lambda: _FakeElevenLabs()

    response = client.post("/api/v1/voice/tts", json={"text": "hello"})

    assert response.status_code == 200
    data = response.json()
    assert data["audio_url"].startswith("data:audio/mpeg")
    assert data["text"] == "hello"

    client.app.dependency_overrides.clear()


def test_cached_tts_audio_endpoint(client):
    audio_id = ElevenLabsClient.cache_audio_bytes(b"fake-mp3")
    client.app.dependency_overrides[deps.get_elevenlabs_client] = lambda: _FakeElevenLabs()

    response = client.get(f"/api/v1/voice/cache/{audio_id}")

    assert response.status_code == 200
    assert response.content == b"fake-mp3"
    assert response.headers["content-type"].startswith("audio/mpeg")

    client.app.dependency_overrides.clear()


def test_placeholder_voice_id_falls_back_to_default():
    elevenlabs = ElevenLabsClient(client=None, api_key="test", default_voice_id="real-default-voice")

    assert elevenlabs._resolve_voice("11labs-en-default") == "real-default-voice"
    assert elevenlabs._resolve_voice("elevenlabs-ar-default") == "real-default-voice"
    assert elevenlabs._resolve_voice("cNYrMw9glwJZXR8RwbuR") == "cNYrMw9glwJZXR8RwbuR"
