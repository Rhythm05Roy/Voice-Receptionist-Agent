from src.api import deps


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
