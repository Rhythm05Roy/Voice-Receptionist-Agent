from src.api import deps


class _FakeEngine:
    async def start_session(self, call_id: str, agent_id: str | None = None) -> dict:
        return {"text": "hello", "audio_url": "data:audio/mpeg;base64,AAA"}

    def end_call(self, call_id: str) -> None:
        return None


def test_preview_returns_audio_url(client):
    deps_override = {deps.get_conversation_engine: lambda: _FakeEngine()}
    client.app.dependency_overrides.update(deps_override)

    response = client.post("/api/v1/agent/preview", json={"agent_id": "agent-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["audio_url"].startswith("data:audio/mpeg")
    assert body["text"]

    client.app.dependency_overrides.clear()
