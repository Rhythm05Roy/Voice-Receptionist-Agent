from src.api import deps


class _FakeEngine:
    def __init__(self):
        self._sessions = set()

    def has_session(self, call_id: str) -> bool:
        return call_id in self._sessions

    async def start_session(self, call_id: str, agent_id: str | None = None):
        self._sessions.add(call_id)
        return {"text": "مرحبا", "audio_url": "data:audio/mpeg;base64,AAA"}

    async def process_user_input(self, call_id: str, transcribed_text: str):
        return {"action": "speak", "text_to_speak": f"echo {transcribed_text}", "transfer_number": None}


class _FakeVonage:
    def build_talk_ncco(self, text: str, voice_name: str | None = None):
        return [{"action": "talk", "text": text, "voiceName": voice_name or "Zeina"}]

    def build_listen_action(self, event_url=None, speech_timeout=5):
        return {"action": "listen", "eventUrl": event_url or []}

    def build_hangup_ncco(self):
        return {"action": "hangup"}

    def build_action_ncco(self, action: dict, from_number: str | None = None, event_url=None):
        return self.build_talk_ncco(action.get("text_to_speak", "")) + [self.build_listen_action(event_url=event_url)]


def test_inbound_webhook_greets_on_first_call(client):
    client.app.dependency_overrides.update(
        {
            deps.get_conversation_engine: lambda: _FakeEngine(),
            deps.get_vonage_client: lambda: _FakeVonage(),
            deps.rate_limit_webhook: lambda: None,
        }
    )

    payload = {"from": "+973111", "to": "+973222", "uuid": "abc"}
    response = client.post("/api/v1/telephony/webhook/inbound", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["ncco"][0]["action"] == "talk"
    assert body["ncco"][1]["action"] == "listen"

    client.app.dependency_overrides.clear()


def test_inbound_webhook_processes_speech(client):
    engine = _FakeEngine()
    engine._sessions.add("abc")
    client.app.dependency_overrides.update(
        {
            deps.get_conversation_engine: lambda: engine,
            deps.get_vonage_client: lambda: _FakeVonage(),
            deps.rate_limit_webhook: lambda: None,
        }
    )

    payload = {
        "from": "+973111",
        "to": "+973222",
        "uuid": "abc",
        "speech": {"results": [{"text": "need cleaning"}]},
    }
    response = client.post("/api/v1/telephony/webhook/inbound", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["ncco"][0]["text"].startswith("echo need cleaning")

    client.app.dependency_overrides.clear()
