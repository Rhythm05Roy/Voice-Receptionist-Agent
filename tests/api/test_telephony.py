from src.api import deps


class _FakeEngine:
    def __init__(self):
        self._sessions = set()
        self.ended = []

    async def has_session(self, call_id: str) -> bool:
        return call_id in self._sessions

    async def start_session(self, call_id: str, agent_id: str | None = None):
        self._sessions.add(call_id)
        return {"text": "hello", "audio_url": "data:audio/mpeg;base64,AAA"}

    async def process_user_input(self, call_id: str, transcribed_text: str, agent_id: str | None = None):
        if transcribed_text.lower() in {"bye", "hangup"}:
            return {"action": "hangup", "text_to_speak": "bye", "transfer_number": None}
        return {"action": "speak", "text_to_speak": f"echo {transcribed_text}", "transfer_number": None}

    async def end_call(self, call_id: str) -> None:
        self.ended.append(call_id)


class _FakeVonage:
    """Backward-compat fake that works with both Vonage and Twilio NCCO bridge."""
    def build_talk_ncco(self, text: str, voice_name: str | None = None):
        return [{"action": "talk", "text": text, "voiceName": voice_name or "Polly.Joanna"}]

    def build_listen_action(self, event_url=None, speech_timeout=7):
        return {"action": "listen", "eventUrl": event_url or [], "speechTimeout": speech_timeout}

    def build_hangup_ncco(self):
        return {"action": "hangup"}

    def build_action_ncco(self, action: dict, from_number: str | None = None, event_url=None):
        if action.get("action") == "hangup":
            return self.build_talk_ncco(action.get("text_to_speak", "")) + [self.build_hangup_ncco()]
        return self.build_talk_ncco(action.get("text_to_speak", "")) + [self.build_listen_action(event_url=event_url)]


class _FakeBackend:
    async def resolve_agent_id_for_inbound(self, to_number: str | None):
        return "default"


def _overrides(client, engine):
    client.app.dependency_overrides.update(
        {
            deps.get_conversation_engine: lambda: engine,
            deps.get_vonage_client: lambda: _FakeVonage(),
            deps.get_backend_client: lambda: _FakeBackend(),
            deps.rate_limit_webhook: lambda: None,
        }
    )


def test_inbound_webhook_greets_on_first_call(client):
    engine = _FakeEngine()
    _overrides(client, engine)

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
    _overrides(client, engine)

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


def test_inbound_webhook_handles_first_turn_speech(client):
    engine = _FakeEngine()
    _overrides(client, engine)

    payload = {
        "from": "+973111",
        "to": "+973222",
        "uuid": "first-speech",
        "speech": {"results": [{"text": "book cleaning"}]},
    }
    response = client.post("/api/v1/telephony/webhook/inbound", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["ncco"][0]["text"].startswith("echo book cleaning")

    client.app.dependency_overrides.clear()


def test_event_webhook_ends_call_on_terminal_status(client):
    engine = _FakeEngine()
    _overrides(client, engine)

    response = client.post("/api/v1/telephony/webhook/event", json={"uuid": "abc", "status": "completed"})
    assert response.status_code == 200
    assert "abc" in engine.ended

    client.app.dependency_overrides.clear()
