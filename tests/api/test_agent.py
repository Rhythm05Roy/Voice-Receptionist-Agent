from src.api import deps


class _FakeEngine:
    async def start_session(self, call_id: str, agent_id: str | None = None, caller_number: str | None = None, called_number: str | None = None) -> dict:
        return {"text": "hello", "audio_url": "data:audio/mpeg;base64,AAA"}

    async def end_call(self, call_id: str) -> None:
        return None

    async def build_call_report(self, call_id: str) -> dict:
        return {
            "call_id": call_id,
            "agent_id": "agent-1",
            "business_name": "Test Biz",
            "customer_details": {"phone_number": "+14165550101", "name": "Ridam"},
            "order_or_booked_service": {"interaction_type": "booking", "service_type": "Home Deep Cleaning"},
            "configured_intake_questions": [],
            "call_analytics": {"duration_seconds": 32.5, "turn_count": 4, "outcome": "completed"},
            "transcript": [{"role": "user", "content": "Need cleaning"}],
        }


class _FakeBackend:
    def __init__(self):
        self.bindings = {}
        self.forwarding = {}

    async def fetch_agent_ui_context(self, agent_id: str | None = None) -> dict:
        return {
            "agent_id": agent_id or "default",
            "business_name": "Test Biz",
            "greeting": "Hello",
            "language": "en",
            "coverage_country": "Bahrain",
            "coverage_areas": ["Manama"],
            "booking_required_fields": ["service_type", "location"],
            "fallback_phone": "+97317000000",
            "services": [
                {
                    "service_id": "cleaning",
                    "name": "Home Deep Cleaning",
                    "description": "Deep clean",
                    "base_price_bhd": "12-45 BHD",
                    "price_note": "Depends on home size",
                }
            ],
            "faqs": {"pricing": "Starts from 12 BHD"},
            "notes": ["dummy"],
        }

    async def answer_business_query(self, query: str, agent_id: str | None = None) -> dict:
        return {
            "answer": f"Query received: {query}",
            "suggested_services": ["Home Deep Cleaning"],
        }

    async def track_booking(self, booking_id: str, agent_id: str | None = None) -> dict:
        return {
            "status": "scheduled",
            "booking_ref": booking_id,
            "message": "Technician is scheduled for this booking.",
            "service_name": "Home Deep Cleaning",
            "location": "Manama",
            "preferred_time": "6:00 PM",
        }

    def bind_phone_number(self, *, agent_id: str, phone_number: str, phone_number_sid: str = "", friendly_name: str = ""):
        self.bindings[agent_id] = {
            "phone_number": phone_number,
            "phone_number_sid": phone_number_sid,
            "friendly_name": friendly_name,
        }
        return self.bindings[agent_id]

    def set_call_forwarding(self, *, agent_id: str, forwarding_number: str):
        self.forwarding[agent_id] = forwarding_number
        return {"agent_id": agent_id, "forwarding_number": forwarding_number}

    def get_phone_assignment(self, agent_id: str):
        assignment = self.bindings.get(agent_id)
        if not assignment:
            return None
        payload = dict(assignment)
        if agent_id in self.forwarding:
            payload["forwarding_number"] = self.forwarding[agent_id]
        return payload

    def rebind_phone_number(self, *, agent_id: str, phone_number: str, phone_number_sid: str = "", friendly_name: str = ""):
        for current_agent_id, assignment in list(self.bindings.items()):
            if assignment.get("phone_number") == phone_number and current_agent_id != agent_id:
                self.bindings.pop(current_agent_id, None)
        return self.bind_phone_number(
            agent_id=agent_id,
            phone_number=phone_number,
            phone_number_sid=phone_number_sid,
            friendly_name=friendly_name,
        )

    def release_phone_number(self, *, agent_id: str | None = None, phone_number: str | None = None, phone_number_sid: str | None = None):
        if agent_id and agent_id in self.bindings:
            assignment = self.bindings.pop(agent_id)
            forwarding_number = self.forwarding.get(agent_id)
            return {
                "agent_id": agent_id,
                "phone_number": assignment.get("phone_number"),
                "phone_number_sid": assignment.get("phone_number_sid"),
                "forwarding_number": forwarding_number,
            }
        return None


class _FakeTwilio:
    credentials_available = True

    def search_available_numbers(self, **kwargs) -> list[dict]:
        return [
            {
                "phone_number": "+14385335861",
                "friendly_name": "Montreal Test Number",
                "locality": "Montreal",
                "region": "QC",
                "iso_country": kwargs["country_code"],
                "capabilities": {"voice": True, "sms": True},
            }
        ]

    def provision_incoming_number(self, **kwargs) -> dict:
        return {
            "agent_id": kwargs["agent_id"],
            "phone_number_sid": "PN123",
            "phone_number": "+14165550101",
            "friendly_name": kwargs.get("friendly_name") or "agent:agent-1:+14165550101",
            "voice_url": f"{kwargs['public_base_url']}/api/v1/twilio/webhook/incoming?agent_id={kwargs['agent_id']}",
            "status_callback": f"{kwargs['public_base_url']}/api/v1/twilio/webhook/status?agent_id={kwargs['agent_id']}",
            "capabilities": {"voice": True, "sms": True},
            "country_code": kwargs["country_code"],
            "number_type": kwargs["number_type"],
            "account_sid": "AC123",
        }

    def update_incoming_number_binding(self, **kwargs) -> dict:
        return {
            "agent_id": kwargs["agent_id"],
            "phone_number_sid": kwargs["phone_number_sid"],
            "phone_number": "+14165550101",
            "friendly_name": kwargs.get("friendly_name") or "rebound-line",
            "voice_url": f"{kwargs['public_base_url']}/api/v1/twilio/webhook/incoming?agent_id={kwargs['agent_id']}",
            "status_callback": f"{kwargs['public_base_url']}/api/v1/twilio/webhook/status?agent_id={kwargs['agent_id']}",
            "capabilities": {"voice": True},
            "account_sid": "AC123",
        }

    def release_incoming_number(self, **kwargs) -> None:
        return None


class _FakeSettings:
    def __init__(self, public_base_url: str = "https://voice.example.com"):
        self.public_base_url = public_base_url


def test_preview_returns_audio_url(client):
    backend = _FakeBackend()
    deps_override = {
        deps.get_conversation_engine: lambda: _FakeEngine(),
        deps.get_backend_client: lambda: backend,
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.post("/api/v1/agent/preview", json={"agent_id": "agent-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["audio_url"].startswith("data:audio/mpeg")
    assert body["text"]

    client.app.dependency_overrides.clear()


def test_context_returns_ui_payload(client):
    backend = _FakeBackend()
    deps_override = {
        deps.get_conversation_engine: lambda: _FakeEngine(),
        deps.get_backend_client: lambda: backend,
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.get("/api/v1/agent/context", params={"agent_id": "agent-9"})
    assert response.status_code == 200
    body = response.json()
    assert body["agent_id"] == "agent-9"
    assert body["business_name"] == "Test Biz"
    assert body["services"]

    client.app.dependency_overrides.clear()


def test_query_returns_answer(client):
    backend = _FakeBackend()
    deps_override = {
        deps.get_conversation_engine: lambda: _FakeEngine(),
        deps.get_backend_client: lambda: backend,
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.post("/api/v1/agent/query", json={"text": "what services do you provide?"})
    assert response.status_code == 200
    body = response.json()
    assert "Query received" in body["answer"]
    assert body["suggested_services"]

    client.app.dependency_overrides.clear()


def test_track_booking_returns_status(client):
    backend = _FakeBackend()
    deps_override = {
        deps.get_conversation_engine: lambda: _FakeEngine(),
        deps.get_backend_client: lambda: backend,
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.post("/api/v1/agent/track-booking", json={"booking_id": "DUMMY-10001"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "scheduled"
    assert body["booking_ref"] == "DUMMY-10001"

    client.app.dependency_overrides.clear()


def test_call_report_returns_structured_payload(client):
    backend = _FakeBackend()
    deps_override = {
        deps.get_conversation_engine: lambda: _FakeEngine(),
        deps.get_backend_client: lambda: backend,
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.post("/api/v1/agent/call-report", json={"call_id": "call-123"})

    assert response.status_code == 200
    body = response.json()
    assert body["call_id"] == "call-123"
    assert body["customer_details"]["name"] == "Ridam"
    assert body["order_or_booked_service"]["interaction_type"] == "booking"

    client.app.dependency_overrides.clear()


def test_search_phone_numbers_returns_matches(client):
    client.app.dependency_overrides.update({deps.get_twilio_client: lambda: _FakeTwilio()})

    response = client.get(
        "/api/v1/agent/phone-numbers/search",
        params={"country_code": "CA", "number_type": "local", "area_code": 438, "limit": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["phone_number"] == "+14385335861"
    assert body[0]["locality"] == "Montreal"

    client.app.dependency_overrides.clear()


def test_provision_phone_number_returns_allocation(client):
    backend = _FakeBackend()
    deps_override = {
        deps.get_settings_dep: lambda: _FakeSettings(),
        deps.get_twilio_client: lambda: _FakeTwilio(),
        deps.get_backend_client: lambda: backend,
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.post(
        "/api/v1/agent/phone-numbers/provision",
        json={
            "agent_id": "agent-1",
            "country_code": "CA",
            "number_type": "local",
            "area_code": 514,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_id"] == "agent-1"
    assert body["phone_number_sid"] == "PN123"
    assert body["voice_url"].endswith("/api/v1/twilio/webhook/incoming?agent_id=agent-1")
    assert backend.bindings["agent-1"]["phone_number"] == "+14165550101"

    client.app.dependency_overrides.clear()


def test_provision_phone_number_requires_public_base_url(client):
    backend = _FakeBackend()
    deps_override = {
        deps.get_settings_dep: lambda: _FakeSettings(public_base_url=""),
        deps.get_twilio_client: lambda: _FakeTwilio(),
        deps.get_backend_client: lambda: backend,
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.post(
        "/api/v1/agent/phone-numbers/provision",
        json={"agent_id": "agent-1", "country_code": "CA", "number_type": "local"},
    )

    assert response.status_code == 400
    assert "PUBLIC_BASE_URL" in response.json()["detail"]

    client.app.dependency_overrides.clear()


def test_call_forwarding_updates_agent_binding(client):
    backend = _FakeBackend()
    backend.bind_phone_number(agent_id="agent-1", phone_number="+14165550101")
    client.app.dependency_overrides.update({deps.get_backend_client: lambda: backend})

    response = client.post(
        "/api/v1/agent/phone-numbers/forwarding",
        json={"agent_id": "agent-1", "forwarding_number": "+97317000099"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_id"] == "agent-1"
    assert body["forwarding_number"] == "+97317000099"
    assert body["assigned_phone_number"] == "+14165550101"
    assert backend.forwarding["agent-1"] == "+97317000099"

    client.app.dependency_overrides.clear()


def test_get_phone_assignment_returns_current_binding(client):
    backend = _FakeBackend()
    backend.bind_phone_number(agent_id="agent-1", phone_number="+14165550101", phone_number_sid="PN123")
    backend.set_call_forwarding(agent_id="agent-1", forwarding_number="+97317000099")
    client.app.dependency_overrides.update({deps.get_backend_client: lambda: backend})

    response = client.get("/api/v1/agent/phone-numbers/assignment", params={"agent_id": "agent-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["phone_number"] == "+14165550101"
    assert body["phone_number_sid"] == "PN123"
    assert body["forwarding_number"] == "+97317000099"

    client.app.dependency_overrides.clear()


def test_rebind_phone_number_updates_assignment(client):
    backend = _FakeBackend()
    backend.bind_phone_number(agent_id="agent-old", phone_number="+14165550101", phone_number_sid="PN123")
    deps_override = {
        deps.get_settings_dep: lambda: _FakeSettings(),
        deps.get_twilio_client: lambda: _FakeTwilio(),
        deps.get_backend_client: lambda: backend,
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.post(
        "/api/v1/agent/phone-numbers/rebind",
        json={
            "agent_id": "agent-new",
            "phone_number_sid": "PN123",
            "phone_number": "+14165550101",
            "friendly_name": "new-line",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_id"] == "agent-new"
    assert backend.bindings["agent-new"]["phone_number"] == "+14165550101"
    assert "agent-old" not in backend.bindings

    client.app.dependency_overrides.clear()


def test_release_phone_number_clears_assignment(client):
    backend = _FakeBackend()
    backend.bind_phone_number(agent_id="agent-1", phone_number="+14165550101", phone_number_sid="PN123")
    backend.set_call_forwarding(agent_id="agent-1", forwarding_number="+97317000099")
    deps_override = {
        deps.get_twilio_client: lambda: _FakeTwilio(),
        deps.get_backend_client: lambda: backend,
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.post(
        "/api/v1/agent/phone-numbers/release",
        json={"agent_id": "agent-1", "release_provider_number": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "released"
    assert body["phone_number"] == "+14165550101"
    assert backend.get_phone_assignment("agent-1") is None

    client.app.dependency_overrides.clear()
