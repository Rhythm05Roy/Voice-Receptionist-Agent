from src.api import deps


class _FakeEngine:
    async def start_session(self, call_id: str, agent_id: str | None = None) -> dict:
        return {"text": "hello", "audio_url": "data:audio/mpeg;base64,AAA"}

    def end_call(self, call_id: str) -> None:
        return None


class _FakeBackend:
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


def test_preview_returns_audio_url(client):
    deps_override = {
        deps.get_conversation_engine: lambda: _FakeEngine(),
        deps.get_backend_client: lambda: _FakeBackend(),
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.post("/api/v1/agent/preview", json={"agent_id": "agent-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["audio_url"].startswith("data:audio/mpeg")
    assert body["text"]

    client.app.dependency_overrides.clear()


def test_context_returns_ui_payload(client):
    deps_override = {
        deps.get_conversation_engine: lambda: _FakeEngine(),
        deps.get_backend_client: lambda: _FakeBackend(),
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
    deps_override = {
        deps.get_conversation_engine: lambda: _FakeEngine(),
        deps.get_backend_client: lambda: _FakeBackend(),
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.post("/api/v1/agent/query", json={"text": "what services do you provide?"})
    assert response.status_code == 200
    body = response.json()
    assert "Query received" in body["answer"]
    assert body["suggested_services"]

    client.app.dependency_overrides.clear()


def test_track_booking_returns_status(client):
    deps_override = {
        deps.get_conversation_engine: lambda: _FakeEngine(),
        deps.get_backend_client: lambda: _FakeBackend(),
    }
    client.app.dependency_overrides.update(deps_override)

    response = client.post("/api/v1/agent/track-booking", json={"booking_id": "DUMMY-10001"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "scheduled"
    assert body["booking_ref"] == "DUMMY-10001"

    client.app.dependency_overrides.clear()
