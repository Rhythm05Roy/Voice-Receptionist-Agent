import asyncio

import httpx

from src.core.services.backend_client import BackendClient


BUSINESS_PAYLOAD = [
    {
        "id": 1,
        "owner": 2,
        "name": "Urban Glow Salon",
        "business_email": "contact@urbanglowsalon.com",
        "business_website": "https://urbanglowsalon.com",
        "description": (
            "Urban Glow Salon offers premium hair, nail, and spa services with a focus "
            "on personalized customer care and modern styling trends."
        ),
        "business_type": "salon & spa",
        "services": [
            {
                "id": 1,
                "business": 1,
                "name": "Special haircut",
                "category_name": "hair cut",
                "description": "Personalized haircut and styling service.",
                "price": "500.00",
                "duration": 50,
                "allow_booking": True,
            }
        ],
        "faqs": [
            {
                "id": 1,
                "question": "Do I need an appointment?",
                "answer": "Appointments are recommended, but walk-ins are welcome depending on availability.",
            }
        ],
        "additional_info": [{"id": 1, "content": "It is a good business"}],
        "hours": [
            {"id": 3, "business": 1, "day": 3, "open_time": "04:00:22", "close_time": "09:00:22", "is_closed": False},
            {"id": 1, "business": 1, "day": 4, "open_time": "08:00:22", "close_time": "09:00:22", "is_closed": False},
        ],
        "policies": [
            {"id": 1, "policy_type": "cancellation", "content": "Please provide 24-hour notice for cancellations to avoid charges."},
            {"id": 2, "policy_type": "payment", "content": "We accept all major credit cards, cash, and mobile payment options."},
            {"id": 3, "policy_type": "deposit", "content": "A 30% deposit is required for bookings over 2 hours."},
        ],
        "address": "House 22, Road 15, Mirpur, Dhaka, Bangladesh",
        "timezone": "pacific",
        "is_active": True,
    }
]

INTAKE_PAYLOAD = [
    {
        "business": 1,
        "question": "Are you currently pregnant or under medical treatment that may affect skin treatments?",
        "answer_type": "yes_no",
        "when_to_ask": "tagged_services",
        "specific_categories": [2],
        "is_required": True,
        "is_active": True,
        "disqualification_rules": [
            {
                "disqualifying_value": "yes",
                "message_to_caller": "Certain skin treatments may not be suitable during pregnancy or medical treatment. Please consult staff before booking.",
            }
        ],
    }
]

LANGUAGE_PAYLOAD = [
    {"business": 1, "language_code": "en", "is_selected": True, "is_default": True},
    {"business": 1, "language_code": "ar", "is_selected": True, "is_default": False},
]

VOICE_PAYLOAD = [
    {"business": 1, "language_code": "en", "voice_id": "voice-en-123", "is_selected": True},
    {"business": 1, "language_code": "ar", "voice_id": "voice-ar-456", "is_selected": True},
]


def _make_client() -> BackendClient:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "http://example.test/api/business/businesses/":
            return httpx.Response(200, json=BUSINESS_PAYLOAD)
        if url == "http://example.test/api/agent/intake-questions/":
            return httpx.Response(200, json=INTAKE_PAYLOAD)
        if url == "http://example.test/api/agent/languages/":
            return httpx.Response(200, json=LANGUAGE_PAYLOAD)
        if url == "http://example.test/api/agent/agent-voices/":
            return httpx.Response(200, json=VOICE_PAYLOAD)
        raise AssertionError(f"Unexpected URL: {url}")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return BackendClient(
        client=client,
        base_url="http://example.test/api/business/businesses/",
        api_key="",
        local_test_mode=False,
    )


def _make_root_client() -> BackendClient:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "http://example.test/api/business/businesses/":
            return httpx.Response(200, json=BUSINESS_PAYLOAD)
        if url == "http://example.test/api/agent/intake-questions/":
            return httpx.Response(200, json=INTAKE_PAYLOAD)
        if url == "http://example.test/api/agent/languages/":
            return httpx.Response(200, json=LANGUAGE_PAYLOAD)
        if url == "http://example.test/api/agent/agent-voices/":
            return httpx.Response(200, json=VOICE_PAYLOAD)
        raise AssertionError(f"Unexpected URL: {url}")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return BackendClient(
        client=client,
        base_url="http://example.test/",
        api_key="",
        local_test_mode=False,
    )


def test_fetch_agent_config_from_business_catalog():
    backend = _make_client()
    config = asyncio.run(backend.fetch_agent_config("1"))

    assert config.agent_id == "1"
    assert config.business_name == "Urban Glow Salon"
    assert config.service_catalog[0].name == "Special haircut"
    assert config.service_catalog[0].base_price == "500.00"
    assert config.cancellation_policy.startswith("Please provide 24-hour notice")
    assert "Do I need an appointment?" in config.faqs
    assert config.supported_languages == ["en", "ar"]
    assert config.default_greeting_language == "en"
    assert config.language_voice_map == {"en": "voice-en-123", "ar": "voice-ar-456"}
    assert config.intake_questions[0].question.startswith("Are you currently pregnant")

    asyncio.run(backend.client.aclose())


def test_fetch_agent_config_from_api_root_base_url():
    backend = _make_root_client()
    config = asyncio.run(backend.fetch_agent_config("1"))

    assert config.agent_id == "1"
    assert config.business_name == "Urban Glow Salon"
    assert config.supported_languages == ["en", "ar"]
    assert config.language_voice_map == {"en": "voice-en-123", "ar": "voice-ar-456"}

    asyncio.run(backend.client.aclose())


def test_fetch_agent_ui_context_from_business_catalog():
    backend = _make_client()
    payload = asyncio.run(backend.fetch_agent_ui_context("1"))

    assert payload["business_name"] == "Urban Glow Salon"
    assert payload["services"][0]["name"] == "Special haircut"
    assert "Thu:" in payload["business_hours"]
    assert payload["payment_policy"].startswith("We accept all major credit cards")
    assert payload["supported_languages"] == ["en", "ar"]
    assert payload["intake_questions"][0]["answer_type"] == "yes_no"

    asyncio.run(backend.client.aclose())


def test_fetch_intake_question_configs_from_business_catalog_agent_routes():
    backend = _make_client()

    questions = asyncio.run(backend.fetch_intake_question_configs("1"))

    assert questions[0]["question"].startswith("Are you currently pregnant")
    assert questions[0]["when_to_ask"] == "tagged_services"

    asyncio.run(backend.client.aclose())


def test_runtime_phone_binding_resolves_agent_for_inbound():
    backend = _make_client()
    backend.bind_phone_number(agent_id="1", phone_number="+15145550101", phone_number_sid="PN123")

    resolved = asyncio.run(backend.resolve_agent_id_for_inbound("+15145550101"))

    assert resolved == "1"
    asyncio.run(backend.client.aclose())


def test_runtime_forwarding_overrides_agent_config():
    backend = _make_client()
    backend.set_call_forwarding(agent_id="1", forwarding_number="+97317000088")

    config = asyncio.run(backend.fetch_agent_config("1"))

    assert config.fallback_phone == "+97317000088"
    asyncio.run(backend.client.aclose())


def test_build_business_query_answer_focuses_on_specific_service():
    backend = _make_client()
    config = asyncio.run(backend.fetch_agent_config("1"))

    result = backend.build_business_query_answer(config, "Tell me about the special haircut service in detail")

    assert "Special haircut" in result["answer"]
    assert result["matched_services"] == ["Special haircut"]
    assert "Urban Glow Salon offers" not in result["answer"]

    asyncio.run(backend.client.aclose())


def test_build_business_query_answer_avoids_irrelevant_faq_leakage_for_payment_questions():
    backend = _make_client()
    config = asyncio.run(backend.fetch_agent_config("1"))
    config = config.model_copy(
        update={
            "payment_policy": "",
            "deposit_policy": "A 30% deposit is required for bookings over 2 hours.",
            "cancellation_policy": (
                "Please provide 24-hour notice for cancellations. "
                "Digital payments are accepted and refunds are processed within 3 business days."
            ),
            "faqs": {
                "Do you have vegetarian options?": "Yes, we offer vegetarian dishes.",
                "Do you offer delivery or takeaway?": "Yes, takeaway is available.",
            },
        }
    )

    result = backend.build_business_query_answer(config, "Do you accept digital payments and how do refunds work?")

    assert "digital payments" in result["answer"].lower()
    assert "refund" in result["answer"].lower()
    assert "vegetarian" not in result["answer"].lower()
    assert "takeaway" not in result["answer"].lower()

    asyncio.run(backend.client.aclose())
