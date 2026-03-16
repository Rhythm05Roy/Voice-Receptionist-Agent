from fastapi import Depends, Request
from httpx import AsyncClient
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import get_settings, Settings
from src.core.services.backend_client import BackendClient
from src.core.services.elevenlabs import ElevenLabsClient
from src.core.services.twilio_client import TwilioClient
from src.core.services.openai import OpenAIClient
from src.core.conversation.engine import ConversationEngine

limiter = Limiter(key_func=get_remote_address, default_limits=[])


@limiter.limit("10/minute")
async def rate_limit_webhook(request: Request):
    return None


def get_settings_dep() -> Settings:
    return get_settings()


def get_http_client(request: Request) -> AsyncClient:
    return request.app.state.http_client


def get_backend_client(
    settings: Settings = Depends(get_settings_dep), client: AsyncClient = Depends(get_http_client)
) -> BackendClient:
    return BackendClient(
        client=client,
        base_url=str(settings.backend_base_url),
        api_key=settings.backend_api_key,
        local_test_mode=settings.local_test_mode,
    )


def get_elevenlabs_client(
    settings: Settings = Depends(get_settings_dep), client: AsyncClient = Depends(get_http_client)
) -> ElevenLabsClient:
    return ElevenLabsClient(
        client=client,
        api_key=settings.elevenlabs_api_key,
        default_voice_id=settings.elevenlabs_voice_id,
    )


def get_twilio_client(settings: Settings = Depends(get_settings_dep)) -> TwilioClient:
    return TwilioClient(
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
        phone_number=settings.twilio_phone_number,
        websocket_url=settings.twilio_websocket_url,
    )


# Legacy — kept for backward compatibility with existing Vonage telephony router
def get_vonage_client(settings: Settings = Depends(get_settings_dep)):
    try:
        from src.core.services.vonage import VonageClient
        return VonageClient(
            api_key=settings.vonage_api_key,
            api_secret=settings.vonage_api_secret,
            application_id=settings.vonage_application_id,
            private_key=settings.vonage_private_key,
        )
    except ImportError:
        # Return the Twilio client with NCCO compat layer
        return get_twilio_client(settings)


def get_llm_client(settings: Settings = Depends(get_settings_dep)) -> OpenAIClient:
    return OpenAIClient(api_key=settings.openai_api_key, model="gpt-4o")


def get_conversation_engine(request: Request) -> ConversationEngine:
    """Return the singleton engine created during app lifespan."""
    engine = getattr(request.app.state, "conversation_engine", None)
    if engine is not None:
        return engine

    # Fallback for tests that don't use the lifespan
    from src.config import get_settings
    settings = get_settings()
    http_client = request.app.state.http_client
    backend_client = BackendClient(
        client=http_client,
        base_url=str(settings.backend_base_url),
        api_key=settings.backend_api_key,
        local_test_mode=settings.local_test_mode,
    )
    llm_client = OpenAIClient(api_key=settings.openai_api_key, model="gpt-4o")
    tts_client = ElevenLabsClient(
        client=http_client,
        api_key=settings.elevenlabs_api_key,
        default_voice_id=settings.elevenlabs_voice_id,
    )
    from src.core.conversation.engine import CallSessionManager
    engine = ConversationEngine(
        backend_client=backend_client,
        llm_client=llm_client,
        tts_client=tts_client,
        environment=settings.environment,
        session_manager=CallSessionManager(),
        context_refresh_ttl_seconds=settings.context_refresh_ttl_seconds,
    )
    request.app.state.conversation_engine = engine
    return engine
