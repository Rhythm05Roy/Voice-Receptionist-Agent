"""Twilio webhooks for inbound voice calls."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import Response
from loguru import logger

from src.api.deps import (
    get_backend_client,
    get_conversation_engine,
    get_elevenlabs_client,
    get_settings_dep,
    get_twilio_client,
    rate_limit_webhook,
)
from src.config import Settings
from src.core.conversation.engine import ConversationEngine
from src.core.services.backend_client import BackendClient
from src.core.services.elevenlabs import ElevenLabsClient
from src.core.services.twilio_client import TwilioClient

router = APIRouter(prefix="/twilio", tags=["twilio"])

TERMINAL_STATUSES = {"completed", "failed", "busy", "no-answer", "canceled"}
TWIML_CONTENT_TYPE = "application/xml"


def _signed_url(request: Request, settings: Settings) -> str:
    if settings.public_base_url:
        base = settings.public_base_url.rstrip("/")
        query = f"?{request.url.query}" if request.url.query else ""
        return f"{base}{request.url.path}{query}"
    return str(request.url)


def _action_url(request: Request, settings: Settings) -> str:
    query = f"?{request.url.query}" if request.url.query else ""
    if settings.public_base_url:
        return f"{settings.public_base_url.rstrip('/')}{request.url.path}{query}"
    return f"{request.url_for('twilio_incoming_call')}{query}"


async def _build_play_url(
    *,
    text: str,
    call_id: str,
    settings: Settings,
    engine: ConversationEngine,
    elevenlabs_client: ElevenLabsClient,
) -> str | None:
    if not text or not settings.public_base_url:
        return None

    voice_id = None
    session = await engine.sessions.get(call_id)
    if session is not None:
        preferred_language = session.preferred_language or session.agent_config.default_greeting_language or session.agent_config.language
        voice_id = session.agent_config.language_voice_map.get((preferred_language or "en").lower())

    try:
        audio_bytes = await elevenlabs_client.synthesize_audio_bytes(text, voice_id=voice_id)
    except Exception:  # noqa: BLE001
        logger.bind(call_id=call_id).warning("ElevenLabs synthesis failed for Twilio playback")
        return None

    audio_id = ElevenLabsClient.cache_audio_bytes(audio_bytes)
    return f"{settings.public_base_url.rstrip('/')}/api/v1/voice/cache/{audio_id}"


@router.post("/webhook/incoming", name="twilio_incoming_call")
async def incoming_call(
    request: Request,
    _: None = Depends(rate_limit_webhook),
    settings: Settings = Depends(get_settings_dep),
    engine: ConversationEngine = Depends(get_conversation_engine),
    twilio_client: TwilioClient = Depends(get_twilio_client),
    elevenlabs_client: ElevenLabsClient = Depends(get_elevenlabs_client),
    backend_client: BackendClient = Depends(get_backend_client),
    agent_id: str | None = Query(default=None),
    CallSid: str = Form(""),
    From: str = Form(""),
    To: str = Form(""),
    SpeechResult: str = Form(""),
) -> Response:
    call_id = CallSid.strip() or "unknown-call"
    action_url = _action_url(request, settings)

    try:
        form_data = {k: str(v) for k, v in (await request.form()).items()}
        if settings.twilio_validate_signature:
            signature = request.headers.get("X-Twilio-Signature", "")
            valid = twilio_client.verify_signature(
                auth_token=settings.twilio_auth_token,
                signature=signature,
                url=_signed_url(request, settings),
                params=form_data,
            )
            if not valid:
                logger.bind(call_id=call_id).warning("Rejected Twilio webhook: invalid signature")
                return Response(content="Invalid Twilio signature", status_code=403)

        speech_text = SpeechResult.strip()
        logger.bind(call_id=call_id).info(
            "Twilio inbound webhook",
            has_speech=bool(speech_text),
            from_number=From,
            to_number=To,
        )

        if not await engine.has_session(call_id):
            resolved_agent_id = agent_id or await backend_client.resolve_agent_id_for_inbound(To)
            greeting = await engine.start_session(
                call_id=call_id,
                agent_id=resolved_agent_id,
                caller_number=From,
                called_number=To,
            )

            if not speech_text:
                play_url = await _build_play_url(
                    text=greeting["text"],
                    call_id=call_id,
                    settings=settings,
                    engine=engine,
                    elevenlabs_client=elevenlabs_client,
                )
                twiml = twilio_client.build_gather_twiml(
                    text=greeting["text"],
                    action_url=action_url,
                    play_url=play_url,
                )
                return Response(content=twiml, media_type=TWIML_CONTENT_TYPE)

        if not speech_text:
            play_url = await _build_play_url(
                text="I did not catch that. Please go ahead.",
                call_id=call_id,
                settings=settings,
                engine=engine,
                elevenlabs_client=elevenlabs_client,
            )
            twiml = twilio_client.build_gather_twiml(
                text="I did not catch that. Please go ahead.",
                action_url=action_url,
                play_url=play_url,
            )
            return Response(content=twiml, media_type=TWIML_CONTENT_TYPE)

        action = await engine.process_user_input(call_id=call_id, transcribed_text=speech_text, agent_id=agent_id)
        play_url = await _build_play_url(
            text=str(action.get("text_to_speak") or ""),
            call_id=call_id,
            settings=settings,
            engine=engine,
            elevenlabs_client=elevenlabs_client,
        )
        twiml = twilio_client.build_action_twiml(action=action, action_url=action_url, play_url=play_url)

        if action.get("action") == "hangup":
            await engine.end_call(call_id)

        return Response(content=twiml, media_type=TWIML_CONTENT_TYPE)
    except Exception:  # noqa: BLE001
        logger.bind(call_id=call_id).exception("Twilio incoming webhook failed")
        play_url = await _build_play_url(
            text="Sorry, something went wrong. Please try again.",
            call_id=call_id,
            settings=settings,
            engine=engine,
            elevenlabs_client=elevenlabs_client,
        )
        fallback = twilio_client.build_gather_twiml(
            text="Sorry, something went wrong. Please try again.",
            action_url=action_url,
            play_url=play_url,
        )
        return Response(content=fallback, media_type=TWIML_CONTENT_TYPE)


@router.post("/webhook/status")
async def call_status(
    _: Request,
    __: None = Depends(rate_limit_webhook),
    engine: ConversationEngine = Depends(get_conversation_engine),
    CallSid: str = Form(""),
    CallStatus: str = Form(""),
    CallDuration: str = Form(""),
) -> dict[str, str]:
    status = CallStatus.strip().lower()
    if status in TERMINAL_STATUSES:
        await engine.end_call(CallSid)
        logger.bind(call_id=CallSid).info("Twilio call ended", status=status, duration=CallDuration)
    return {"status": status or "unknown"}


@router.post("/webhook/diagnostic")
async def diagnostic_call(
    twilio_client: TwilioClient = Depends(get_twilio_client),
) -> Response:
    twiml = twilio_client.build_diagnostic_twiml(
        "This is a Twilio diagnostic call. If you hear this message, call audio is working.",
    )
    return Response(content=twiml, media_type=TWIML_CONTENT_TYPE)
