"""Twilio webhooks for inbound voice calls."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import Response
from loguru import logger

from src.api.deps import (
    get_backend_client,
    get_conversation_engine,
    get_settings_dep,
    get_twilio_client,
    rate_limit_webhook,
)
from src.config import Settings
from src.core.conversation.engine import ConversationEngine
from src.core.services.backend_client import BackendClient
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
    if settings.public_base_url:
        return f"{settings.public_base_url.rstrip('/')}{request.url.path}"
    return str(request.url_for("twilio_incoming_call"))


@router.post("/webhook/incoming", name="twilio_incoming_call")
async def incoming_call(
    request: Request,
    _: None = Depends(rate_limit_webhook),
    settings: Settings = Depends(get_settings_dep),
    engine: ConversationEngine = Depends(get_conversation_engine),
    twilio_client: TwilioClient = Depends(get_twilio_client),
    backend_client: BackendClient = Depends(get_backend_client),
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
            agent_id = await backend_client.resolve_agent_id_for_inbound(To)
            greeting = await engine.start_session(call_id=call_id, agent_id=agent_id)

            if not speech_text:
                twiml = twilio_client.build_gather_twiml(text=greeting["text"], action_url=action_url)
                return Response(content=twiml, media_type=TWIML_CONTENT_TYPE)

        if not speech_text:
            twiml = twilio_client.build_gather_twiml(
                text="I did not catch that. Please go ahead.",
                action_url=action_url,
            )
            return Response(content=twiml, media_type=TWIML_CONTENT_TYPE)

        action = await engine.process_user_input(call_id=call_id, transcribed_text=speech_text)
        twiml = twilio_client.build_action_twiml(action=action, action_url=action_url)

        if action.get("action") == "hangup":
            await engine.end_call(call_id)

        return Response(content=twiml, media_type=TWIML_CONTENT_TYPE)
    except Exception:  # noqa: BLE001
        logger.bind(call_id=call_id).exception("Twilio incoming webhook failed")
        fallback = twilio_client.build_gather_twiml(
            text="Sorry, something went wrong. Please try again.",
            action_url=action_url,
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
