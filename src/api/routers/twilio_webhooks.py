"""Twilio webhook endpoints — incoming calls & status updates.

Supports two modes:
1. **HTTP Gather mode**: Twilio's <Gather input="speech"> for simple webhook-per-turn
2. **Media Streams mode**: WebSocket for real-time bidirectional audio (future)

The HTTP Gather mode provides the simplest Twilio migration path
while still benefiting from the new LLM-driven engine.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import Response
from loguru import logger

from src.api.deps import (
    get_backend_client,
    get_conversation_engine,
    get_twilio_client,
    rate_limit_webhook,
)
from src.core.conversation.engine import ConversationEngine
from src.core.services.backend_client import BackendClient
from src.core.services.twilio_client import TwilioClient

router = APIRouter(prefix="/twilio", tags=["twilio"])

TERMINAL_STATUSES = {"completed", "failed", "busy", "no-answer", "canceled"}
TWIML_CONTENT_TYPE = "application/xml"


@router.post("/webhook/incoming")
async def incoming_call(
    request: Request,
    _: None = Depends(rate_limit_webhook),
    engine: ConversationEngine = Depends(get_conversation_engine),
    twilio_client: TwilioClient = Depends(get_twilio_client),
    backend_client: BackendClient = Depends(get_backend_client),
    CallSid: str = Form(""),
    From: str = Form(""),
    To: str = Form(""),
    SpeechResult: str = Form(""),
    Confidence: str = Form(""),
) -> Response:
    """Handle incoming Twilio calls using <Gather input='speech'>.

    Twilio POSTs form data with CallSid, From, To, and optional
    SpeechResult when speech gathering completes.
    """
    call_id = CallSid
    action_url = str(request.url_for("incoming_call"))

    try:
        logger.bind(call_id=call_id).info(
            "Twilio webhook",
            has_speech=bool(SpeechResult),
            from_number=From,
            to_number=To,
        )

        speech_text = SpeechResult.strip()

        # First hit — no session yet
        if not await engine.has_session(call_id):
            agent_id = await backend_client.resolve_agent_id_for_inbound(To)
            greeting = await engine.start_session(call_id=call_id, agent_id=agent_id)

            if not speech_text:
                twiml = twilio_client.build_gather_twiml(
                    text=greeting["text"],
                    action_url=action_url,
                )
                return Response(content=twiml, media_type=TWIML_CONTENT_TYPE)

        # No speech detected
        if not speech_text:
            twiml = twilio_client.build_gather_twiml(
                text="I'm sorry, I didn't catch that. Please go ahead.",
                action_url=action_url,
            )
            return Response(content=twiml, media_type=TWIML_CONTENT_TYPE)

        # Process speech through the LLM-driven engine
        action = await engine.process_user_input(
            call_id=call_id,
            transcribed_text=speech_text,
        )

        twiml = twilio_client.build_action_twiml(
            action=action,
            action_url=action_url,
        )

        if action.get("action") == "hangup":
            await engine.end_call(call_id)

        return Response(content=twiml, media_type=TWIML_CONTENT_TYPE)

    except Exception:  # noqa: BLE001
        logger.bind(call_id=call_id).exception("Twilio webhook failed")
        twiml = twilio_client.build_gather_twiml(
            text="Sorry, something went wrong. Could you repeat that?",
            action_url=action_url,
        )
        return Response(content=twiml, media_type=TWIML_CONTENT_TYPE)


@router.post("/webhook/status")
async def call_status(
    request: Request,
    _: None = Depends(rate_limit_webhook),
    engine: ConversationEngine = Depends(get_conversation_engine),
    CallSid: str = Form(""),
    CallStatus: str = Form(""),
    CallDuration: str = Form(""),
) -> dict[str, str]:
    """Handle Twilio call status webhooks."""
    status = CallStatus.strip().lower()
    if status in TERMINAL_STATUSES:
        await engine.end_call(CallSid)
        logger.bind(call_id=CallSid).info(
            "Call ended",
            status=status,
            duration=CallDuration,
        )
    return {"status": status or "unknown"}
