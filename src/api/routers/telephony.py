from fastapi import APIRouter, Depends, Request
from loguru import logger

from src.api.deps import get_conversation_engine, get_vonage_client, rate_limit_webhook
from src.core.conversation.engine import ConversationEngine
from src.core.services.vonage import VonageClient
from src.schemas import TelephonyWebhookRequest, TelephonyResponse

router = APIRouter(prefix="/telephony", tags=["telephony"])


@router.post("/webhook/inbound", response_model=TelephonyResponse)
async def inbound_webhook(
    payload: TelephonyWebhookRequest,
    request: Request,
    _: None = Depends(rate_limit_webhook),
    engine: ConversationEngine = Depends(get_conversation_engine),
    vonage_client: VonageClient = Depends(get_vonage_client),
) -> TelephonyResponse:
    try:
        event_url = [str(request.url)]

        if not engine.has_session(payload.uuid):
            greeting = await engine.start_session(call_id=payload.uuid)
            ncco = vonage_client.build_talk_ncco(greeting["text"])
            ncco.append(vonage_client.build_listen_action(event_url=event_url))
            return TelephonyResponse(ncco=ncco)

        speech_text = payload.speech_text
        if speech_text:
            action = await engine.process_user_input(call_id=payload.uuid, transcribed_text=speech_text)
            ncco = vonage_client.build_action_ncco(action, from_number=payload.to_number, event_url=event_url)
            if action.get("action") == "hangup":
                engine.end_call(payload.uuid)
            return TelephonyResponse(ncco=ncco)

        ncco = [
            *vonage_client.build_talk_ncco("Sorry, I didn't catch that. Please try again."),
            vonage_client.build_listen_action(event_url=event_url),
        ]
        return TelephonyResponse(ncco=ncco)

    except Exception:  # noqa: BLE001
        logger.exception("Inbound webhook failed")
        engine.end_call(payload.uuid)
        apology = vonage_client.build_talk_ncco("عذراً، صار خطأ. بنقفل المكالمة الآن.")
        apology.append(vonage_client.build_hangup_ncco())
        return TelephonyResponse(ncco=apology)
