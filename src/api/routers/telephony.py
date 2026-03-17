from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from loguru import logger

from src.api.deps import (
    get_backend_client,
    get_conversation_engine,
    get_vonage_client,
    rate_limit_webhook,
)
from src.core.conversation.engine import ConversationEngine
from src.core.services.backend_client import BackendClient
from src.core.services.vonage import VonageClient
from src.schemas import TelephonyEventRequest, TelephonyResponse, TelephonyWebhookRequest

router = APIRouter(prefix="/telephony", tags=["telephony"])

TERMINAL_STATUSES = {"completed", "failed", "timeout", "rejected", "busy", "cancelled"}


class _CallRuntimeStores:
    def __init__(self, request: Request):
        self.request = request

    @property
    def no_input(self) -> dict[str, int]:
        store = getattr(self.request.app.state, "no_input_store", None)
        if store is None:
            store = {}
            self.request.app.state.no_input_store = store
        return store

    @property
    def error_count(self) -> dict[str, int]:
        store = getattr(self.request.app.state, "error_count_store", None)
        if store is None:
            store = {}
            self.request.app.state.error_count_store = store
        return store

    def clear(self, call_id: str) -> None:
        self.no_input.pop(call_id, None)
        self.error_count.pop(call_id, None)


@router.post("/webhook/inbound", response_model=TelephonyResponse)
async def inbound_webhook(
    payload: TelephonyWebhookRequest,
    request: Request,
    _: None = Depends(rate_limit_webhook),
    engine: ConversationEngine = Depends(get_conversation_engine),
    vonage_client: VonageClient = Depends(get_vonage_client),
    backend_client: BackendClient = Depends(get_backend_client),
) -> TelephonyResponse:
    stores = _CallRuntimeStores(request)
    call_id = payload.uuid
    event_url = [str(request.url)]

    try:
        logger.bind(call_id=call_id).info(
            "Inbound webhook",
            has_speech=bool(payload.speech_text),
            from_number=payload.from_number,
            to_number=payload.to_number,
        )

        agent_id = await backend_client.resolve_agent_id_for_inbound(payload.to_number)
        speech_text = (payload.speech_text or "").strip()

        if speech_text:
            stores.no_input.pop(call_id, None)
        else:
            stores.no_input[call_id] = stores.no_input.get(call_id, 0) + 1

        if not await engine.has_session(call_id):
            greeting = await engine.start_session(
                call_id=call_id,
                agent_id=agent_id,
                caller_number=payload.from_number,
                called_number=payload.to_number,
            )
            if not speech_text:
                stores.error_count.pop(call_id, None)
                ncco = vonage_client.build_talk_ncco(greeting["text"])
                ncco.append(vonage_client.build_listen_action(event_url=event_url, speech_timeout=7))
                return TelephonyResponse(ncco=ncco)

        if not speech_text:
            no_input_count = stores.no_input.get(call_id, 0)
            if no_input_count >= 3:
                await engine.end_call(call_id)
                stores.clear(call_id)
                ncco = vonage_client.build_talk_ncco("I still cannot hear you. Please call again when ready.")
                ncco.append(vonage_client.build_hangup_ncco())
                return TelephonyResponse(ncco=ncco)

            ncco = vonage_client.build_talk_ncco("Sorry, I did not catch that. Please continue.")
            ncco.append(vonage_client.build_listen_action(event_url=event_url, speech_timeout=7))
            return TelephonyResponse(ncco=ncco)

        action = await engine.process_user_input(
            call_id=call_id,
            transcribed_text=speech_text,
            agent_id=agent_id,
        )
        ncco = vonage_client.build_action_ncco(action, from_number=payload.to_number, event_url=event_url)

        if action.get("action") == "hangup":
            await engine.end_call(call_id)
            stores.clear(call_id)
        else:
            stores.error_count.pop(call_id, None)

        return TelephonyResponse(ncco=ncco)

    except Exception:  # noqa: BLE001
        logger.bind(call_id=call_id).exception("Inbound webhook failed")
        errors = stores.error_count.get(call_id, 0) + 1
        stores.error_count[call_id] = errors

        if errors >= 2:
            await engine.end_call(call_id)
            stores.clear(call_id)
            fail_ncco = vonage_client.build_talk_ncco("Sorry, something went wrong. Please call us again.")
            fail_ncco.append(vonage_client.build_hangup_ncco())
            return TelephonyResponse(ncco=fail_ncco)

        retry_ncco = vonage_client.build_talk_ncco("Sorry, I did not catch that. Please try again.")
        retry_ncco.append(vonage_client.build_listen_action(event_url=event_url, speech_timeout=7))
        return TelephonyResponse(ncco=retry_ncco)


@router.post("/webhook/event")
async def call_event_webhook(
    payload: TelephonyEventRequest,
    request: Request,
    _: None = Depends(rate_limit_webhook),
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> dict[str, str]:
    status = (payload.status or "").strip().lower()
    if status in TERMINAL_STATUSES:
        await engine.end_call(payload.uuid)
        stores = _CallRuntimeStores(request)
        stores.clear(payload.uuid)

    return {"ok": "true", "status": status or "unknown"}
