import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from pydantic import BaseModel

from src.api.deps import get_backend_client, get_conversation_engine, get_elevenlabs_client, get_settings_dep, get_twilio_client
from src.config import Settings
from src.core.conversation.engine import ConversationEngine
from src.core.services.backend_client import BackendClient
from src.core.services.elevenlabs import ElevenLabsClient
from src.core.services.twilio_client import TwilioClient
from src.schemas import (
    AgentBusinessQueryRequest,
    AgentBusinessQueryResponse,
    AgentCallForwardingRequest,
    AgentCallForwardingResponse,
    AgentCallReportRequest,
    AgentCallReportPushResponse,
    AgentCallReportResponse,
    AgentPhoneNumberAssignmentRequest,
    AgentPhoneNumberAssignmentResponse,
    AgentPhoneNumberProvisionRequest,
    AgentPhoneNumberProvisionResponse,
    AgentPhoneNumberSearchItem,
    AgentPhoneNumberSearchRequest,
    AgentPhoneNumberRebindRequest,
    AgentPhoneNumberRebindResponse,
    AgentPhoneNumberReleaseRequest,
    AgentPhoneNumberReleaseResponse,
    AgentPreviewRequest,
    AgentTrackBookingRequest,
    AgentTrackBookingResponse,
    AgentTestVoiceResponse,
    AgentTestVoiceStartRequest,
    AgentTestVoiceTurnRequest,
    AgentUIContextResponse,
    TTSResponse,
)

router = APIRouter(prefix="/agent", tags=["agent"])


async def _run_phone_number_search(
    *,
    country_code: str,
    number_type: str,
    area_code: int | None,
    contains: str | None,
    limit: int,
    twilio_client: TwilioClient,
) -> list[AgentPhoneNumberSearchItem]:
    if not twilio_client.credentials_available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Twilio credentials are not configured for number search.",
        )
    if number_type != "local" and area_code is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="area_code is only supported for local number searches.",
        )

    normalized_country = country_code.upper()
    search_variants: list[dict[str, str | int | None]] = [
        {
            "country_code": normalized_country,
            "number_type": number_type,
            "limit": limit,
            "area_code": area_code,
            "contains": contains,
        }
    ]
    if contains:
        search_variants.append(
            {
                "country_code": normalized_country,
                "number_type": number_type,
                "limit": limit,
                "area_code": area_code,
                "contains": None,
            }
        )
    if area_code is not None:
        search_variants.append(
            {
                "country_code": normalized_country,
                "number_type": number_type,
                "limit": limit,
                "area_code": None,
                "contains": None,
            }
        )

    try:
        matches: list[dict[str, str | dict[str, Any] | None]] = []
        for variant in search_variants:
            matches = await run_in_threadpool(
                twilio_client.search_available_numbers,
                country_code=str(variant["country_code"]),
                number_type=str(variant["number_type"]),
                limit=int(variant["limit"] or limit),
                area_code=variant["area_code"] if isinstance(variant["area_code"], int) else None,
                contains=str(variant["contains"]) if variant["contains"] else None,
            )
            if matches:
                if variant["area_code"] != area_code or variant["contains"] != contains:
                    logger.info(
                        "Phone number search used broader fallback",
                        country_code=normalized_country,
                        number_type=number_type,
                        requested_area_code=area_code,
                        requested_contains=contains,
                        returned_count=len(matches),
                    )
                break
    except ValueError as exc:
        logger.warning("Phone number search rejected", reason=str(exc), country_code=country_code, number_type=number_type)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Phone number search failed", country_code=country_code, number_type=number_type)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to search available phone numbers from Twilio.",
        ) from exc

    return [AgentPhoneNumberSearchItem(**item) for item in matches]


# ── Existing Endpoints ───────────────────────────────────────────

@router.post("/preview", response_model=TTSResponse)
async def preview_agent_greeting(
    payload: AgentPreviewRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> TTSResponse:
    call_id = f"preview-{uuid.uuid4()}"
    result = await engine.start_session(call_id=call_id, agent_id=payload.agent_id)
    await engine.end_call(call_id)
    return TTSResponse(audio_url=result["audio_url"], text=result["text"])


@router.get("/context", response_model=AgentUIContextResponse)
async def get_agent_context(
    agent_id: str | None = None,
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentUIContextResponse:
    payload = await backend_client.fetch_agent_ui_context(agent_id=agent_id)
    return AgentUIContextResponse(**payload)


@router.post("/query", response_model=AgentBusinessQueryResponse)
async def answer_agent_business_query(
    payload: AgentBusinessQueryRequest,
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentBusinessQueryResponse:
    result = await backend_client.answer_business_query(query=payload.text, agent_id=payload.agent_id)
    return AgentBusinessQueryResponse(**result)


@router.post("/track-booking", response_model=AgentTrackBookingResponse)
async def track_booking(
    payload: AgentTrackBookingRequest,
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentTrackBookingResponse:
    result = await backend_client.track_booking(payload.booking_id, agent_id=payload.agent_id)
    return AgentTrackBookingResponse(**result)


@router.post("/phone-numbers/search", response_model=list[AgentPhoneNumberSearchItem])
async def search_phone_numbers(
    payload: AgentPhoneNumberSearchRequest,
    twilio_client: TwilioClient = Depends(get_twilio_client),
) -> list[AgentPhoneNumberSearchItem]:
    return await _run_phone_number_search(
        country_code=payload.country_code,
        number_type=payload.number_type,
        area_code=payload.area_code,
        contains=payload.contains,
        limit=payload.limit,
        twilio_client=twilio_client,
    )


@router.get("/phone-numbers/search", response_model=list[AgentPhoneNumberSearchItem])
async def search_phone_numbers_legacy(
    country_code: str = Query(default="CA", min_length=2, max_length=2),
    number_type: str = Query(default="local"),
    area_code: int | None = Query(default=None, ge=100, le=999),
    contains: str | None = Query(default=None, min_length=2, max_length=16),
    limit: int = Query(default=10, ge=1, le=20),
    twilio_client: TwilioClient = Depends(get_twilio_client),
) -> list[AgentPhoneNumberSearchItem]:
    return await _run_phone_number_search(
        country_code=country_code,
        number_type=number_type,
        area_code=area_code,
        contains=contains,
        limit=limit,
        twilio_client=twilio_client,
    )


@router.post("/call-report", response_model=AgentCallReportResponse)
async def get_call_report(
    payload: AgentCallReportRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> AgentCallReportResponse:
    report = await engine.build_call_report(payload.call_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call report not found.")
    return AgentCallReportResponse(**report)


@router.post("/call-report/push", response_model=AgentCallReportPushResponse)
async def push_call_report(
    payload: AgentCallReportRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentCallReportPushResponse:
    report = await engine.build_call_report(payload.call_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call report not found.")

    try:
        provider_response = await backend_client.post_call_report(report)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to push call report", call_id=payload.call_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to push call report to backend.",
        ) from exc

    return AgentCallReportPushResponse(
        status=str(provider_response.get("status", "unknown")),
        detail=str(provider_response.get("detail", "Call report delivered.")),
        external_call_id=provider_response.get("external_call_id"),
        provider_response=provider_response,
    )


@router.post("/phone-numbers/provision", response_model=AgentPhoneNumberProvisionResponse)
async def provision_agent_phone_number(
    payload: AgentPhoneNumberProvisionRequest,
    settings: Settings = Depends(get_settings_dep),
    twilio_client: TwilioClient = Depends(get_twilio_client),
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentPhoneNumberProvisionResponse:
    if not twilio_client.credentials_available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Twilio credentials are not configured for number provisioning.",
        )
    if not settings.public_base_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PUBLIC_BASE_URL must be configured before provisioning phone numbers.",
        )
    if payload.number_type != "local" and payload.area_code is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="area_code is only supported for local number searches.",
        )

    try:
        provisioned = await run_in_threadpool(
            twilio_client.provision_incoming_number,
            agent_id=payload.agent_id,
            public_base_url=settings.public_base_url,
            country_code=payload.country_code,
            number_type=payload.number_type,
            area_code=payload.area_code,
            contains=payload.contains,
            phone_number=payload.phone_number,
            friendly_name=payload.friendly_name,
            address_sid=payload.address_sid,
            bundle_sid=payload.bundle_sid,
            identity_sid=payload.identity_sid,
        )
    except ValueError as exc:
        logger.warning("Phone number provisioning rejected", reason=str(exc), agent_id=payload.agent_id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Phone number provisioning failed", agent_id=payload.agent_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to provision phone number from Twilio.",
        ) from exc

    logger.info(
        "Provisioned phone number",
        agent_id=payload.agent_id,
        phone_number=provisioned["phone_number"],
        phone_number_sid=provisioned["phone_number_sid"],
    )
    backend_client.bind_phone_number(
        agent_id=payload.agent_id,
        phone_number=provisioned["phone_number"],
        phone_number_sid=provisioned["phone_number_sid"],
        friendly_name=provisioned["friendly_name"],
    )
    return AgentPhoneNumberProvisionResponse(**provisioned)


@router.post("/phone-numbers/forwarding", response_model=AgentCallForwardingResponse)
async def configure_call_forwarding(
    payload: AgentCallForwardingRequest,
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentCallForwardingResponse:
    forwarding = backend_client.set_call_forwarding(
        agent_id=payload.agent_id,
        forwarding_number=payload.forwarding_number,
    )
    assignment = backend_client.get_phone_assignment(payload.agent_id) or {}
    logger.info(
        "Configured call forwarding",
        agent_id=payload.agent_id,
        forwarding_number=payload.forwarding_number,
        assigned_phone_number=assignment.get("phone_number"),
    )
    return AgentCallForwardingResponse(
        agent_id=forwarding["agent_id"],
        forwarding_number=forwarding["forwarding_number"],
        assigned_phone_number=assignment.get("phone_number"),
    )


async def _get_phone_number_assignment_response(
    *,
    agent_id: str,
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentPhoneNumberAssignmentResponse:
    assignment = backend_client.get_phone_assignment(agent_id)
    if not assignment:
        return AgentPhoneNumberAssignmentResponse(agent_id=agent_id)
    return AgentPhoneNumberAssignmentResponse(agent_id=agent_id, **assignment)


@router.post("/phone-numbers/assignment", response_model=AgentPhoneNumberAssignmentResponse)
async def get_phone_number_assignment_post(
    payload: AgentPhoneNumberAssignmentRequest,
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentPhoneNumberAssignmentResponse:
    return await _get_phone_number_assignment_response(agent_id=payload.agent_id, backend_client=backend_client)


@router.get("/phone-numbers/assignment", response_model=AgentPhoneNumberAssignmentResponse)
async def get_phone_number_assignment(
    agent_id: str,
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentPhoneNumberAssignmentResponse:
    return await _get_phone_number_assignment_response(agent_id=agent_id, backend_client=backend_client)


@router.post("/phone-numbers/rebind", response_model=AgentPhoneNumberRebindResponse)
async def rebind_phone_number(
    payload: AgentPhoneNumberRebindRequest,
    settings: Settings = Depends(get_settings_dep),
    twilio_client: TwilioClient = Depends(get_twilio_client),
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentPhoneNumberRebindResponse:
    if not twilio_client.credentials_available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Twilio credentials are not configured for number rebinding.",
        )
    if not settings.public_base_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PUBLIC_BASE_URL must be configured before rebinding phone numbers.",
        )

    try:
        rebound = await run_in_threadpool(
            twilio_client.update_incoming_number_binding,
            phone_number_sid=payload.phone_number_sid,
            agent_id=payload.agent_id,
            public_base_url=settings.public_base_url,
            friendly_name=payload.friendly_name,
        )
    except ValueError as exc:
        logger.warning("Phone number rebind rejected", reason=str(exc), agent_id=payload.agent_id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Phone number rebind failed", agent_id=payload.agent_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to rebind phone number in Twilio.",
        ) from exc

    backend_client.rebind_phone_number(
        agent_id=payload.agent_id,
        phone_number=payload.phone_number,
        phone_number_sid=payload.phone_number_sid,
        friendly_name=rebound["friendly_name"],
    )
    logger.info(
        "Rebound phone number",
        agent_id=payload.agent_id,
        phone_number=payload.phone_number,
        phone_number_sid=payload.phone_number_sid,
    )
    return AgentPhoneNumberRebindResponse(
        agent_id=payload.agent_id,
        phone_number_sid=payload.phone_number_sid,
        phone_number=payload.phone_number,
        friendly_name=rebound["friendly_name"],
        voice_url=rebound["voice_url"],
        status_callback=rebound["status_callback"],
        capabilities=rebound["capabilities"],
        account_sid=rebound["account_sid"],
    )


@router.post("/phone-numbers/release", response_model=AgentPhoneNumberReleaseResponse)
async def release_phone_number(
    payload: AgentPhoneNumberReleaseRequest,
    twilio_client: TwilioClient = Depends(get_twilio_client),
    backend_client: BackendClient = Depends(get_backend_client),
) -> AgentPhoneNumberReleaseResponse:
    existing = None
    if payload.agent_id:
        existing = backend_client.get_phone_assignment(payload.agent_id)
    phone_number_sid = payload.phone_number_sid or (existing or {}).get("phone_number_sid")

    if payload.release_provider_number:
        if not twilio_client.credentials_available:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Twilio credentials are not configured for number release.",
            )
        if not phone_number_sid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="phone_number_sid is required to release the provider number.",
            )
        try:
            await run_in_threadpool(twilio_client.release_incoming_number, phone_number_sid=phone_number_sid)
        except ValueError as exc:
            logger.warning("Phone number release rejected", reason=str(exc), phone_number_sid=phone_number_sid)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Phone number release failed", phone_number_sid=phone_number_sid)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to release phone number from Twilio.",
            ) from exc

    released = backend_client.release_phone_number(
        agent_id=payload.agent_id,
        phone_number=payload.phone_number,
        phone_number_sid=phone_number_sid,
    )
    if not released:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No matching phone-number assignment found.")

    logger.info(
        "Released phone number",
        agent_id=released.get("agent_id"),
        phone_number=released.get("phone_number"),
        phone_number_sid=released.get("phone_number_sid"),
    )
    return AgentPhoneNumberReleaseResponse(
        status="released",
        agent_id=released.get("agent_id"),
        phone_number=released.get("phone_number"),
        phone_number_sid=released.get("phone_number_sid"),
        forwarding_number=released.get("forwarding_number"),
    )


# ── Agent Test Endpoints (Figma: "Talk Your Agent" button) ───────


class TestStartRequest(BaseModel):
    agent_id: str | None = None


class TestTurnRequest(BaseModel):
    session_id: str
    text: str
    agent_id: str | None = None


class TestResponse(BaseModel):
    session_id: str
    text: str
    action: str = "speak"
    is_active: bool = True


@router.post("/test-start", response_model=TestResponse)
async def test_start(
    payload: TestStartRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> TestResponse:
    """Start a test conversation with the agent. Returns greeting."""
    session_id = f"test-{uuid.uuid4()}"
    result = await engine.start_session(
        call_id=session_id,
        agent_id=payload.agent_id,
        is_test=True,
    )
    return TestResponse(
        session_id=session_id,
        text=result["text"],
        action="speak",
        is_active=True,
    )


@router.post("/test-voice/start", response_model=AgentTestVoiceResponse)
async def test_voice_start(
    payload: AgentTestVoiceStartRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> AgentTestVoiceResponse:
    """Start a direct non-telephony voice test session."""
    session_id = f"voice-test-{uuid.uuid4()}"
    result = await engine.start_session(
        call_id=session_id,
        agent_id=payload.agent_id,
        is_test=True,
    )
    return AgentTestVoiceResponse(
        session_id=session_id,
        text=result["text"],
        audio_url=result.get("audio_url"),
        action="speak",
        is_active=True,
        transfer_number=None,
    )


@router.post("/test-turn", response_model=TestResponse)
async def test_turn(
    payload: TestTurnRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> TestResponse:
    """Send a message in a test conversation."""
    if not await engine.has_session(payload.session_id):
        # Auto-start if session expired
        await engine.start_session(
            call_id=payload.session_id,
            agent_id=payload.agent_id,
            is_test=True,
        )

    result = await engine.process_user_input(
        call_id=payload.session_id,
        transcribed_text=payload.text,
        agent_id=payload.agent_id,
    )

    is_active = result.get("action") != "hangup"

    return TestResponse(
        session_id=payload.session_id,
        text=result["text_to_speak"],
        action=result["action"],
        is_active=is_active,
    )


@router.post("/test-voice/turn", response_model=AgentTestVoiceResponse)
async def test_voice_turn(
    payload: AgentTestVoiceTurnRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
    elevenlabs_client: ElevenLabsClient = Depends(get_elevenlabs_client),
) -> AgentTestVoiceResponse:
    """Process a direct text turn and return synthesized audio without Twilio."""
    if not await engine.has_session(payload.session_id):
        await engine.start_session(
            call_id=payload.session_id,
            agent_id=payload.agent_id,
            is_test=True,
        )

    result = await engine.process_user_input(
        call_id=payload.session_id,
        transcribed_text=payload.text,
        agent_id=payload.agent_id,
    )
    is_active = result.get("action") != "hangup"

    audio_url: str | None = None
    text_to_speak = result.get("text_to_speak") or ""
    voice_id: str | None = None
    session_getter = getattr(engine, "sessions", None)
    if session_getter is not None and hasattr(session_getter, "get"):
        session = await engine.sessions.get(payload.session_id)
        if session is not None:
            preferred_language = (session.preferred_language or session.agent_config.default_greeting_language or "en").lower()
            voice_id = session.agent_config.language_voice_map.get(preferred_language)
    if text_to_speak:
        try:
            audio_url = await elevenlabs_client.synthesize_text(text_to_speak, voice_id=voice_id)
        except Exception:  # noqa: BLE001
            logger.warning("Direct test voice synthesis failed", session_id=payload.session_id)

    return AgentTestVoiceResponse(
        session_id=payload.session_id,
        text=text_to_speak,
        audio_url=audio_url,
        action=result.get("action", "speak"),
        is_active=is_active,
        transfer_number=result.get("transfer_number"),
    )


@router.post("/test-end")
async def test_end(
    payload: TestTurnRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> dict:
    """End a test conversation and return summary."""
    await engine.end_call(payload.session_id)
    return {
        "status": "ended",
        "session_id": payload.session_id,
    }


@router.post("/test-voice/end")
async def test_voice_end(
    payload: AgentTestVoiceTurnRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
) -> dict:
    """End a direct non-telephony voice test session."""
    await engine.end_call(payload.session_id)
    return {
        "status": "ended",
        "session_id": payload.session_id,
    }
