"""Onboarding API — voice-driven agent setup (Figma: SOW screens)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.deps import get_llm_client
from src.core.conversation.onboarding_engine import OnboardingEngine
from src.core.services.openai import OpenAIClient

router = APIRouter(prefix="/onboarding", tags=["onboarding"])

# Singleton engine — created lazily
_engine: OnboardingEngine | None = None


def _get_engine(llm: OpenAIClient = Depends(get_llm_client)) -> OnboardingEngine:
    global _engine
    if _engine is None:
        _engine = OnboardingEngine(llm_client=llm)
    return _engine


class OnboardingStartRequest(BaseModel):
    session_id: str | None = None


class OnboardingTurnRequest(BaseModel):
    session_id: str
    text: str


class OnboardingResponse(BaseModel):
    session_id: str
    text: str
    step: str
    is_complete: bool = False
    collected_data: dict = Field(default_factory=dict)
    agent_config: dict | None = None


@router.post("/start", response_model=OnboardingResponse)
async def start_onboarding(
    payload: OnboardingStartRequest,
    engine: OnboardingEngine = Depends(_get_engine),
) -> OnboardingResponse:
    """Start a new voice onboarding session."""
    session_id = payload.session_id or str(uuid.uuid4())
    result = await engine.start_session(session_id)
    return OnboardingResponse(
        session_id=session_id,
        text=result["text"],
        step=result["step"],
        is_complete=False,
    )


@router.post("/turn", response_model=OnboardingResponse)
async def onboarding_turn(
    payload: OnboardingTurnRequest,
    engine: OnboardingEngine = Depends(_get_engine),
) -> OnboardingResponse:
    """Process one turn of the onboarding conversation."""
    result = await engine.process_turn(payload.session_id, payload.text)

    agent_config = None
    if result.get("is_complete"):
        agent_config = engine.build_agent_config(payload.session_id)

    return OnboardingResponse(
        session_id=payload.session_id,
        text=result["text"],
        step=result["step"],
        is_complete=result.get("is_complete", False),
        collected_data=result.get("collected_data", {}),
        agent_config=agent_config,
    )


@router.post("/complete")
async def complete_onboarding(
    payload: OnboardingTurnRequest,
    engine: OnboardingEngine = Depends(_get_engine),
) -> dict:
    """Finalize onboarding and return the built agent config."""
    agent_config = engine.build_agent_config(payload.session_id)
    engine.end_session(payload.session_id)
    return {"status": "completed", "agent_config": agent_config}
