"""Analytics API — powers Figma dashboard metrics & activity feed."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from src.core.services.call_store import CallStore

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _get_call_store(request: Request) -> CallStore:
    store = getattr(request.app.state, "call_store", None)
    if store is None:
        store = CallStore()
        request.app.state.call_store = store
    return store


class DashboardSummary(BaseModel):
    calls_today: int = 0
    calls_this_week: int = 0
    total_calls: int = 0
    avg_duration_seconds: float = 0.0
    avg_duration_formatted: str = "0:00"
    pending_actions: int = 0


class CallListItem(BaseModel):
    call_id: str
    caller_number: str = ""
    duration_seconds: float = 0.0
    turn_count: int = 0
    outcome: str = "unknown"
    is_test: bool = False
    transcript_preview: str = ""
    time_ago: str = ""


class CallDetail(BaseModel):
    call_id: str
    agent_id: str = ""
    caller_number: str = ""
    duration_seconds: float = 0.0
    turn_count: int = 0
    outcome: str = "unknown"
    is_test: bool = False
    transcript: list[dict[str, str]] = Field(default_factory=list)
    summary: str = ""


@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    agent_id: str | None = None,
    store: CallStore = Depends(_get_call_store),
) -> DashboardSummary:
    """Dashboard metrics: calls today, this week, avg duration."""
    data = store.get_summary(agent_id=agent_id)
    return DashboardSummary(**data)


@router.get("/calls", response_model=list[CallListItem])
async def get_call_list(
    agent_id: str | None = None,
    limit: int = 20,
    store: CallStore = Depends(_get_call_store),
) -> list[CallListItem]:
    """Activity feed — recent call list."""
    records = store.get_calls(agent_id=agent_id, limit=limit)
    return [
        CallListItem(
            call_id=r.call_id,
            caller_number=r.caller_number,
            duration_seconds=r.duration_seconds,
            turn_count=r.turn_count,
            outcome=r.outcome,
            is_test=r.is_test,
            transcript_preview=r.transcript_preview,
            time_ago=r.time_ago,
        )
        for r in records
    ]


@router.get("/call/{call_id}", response_model=CallDetail)
async def get_call_detail(
    call_id: str,
    store: CallStore = Depends(_get_call_store),
) -> CallDetail:
    """Full call detail with transcript."""
    record = store.get_call(call_id)
    if not record:
        return CallDetail(call_id=call_id, outcome="not_found")
    return CallDetail(
        call_id=record.call_id,
        agent_id=record.agent_id,
        caller_number=record.caller_number,
        duration_seconds=record.duration_seconds,
        turn_count=record.turn_count,
        outcome=record.outcome,
        is_test=record.is_test,
        transcript=record.transcript,
        summary=record.summary,
    )
