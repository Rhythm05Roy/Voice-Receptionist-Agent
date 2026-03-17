"""Call record storage — powers dashboard analytics and activity feed.

In production, this would be backed by Redis or a database.
The in-memory implementation works for development/testing.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class CallRecord(BaseModel):
    call_id: str
    agent_id: str = ""
    caller_number: str = ""
    called_number: str = ""
    started_at: float = Field(default_factory=time.time)
    ended_at: float | None = None
    duration_seconds: float = 0.0
    turn_count: int = 0
    outcome: str = "unknown"  # completed, transferred, abandoned, timeout
    is_test: bool = False
    transcript: list[dict[str, str]] = Field(default_factory=list)
    summary: str = ""
    report_payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def transcript_preview(self) -> str:
        for msg in reversed(self.transcript):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"][:100]
        return "No transcript available"

    @property
    def time_ago(self) -> str:
        if not self.ended_at:
            return "ongoing"
        elapsed = time.time() - self.ended_at
        if elapsed < 60:
            return "just now"
        if elapsed < 3600:
            return f"{int(elapsed / 60)} minutes ago"
        if elapsed < 86400:
            return f"{int(elapsed / 3600)} hours ago"
        return f"{int(elapsed / 86400)} days ago"


class CallStore:
    """In-memory call record store."""

    def __init__(self) -> None:
        self._records: dict[str, CallRecord] = {}

    def start_call(
        self,
        call_id: str,
        agent_id: str = "",
        caller_number: str = "",
        called_number: str = "",
        is_test: bool = False,
    ) -> CallRecord:
        record = self._records.get(call_id)
        if record is None:
            record = CallRecord(
                call_id=call_id,
                agent_id=agent_id,
                caller_number=caller_number,
                called_number=called_number,
                is_test=is_test,
            )
            self._records[call_id] = record
        else:
            if agent_id:
                record.agent_id = agent_id
            if caller_number:
                record.caller_number = caller_number
            if called_number:
                record.called_number = called_number
            record.is_test = is_test
        return record

    def end_call(
        self,
        call_id: str,
        outcome: str = "completed",
        transcript: list[dict[str, str]] | None = None,
        turn_count: int = 0,
        report_payload: dict[str, Any] | None = None,
    ) -> CallRecord | None:
        record = self._records.get(call_id)
        if not record:
            return None
        record.ended_at = time.time()
        record.duration_seconds = record.ended_at - record.started_at
        record.outcome = outcome
        record.turn_count = turn_count
        if transcript:
            record.transcript = transcript
        if report_payload:
            record.report_payload = report_payload
        return record

    def get_call(self, call_id: str) -> CallRecord | None:
        return self._records.get(call_id)

    def get_calls(
        self,
        agent_id: str | None = None,
        limit: int = 20,
        include_test: bool = True,
    ) -> list[CallRecord]:
        results = list(self._records.values())
        if agent_id:
            results = [r for r in results if r.agent_id == agent_id]
        if not include_test:
            results = [r for r in results if not r.is_test]
        results.sort(key=lambda r: r.started_at, reverse=True)
        return results[:limit]

    def get_summary(self, agent_id: str | None = None) -> dict[str, Any]:
        """Dashboard summary — calls today, this week, avg duration."""
        now = time.time()
        day_start = now - 86400
        week_start = now - 86400 * 7

        calls = self.get_calls(agent_id=agent_id, limit=1000, include_test=False)

        calls_today = [c for c in calls if c.started_at >= day_start]
        calls_week = [c for c in calls if c.started_at >= week_start]

        completed = [c for c in calls if c.duration_seconds > 0]
        avg_duration = (
            sum(c.duration_seconds for c in completed) / len(completed)
            if completed
            else 0.0
        )

        pending = [c for c in calls if c.outcome == "callback_requested"]

        return {
            "calls_today": len(calls_today),
            "calls_this_week": len(calls_week),
            "total_calls": len(calls),
            "avg_duration_seconds": round(avg_duration, 1),
            "avg_duration_formatted": _format_duration(avg_duration),
            "pending_actions": len(pending),
        }


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"0:{int(seconds):02d}"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"
