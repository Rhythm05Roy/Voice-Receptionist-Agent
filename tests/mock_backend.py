from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(title="Mock Backend (Test Only)", version="0.1.0")


class BookingRequest(BaseModel):
    answers: dict[str, str] = Field(default_factory=dict)


MOCK_AGENTS: dict[str, dict[str, Any]] = {
    "default": {
        "id": "default",
        "greeting": "Hello, thank you for calling. How can I help you today?",
        "intake_questions": [
            "What service do you need today?",
            "Which area are you located in?",
            "What time works best for a visit?",
        ],
        "language": "en",
        "fallback_phone": None,
    },
    "bahrain-demo": {
        "id": "bahrain-demo",
        "greeting": "مرحبا، هلا فيك. شلون اقدر اخدمك اليوم؟",
        "intake_questions": [
            "شنو نوع الخدمة المطلوبة؟",
            "وين موقعكم في البحرين؟",
            "اي وقت يناسبكم للزيارة؟",
        ],
        "language": "ar",
        "fallback_phone": "+97317000000",
    },
}

BOOKINGS: list[dict[str, Any]] = []


def _expected_token() -> str:
    return os.getenv("BACKEND_API_KEY", "change-me")


def _validate_auth(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    token = authorization.split(" ", 1)[1].strip()
    if token != _expected_token():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/agents/{agent_id}/voice-config")
async def voice_config(
    agent_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _validate_auth(authorization)
    return MOCK_AGENTS.get(agent_id, MOCK_AGENTS["default"])


@app.get("/agents/{agent_id}/intake-questions")
async def intake_questions(
    agent_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, list[str]]:
    _validate_auth(authorization)
    agent = MOCK_AGENTS.get(agent_id, MOCK_AGENTS["default"])
    return {"questions": list(agent.get("intake_questions", []))}


@app.post("/agents/{agent_id}/bookings")
async def create_booking(
    agent_id: str,
    payload: BookingRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _validate_auth(authorization)
    booking = {
        "booking_id": str(uuid.uuid4()),
        "agent_id": agent_id,
        "answers": payload.answers,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    BOOKINGS.append(booking)
    return {
        "status": "confirmed",
        "message": "Your request is registered. Our team will contact you shortly.",
        "booking": booking,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tests.mock_backend:app", host="127.0.0.1", port=9000, reload=True)
