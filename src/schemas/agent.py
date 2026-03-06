from typing import Any

from pydantic import BaseModel, Field


class AgentPreviewRequest(BaseModel):
    agent_id: str | None = Field(default=None, description="Agent identifier")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class AgentServiceItem(BaseModel):
    service_id: str
    name: str
    description: str
    base_price_bhd: str
    price_note: str = ""


class AgentUIContextResponse(BaseModel):
    agent_id: str
    business_name: str
    greeting: str
    language: str
    multilingual_enabled: bool = True
    supported_languages: list[str] = Field(default_factory=list)
    default_greeting_language: str = "en"
    max_call_duration_minutes: int = 5
    coverage_country: str
    coverage_areas: list[str] = Field(default_factory=list)
    booking_required_fields: list[str] = Field(default_factory=list)
    fallback_phone: str | None = None
    services: list[AgentServiceItem] = Field(default_factory=list)
    faqs: dict[str, str] = Field(default_factory=dict)
    business_description: str = ""
    business_hours: str = ""
    cancellation_policy: str = ""
    payment_policy: str = ""
    deposit_policy: str = ""
    additional_information: list[str] = Field(default_factory=list)
    intake_questions: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AgentBusinessQueryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    agent_id: str | None = None


class AgentBusinessQueryResponse(BaseModel):
    answer: str
    suggested_services: list[str] = Field(default_factory=list)


class AgentTrackBookingRequest(BaseModel):
    booking_id: str = Field(..., min_length=3)
    agent_id: str | None = None


class AgentTrackBookingResponse(BaseModel):
    status: str
    booking_ref: str
    message: str
    service_name: str | None = None
    location: str | None = None
    preferred_time: str | None = None
