from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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


class AgentPhoneNumberProvisionRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, description="Agent or business identifier to bind to the number")
    country_code: str = Field(default="CA", min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code")
    number_type: Literal["local", "toll_free", "mobile"] = Field(default="local")
    area_code: int | None = Field(default=None, ge=100, le=999)
    contains: str | None = Field(default=None, min_length=2, max_length=16)
    phone_number: str | None = Field(default=None, description="Exact E.164 number to purchase if already selected")
    friendly_name: str | None = Field(default=None, min_length=3, max_length=64)
    address_sid: str | None = Field(default=None, description="Twilio AddressSid for regulated geographies")
    bundle_sid: str | None = Field(default=None, description="Twilio BundleSid for regulated geographies")
    identity_sid: str | None = Field(default=None, description="Twilio IdentitySid if required by regulation")


class AgentPhoneNumberSearchRequest(BaseModel):
    country_code: str = Field(default="CA", min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code")
    number_type: Literal["local", "toll_free", "mobile"] = Field(default="local")
    area_code: int | None = Field(default=None, ge=100, le=999)
    contains: str | None = Field(default=None, min_length=2, max_length=16)
    limit: int = Field(default=10, ge=1, le=20)


class AgentPhoneNumberAssignmentRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)


class AgentPhoneNumberProvisionResponse(BaseModel):
    agent_id: str
    phone_number_sid: str
    phone_number: str
    friendly_name: str
    voice_url: str
    status_callback: str
    capabilities: dict[str, Any] = Field(default_factory=dict)
    country_code: str
    number_type: str
    account_sid: str


class AgentPhoneNumberSearchItem(BaseModel):
    phone_number: str
    friendly_name: str
    locality: str | None = None
    region: str | None = None
    iso_country: str
    capabilities: dict[str, Any] = Field(default_factory=dict)


class AgentCallForwardingRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, description="Agent or business identifier")
    forwarding_number: str = Field(..., min_length=7, description="Phone number used for live human transfer")


class AgentCallForwardingResponse(BaseModel):
    agent_id: str
    forwarding_number: str
    assigned_phone_number: str | None = None


class AgentPhoneNumberAssignmentResponse(BaseModel):
    agent_id: str
    phone_number: str | None = None
    phone_number_sid: str | None = None
    friendly_name: str | None = None
    forwarding_number: str | None = None


class AgentPhoneNumberRebindRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    phone_number_sid: str = Field(..., min_length=3)
    phone_number: str = Field(..., min_length=7)
    friendly_name: str | None = Field(default=None, min_length=3, max_length=64)


class AgentPhoneNumberRebindResponse(BaseModel):
    agent_id: str
    phone_number_sid: str
    phone_number: str
    friendly_name: str
    voice_url: str
    status_callback: str
    capabilities: dict[str, Any] = Field(default_factory=dict)
    account_sid: str


class AgentPhoneNumberReleaseRequest(BaseModel):
    agent_id: str | None = None
    phone_number: str | None = None
    phone_number_sid: str | None = None
    release_provider_number: bool = True

    @model_validator(mode="after")
    def validate_identifier(self) -> "AgentPhoneNumberReleaseRequest":
        if not self.agent_id and not self.phone_number and not self.phone_number_sid:
            raise ValueError("Provide agent_id, phone_number, or phone_number_sid to release a number")
        return self


class AgentPhoneNumberReleaseResponse(BaseModel):
    status: str
    agent_id: str | None = None
    phone_number: str | None = None
    phone_number_sid: str | None = None
    forwarding_number: str | None = None


class AgentCallReportRequest(BaseModel):
    call_id: str = Field(..., min_length=1)


class AgentCallReportPushResponse(BaseModel):
    status: str
    detail: str
    external_call_id: str | None = None
    provider_response: dict[str, Any] = Field(default_factory=dict)


class AgentCallReportResponse(BaseModel):
    call_id: str
    agent_id: str
    business_name: str | None = None
    summary: str = ""
    action_required: bool = False
    action_type: str | None = None
    booking_status: str | None = None
    final_disposition: str = "completed"
    customer_details: dict[str, Any] = Field(default_factory=dict)
    order_or_booked_service: dict[str, Any] = Field(default_factory=dict)
    call_analytics: dict[str, Any] = Field(default_factory=dict)
    transcript: list[dict[str, str]] = Field(default_factory=list)


class AgentTestVoiceStartRequest(BaseModel):
    agent_id: str | None = None


class AgentTestVoiceTurnRequest(BaseModel):
    session_id: str
    text: str = Field(..., min_length=1)
    agent_id: str | None = None


class AgentTestVoiceResponse(BaseModel):
    session_id: str
    text: str
    audio_url: str | None = None
    action: str = "speak"
    is_active: bool = True
    transfer_number: str | None = None
