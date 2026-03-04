from pydantic import BaseModel, ConfigDict, Field


class ServiceInfo(BaseModel):
    service_id: str
    name: str
    description: str
    base_price_bhd: str
    price_note: str = ""
    keywords: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    agent_id: str = Field(...)
    business_name: str = Field(default="Local Home Services Bahrain")
    greeting: str = Field(default="Hello, how can I help you today?")
    intake_questions: list[str] = Field(default_factory=list)
    language: str = Field(default="ar")
    fallback_phone: str | None = Field(default=None)
    coverage_country: str = Field(default="Bahrain")
    coverage_areas: list[str] = Field(default_factory=list)
    service_catalog: list[ServiceInfo] = Field(default_factory=list)
    booking_required_fields: list[str] = Field(default_factory=list)
    faqs: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class IntakeQuestion(BaseModel):
    question: str
    key: str


class TTSResult(BaseModel):
    audio_url: str
    text: str
