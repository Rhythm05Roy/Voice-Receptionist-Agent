from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QuestionAnswerType = Literal["text", "yes_no", "multiple_choice", "number"]
QuestionAskWhen = Literal["all_bookings", "specific_services"]
DisqualificationAction = Literal["hangup", "transfer"]


class DisqualificationRule(BaseModel):
    if_answer: str
    message_to_caller: str
    action: DisqualificationAction = "hangup"
    transfer_number: str | None = None

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def normalize_rule(cls, value: Any) -> Any:
        if isinstance(value, dict):
            raw = dict(value)
            if raw.get("disqualifying_value") and not raw.get("if_answer"):
                raw["if_answer"] = raw["disqualifying_value"]
            return raw
        return value


class IntakeQuestion(BaseModel):
    key: str
    question: str
    answer_type: QuestionAnswerType = "text"
    required: bool = True
    ask_when: QuestionAskWhen = "all_bookings"
    service_tags: list[str] = Field(default_factory=list)
    specific_categories: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    disqualification_rules: list[DisqualificationRule] = Field(default_factory=list)
    retry_prompt: str | None = None
    validation_regex: str | None = None
    is_active: bool = True

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def normalize_question(cls, value: Any) -> Any:
        if isinstance(value, str):
            cleaned = value.strip() or "Could you share details?"
            return {"key": cls._slug(cleaned), "question": cleaned}
        if isinstance(value, dict):
            raw = dict(value)
            question = raw.get("question") or raw.get("question_text") or raw.get("text")
            if question:
                raw["question"] = str(question)
            key = raw.get("key") or raw.get("id")
            if not key and question:
                raw["key"] = cls._slug(str(question))
            answer_type = raw.get("answer_type") or raw.get("answerType")
            if answer_type:
                normalized = str(answer_type).strip().lower().replace(" ", "_").replace("/", "_")
                if normalized in {"yes", "no", "yes_no", "yesno"}:
                    normalized = "yes_no"
                raw["answer_type"] = normalized
            ask_when = raw.get("ask_when") or raw.get("when_to_ask")
            if ask_when:
                text = str(ask_when).strip().lower().replace(" ", "_")
                if "specific" in text:
                    text = "specific_services"
                elif "tagged" in text:
                    text = "specific_services"
                elif "all" in text:
                    text = "all_bookings"
                raw["ask_when"] = text
            tags = raw.get("service_tags") or raw.get("serviceTags") or raw.get("specific_services")
            if tags is not None:
                raw["service_tags"] = list(tags)
            categories = raw.get("specific_categories")
            if categories is not None:
                raw["specific_categories"] = [str(item) for item in categories]
            rules = raw.get("disqualification_rules") or raw.get("disqualification")
            if rules is not None:
                raw["disqualification_rules"] = list(rules)
            if "is_required" in raw and "required" not in raw:
                raw["required"] = raw["is_required"]
            return raw
        return value

    @staticmethod
    def _slug(text: str) -> str:
        clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
        while "__" in clean:
            clean = clean.replace("__", "_")
        return clean[:48] or "question"


class ServiceInfo(BaseModel):
    service_id: str
    name: str
    description: str
    base_price_bhd: str = Field(default="", description="Legacy field, use base_price instead")
    base_price: str = Field(default="", description="Price range or base price")
    currency: str = Field(default="", description="Currency code, e.g. USD, BHD, EUR")
    price_note: str = ""
    keywords: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def backfill_price(self) -> ServiceInfo:
        """Ensure base_price is populated from legacy field if needed."""
        if not self.base_price and self.base_price_bhd:
            self.base_price = self.base_price_bhd
        if not self.base_price_bhd and self.base_price:
            self.base_price_bhd = self.base_price
        return self


class AgentConfig(BaseModel):
    agent_id: str = Field(...)
    business_name: str = Field(default="Local Home Services Bahrain")
    greeting: str = Field(default="Hello, how can I help you today?")
    intake_questions: list[IntakeQuestion] = Field(default_factory=list)
    language: str = Field(default="en")
    multilingual_enabled: bool = Field(default=True)
    supported_languages: list[str] = Field(default_factory=lambda: ["en", "ar"])
    default_greeting_language: str = Field(default="en")
    language_voice_map: dict[str, str] = Field(default_factory=dict)
    fallback_phone: str | None = Field(default=None)
    max_call_duration_minutes: int = Field(default=15, ge=1, le=60)
    coverage_country: str = Field(default="")
    coverage_areas: list[str] = Field(default_factory=list)
    excluded_areas: list[str] = Field(default_factory=list, description="Country/area names that are out of coverage")
    service_catalog: list[ServiceInfo] = Field(default_factory=list)
    booking_required_fields: list[str] = Field(default_factory=list)
    faqs: dict[str, str] = Field(default_factory=dict)
    business_description: str = Field(default="")
    business_hours: str = Field(default="")
    cancellation_policy: str = Field(default="")
    payment_policy: str = Field(default="")
    deposit_policy: str = Field(default="")
    additional_information: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @field_validator("intake_questions", mode="before")
    @classmethod
    def normalize_intake_questions(cls, value: Any) -> list[IntakeQuestion]:
        if value is None:
            return []
        if isinstance(value, list):
            return [IntakeQuestion.model_validate(item) for item in value]
        return [IntakeQuestion.model_validate(value)]

    @field_validator("supported_languages", mode="before")
    @classmethod
    def normalize_supported_languages(cls, value: Any) -> list[str]:
        if value is None:
            return ["en", "ar"]
        if isinstance(value, str):
            candidates = [part.strip() for part in value.split(",")]
        else:
            candidates = [str(part).strip() for part in value]
        cleaned = [code.lower() for code in candidates if code]
        return cleaned or ["en", "ar"]


class TTSResult(BaseModel):
    audio_url: str
    text: str
