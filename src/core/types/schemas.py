from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    agent_id: str = Field(...)
    greeting: str = Field(default="Hello, how can I help you today?")
    intake_questions: list[str] = Field(default_factory=list)
    language: str = Field(default="ar")
    fallback_phone: str | None = Field(default=None)


class IntakeQuestion(BaseModel):
    question: str
    key: str


class TTSResult(BaseModel):
    audio_url: str
    text: str
