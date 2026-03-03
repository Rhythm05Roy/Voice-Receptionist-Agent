from functools import lru_cache
from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = Field(default="development", alias="ENVIRONMENT", description="Environment name")
    api_v1_prefix: str = Field(default="/api/v1")

    backend_base_url: AnyHttpUrl = Field(default="http://localhost:9000", alias="BACKEND_BASE_URL")
    backend_api_key: str = Field(default="dev-backend-key", alias="BACKEND_API_KEY")

    openai_api_key: str = Field(default="dev-openai-key", alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    elevenlabs_api_key: str = Field(default="dev-elevenlabs-key", alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="dev-voice-id", alias="ELEVENLABS_VOICE_ID")

    assemblyai_api_key: str = Field(default="dev-assemblyai-key", alias="ASSEMBLYAI_API_KEY")

    vonage_api_key: str = Field(default="dev-vonage-key", alias="VONAGE_API_KEY")
    vonage_api_secret: str = Field(default="dev-vonage-secret", alias="VONAGE_API_SECRET")
    vonage_application_id: str = Field(default="dev-vonage-app", alias="VONAGE_APPLICATION_ID")
    vonage_private_key: str = Field(default="dev-vonage-private-key", alias="VONAGE_PRIVATE_KEY")

    request_timeout: int = Field(default=15, description="HTTP client timeout seconds", alias="REQUEST_TIMEOUT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    local_test_mode: bool = Field(default=False, alias="LOCAL_TEST_MODE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]
