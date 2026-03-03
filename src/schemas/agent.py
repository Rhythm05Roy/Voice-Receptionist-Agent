from pydantic import BaseModel, Field


class AgentPreviewRequest(BaseModel):
    agent_id: str | None = Field(default=None, description="Agent identifier")

    model_config = {"populate_by_name": True, "extra": "ignore"}
