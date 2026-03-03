from typing import Sequence
from loguru import logger
from openai import AsyncOpenAI, OpenAIError

from src.api.exceptions import ConversationEngineError


class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate_reply(self, messages: Sequence[dict[str, str]]) -> str:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                temperature=0.3,
                max_tokens=256,
            )
            reply = completion.choices[0].message.content or ""
            logger.debug("LLM reply generated", model=self.model)
            return reply.strip()
        except OpenAIError as exc:
            logger.exception("OpenAI call failed")
            raise ConversationEngineError(str(exc))
