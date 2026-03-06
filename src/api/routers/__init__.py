from fastapi import APIRouter

from .agent import router as agent_router
from .telephony import router as telephony_router
from .twilio_webhooks import router as twilio_router
from .voice import router as voice_router

api_router = APIRouter()
api_router.include_router(agent_router)
api_router.include_router(telephony_router)
api_router.include_router(twilio_router)
api_router.include_router(voice_router)

__all__ = ["api_router"]
