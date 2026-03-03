from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger


class BackendCommunicationError(Exception):
    def __init__(self, message: str = "Failed to communicate with backend"):
        self.message = message
        super().__init__(message)


class VoiceGenerationError(Exception):
    def __init__(self, message: str = "Voice generation failed"):
        self.message = message
        super().__init__(message)


class ConversationEngineError(Exception):
    def __init__(self, message: str = "Conversation engine error"):
        self.message = message
        super().__init__(message)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BackendCommunicationError)
    async def backend_error_handler(_, exc: BackendCommunicationError):
        logger.error(exc.message)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": exc.message},
        )

    @app.exception_handler(VoiceGenerationError)
    async def voice_error_handler(_, exc: VoiceGenerationError):
        logger.error(exc.message)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": exc.message},
        )

    @app.exception_handler(ConversationEngineError)
    async def conversation_error_handler(_, exc: ConversationEngineError):
        logger.error(exc.message)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": exc.message},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_, exc: HTTPException):
        logger.warning(exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
