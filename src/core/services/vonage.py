from loguru import logger
from typing import Any


class VonageClient:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        application_id: str,
        private_key: str,
        voice_name: str = "Zeina",
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.application_id = application_id
        self.private_key = private_key
        self.voice_name = voice_name

    def build_talk_ncco(self, text: str, voice_name: str | None = None) -> list[dict[str, str]]:
        voice = voice_name or self.voice_name
        ncco = [
            {
                "action": "talk",
                "text": text,
                "voiceName": voice,
            }
        ]
        logger.debug("Built talk NCCO", ncco=ncco)
        return ncco

    def build_listen_action(self, event_url: list[str] | None = None, speech_timeout: int = 5) -> dict[str, Any]:
        action: dict[str, Any] = {
            "action": "listen",
            "eventUrl": event_url or [],
            "speechTimeout": speech_timeout,
        }
        return action

    def build_connect_ncco(self, number: str, from_number: str | None = None) -> dict[str, object]:
        connect_action = {
            "action": "connect",
            "endpoint": [
                {"type": "phone", "number": number}
            ],
        }
        if from_number:
            connect_action["from"] = from_number
        logger.debug("Built connect NCCO", endpoint=number)
        return connect_action

    def build_hangup_ncco(self) -> dict[str, str]:
        return {"action": "hangup"}

    def build_action_ncco(self, action: dict[str, str | None], from_number: str | None = None, event_url: list[str] | None = None) -> list[dict]:
        act = action.get("action")
        text = action.get("text_to_speak") or ""
        transfer_number = action.get("transfer_number")

        if act == "transfer" and transfer_number:
            ncco = self.build_talk_ncco(text)
            ncco.append(self.build_connect_ncco(transfer_number, from_number=from_number))
            return ncco

        if act == "hangup":
            ncco = self.build_talk_ncco(text or "شكراً لاتصالك.")
            ncco.append(self.build_hangup_ncco())
            return ncco

        ncco = self.build_talk_ncco(text)
        ncco.append(self.build_listen_action(event_url=event_url))
        return ncco
