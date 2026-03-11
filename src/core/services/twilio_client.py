"""Twilio telephony client for TwiML generation and webhook verification."""

from __future__ import annotations

import base64
import hashlib
import hmac
from html import escape
from typing import Any


class TwilioClient:
    """Generate TwiML responses and helper actions for voice calls."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        phone_number: str,
        websocket_url: str | None = None,
    ) -> None:
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.phone_number = phone_number
        self.websocket_url = websocket_url or ""

    @staticmethod
    def _safe_text(text: str) -> str:
        return escape(text or "", quote=False)

    @property
    def credentials_available(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.phone_number)

    def build_media_stream_twiml(
        self,
        websocket_url: str | None = None,
        greeting: str | None = None,
    ) -> str:
        ws_url = websocket_url or self.websocket_url
        parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<Response>"]
        if greeting:
            parts.append(f'  <Say voice="Polly.Joanna">{self._safe_text(greeting)}</Say>')
        parts.append("  <Connect>")
        parts.append(f'    <Stream url="{ws_url}" />')
        parts.append("  </Connect>")
        parts.append("</Response>")
        return "\n".join(parts)

    def build_gather_twiml(
        self,
        text: str,
        action_url: str,
        timeout: int = 5,
        speech_timeout: str = "auto",
        voice: str = "Polly.Joanna",
        language: str = "en-US",
    ) -> str:
        safe_text = self._safe_text(text)
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<Response>"]
        lines.append(
            '  <Gather input="speech" '
            f'timeout="{timeout}" '
            f'speechTimeout="{speech_timeout}" '
            f'action="{action_url}" '
            'method="POST" '
            'actionOnEmptyResult="true" '
            f'language="{language}" '
            'speechModel="phone_call">'
        )
        if safe_text:
            lines.append(f'    <Say voice="{voice}">{safe_text}</Say>')
        lines.append("  </Gather>")
        lines.append(f'  <Say voice="{voice}">I did not catch that. Please try again.</Say>')
        lines.append(f"  <Redirect>{action_url}</Redirect>")
        lines.append("</Response>")
        return "\n".join(lines)

    def build_hangup_twiml(self, text: str = "", voice: str = "Polly.Joanna") -> str:
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<Response>"]
        if text:
            lines.append(f'  <Say voice="{voice}">{self._safe_text(text)}</Say>')
        lines.append("  <Hangup/>")
        lines.append("</Response>")
        return "\n".join(lines)

    def build_transfer_twiml(
        self,
        text: str,
        transfer_number: str,
        voice: str = "Polly.Joanna",
    ) -> str:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<Response>",
            f'  <Say voice="{voice}">{self._safe_text(text)}</Say>',
            f"  <Dial>{transfer_number}</Dial>",
            "</Response>",
        ]
        return "\n".join(lines)

    def build_action_twiml(
        self,
        action: dict[str, str | None],
        action_url: str,
        voice: str = "Polly.Joanna",
    ) -> str:
        action_type = action.get("action", "speak")
        text = action.get("text_to_speak", "")
        transfer_number = action.get("transfer_number")

        if action_type == "hangup":
            return self.build_hangup_twiml(text or "", voice=voice)
        if action_type == "transfer" and transfer_number:
            return self.build_transfer_twiml(text or "", transfer_number, voice=voice)
        return self.build_gather_twiml(text or "", action_url, voice=voice)

    # Backward-compatible NCCO bridge methods used by /telephony routes.
    def build_talk_ncco(self, text: str, voice_name: str | None = None) -> list[dict[str, Any]]:
        return [{"action": "talk", "text": text, "voiceName": voice_name or "Polly.Joanna"}]

    def build_listen_action(
        self,
        event_url: list[str] | None = None,
        speech_timeout: int = 7,
    ) -> dict[str, Any]:
        return {"action": "listen", "eventUrl": event_url or [], "speechTimeout": speech_timeout}

    def build_hangup_ncco(self) -> dict[str, Any]:
        return {"action": "hangup"}

    def build_action_ncco(
        self,
        action: dict[str, str | None],
        from_number: str | None = None,
        event_url: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        action_type = action.get("action", "speak")
        text = action.get("text_to_speak", "")
        transfer_number = action.get("transfer_number")

        if action_type == "hangup":
            ncco = self.build_talk_ncco(text or "Thank you for calling. Goodbye.")
            ncco.append(self.build_hangup_ncco())
            return ncco

        if action_type == "transfer" and transfer_number:
            ncco = self.build_talk_ncco(text or "Please hold while I transfer you.")
            ncco.append(
                {
                    "action": "connect",
                    "endpoint": [{"type": "phone", "number": transfer_number}],
                }
            )
            return ncco

        ncco = self.build_talk_ncco(text or "")
        ncco.append(self.build_listen_action(event_url=event_url))
        return ncco

    @staticmethod
    def verify_signature(
        auth_token: str,
        signature: str,
        url: str,
        params: dict[str, str],
    ) -> bool:
        """Validate X-Twilio-Signature for application/x-www-form-urlencoded webhooks."""
        if not auth_token or not signature:
            return False
        ordered = sorted((k, str(v)) for k, v in params.items())
        data = url + "".join(f"{k}{v}" for k, v in ordered)
        digest = hmac.new(auth_token.encode("utf-8"), data.encode("utf-8"), hashlib.sha1).digest()
        expected = base64.b64encode(digest).decode("utf-8")
        return hmac.compare_digest(expected, signature.strip())
