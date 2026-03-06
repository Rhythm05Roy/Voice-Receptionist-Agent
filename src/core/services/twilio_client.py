"""Twilio telephony client — TwiML generation & call control.

Replaces the Vonage client for production telephony.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any
from urllib.parse import urlencode

from loguru import logger


class TwilioClient:
    """Generate TwiML responses and manage call control."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        phone_number: str,
        websocket_url: str | None = None,
    ):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.phone_number = phone_number
        self.websocket_url = websocket_url or ""

    def build_media_stream_twiml(
        self,
        websocket_url: str | None = None,
        greeting: str | None = None,
    ) -> str:
        """Build TwiML to connect an incoming call to a WebSocket media stream."""
        ws_url = websocket_url or self.websocket_url
        parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        parts.append("<Response>")

        if greeting:
            # Use Twilio's built-in TTS for the initial greeting (low latency)
            safe_greeting = greeting.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(f'  <Say voice="Polly.Joanna">{safe_greeting}</Say>')

        # Connect to bidirectional WebSocket media stream
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
    ) -> str:
        """Build TwiML for speech-based input gathering (HTTP webhook mode)."""
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return "\n".join([
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<Response>",
            f'  <Gather input="speech" timeout="{timeout}" '
            f'speechTimeout="{speech_timeout}" action="{action_url}" method="POST">',
            f'    <Say voice="{voice}">{safe_text}</Say>',
            "  </Gather>",
            f'  <Say voice="{voice}">I didn\'t catch that. Please try again.</Say>',
            f'  <Redirect>{action_url}</Redirect>',
            "</Response>",
        ])

    def build_say_and_gather_twiml(
        self,
        text: str,
        action_url: str,
        voice: str = "Polly.Joanna",
    ) -> str:
        """Say something and then gather speech input."""
        return self.build_gather_twiml(text, action_url, voice=voice)

    def build_hangup_twiml(self, text: str = "", voice: str = "Polly.Joanna") -> str:
        """Build TwiML to say final message and hang up."""
        parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<Response>"]
        if text:
            safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(f'  <Say voice="{voice}">{safe_text}</Say>')
        parts.append("  <Hangup/>")
        parts.append("</Response>")
        return "\n".join(parts)

    def build_transfer_twiml(
        self,
        text: str,
        transfer_number: str,
        voice: str = "Polly.Joanna",
    ) -> str:
        """Say message then dial a number for transfer."""
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return "\n".join([
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<Response>",
            f'  <Say voice="{voice}">{safe_text}</Say>',
            f"  <Dial>{transfer_number}</Dial>",
            "</Response>",
        ])

    def build_action_twiml(
        self,
        action: dict[str, str | None],
        action_url: str,
        voice: str = "Polly.Joanna",
    ) -> str:
        """Convert a conversation engine action dict into TwiML."""
        action_type = action.get("action", "speak")
        text = action.get("text_to_speak", "")
        transfer_number = action.get("transfer_number")

        if action_type == "hangup":
            return self.build_hangup_twiml(text or "", voice=voice)

        if action_type == "transfer" and transfer_number:
            return self.build_transfer_twiml(text or "", transfer_number, voice=voice)

        return self.build_gather_twiml(text or "", action_url, voice=voice)

    # ── Vonage-compatible bridge methods (for existing telephony router) ──

    def build_talk_ncco(self, text: str, voice_name: str | None = None) -> list[dict[str, Any]]:
        """Backward-compatible NCCO-style talk action."""
        return [{"action": "talk", "text": text, "voiceName": voice_name or "Polly.Joanna"}]

    def build_listen_action(
        self,
        event_url: list[str] | None = None,
        speech_timeout: int = 7,
    ) -> dict[str, Any]:
        """Backward-compatible NCCO-style listen action."""
        return {"action": "listen", "eventUrl": event_url or [], "speechTimeout": speech_timeout}

    def build_hangup_ncco(self) -> dict[str, Any]:
        """Backward-compatible NCCO-style hangup."""
        return {"action": "hangup"}

    def build_action_ncco(
        self,
        action: dict[str, str | None],
        from_number: str | None = None,
        event_url: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert engine action into NCCO-compatible list (for existing webhook flow)."""
        action_type = action.get("action", "speak")
        text = action.get("text_to_speak", "")
        transfer_number = action.get("transfer_number")

        if action_type == "hangup":
            ncco = self.build_talk_ncco(text or "Thank you for calling. Goodbye.")
            ncco.append(self.build_hangup_ncco())
            return ncco

        if action_type == "transfer" and transfer_number:
            ncco = self.build_talk_ncco(text or "Please hold while I transfer you.")
            ncco.append({
                "action": "connect",
                "endpoint": [{"type": "phone", "number": transfer_number}],
            })
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
        """Verify Twilio webhook signature."""
        sorted_params = sorted(params.items())
        data = url + "".join(f"{k}{v}" for k, v in sorted_params)
        expected = hmac.new(
            auth_token.encode("utf-8"),
            data.encode("utf-8"),
            hashlib.sha1,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
