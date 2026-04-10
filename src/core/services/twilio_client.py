"""Twilio telephony client for TwiML generation and webhook verification."""

from __future__ import annotations

import base64
import hashlib
import hmac
from html import escape
from typing import Any
from urllib.parse import quote

try:
    from twilio.base.exceptions import TwilioRestException
    from twilio.rest import Client
except ImportError:  # pragma: no cover - exercised only when dependency missing locally
    Client = None  # type: ignore[assignment]

    class TwilioRestException(Exception):
        pass


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

    def _sdk_client(self) -> Client:
        if Client is None:
            raise ValueError("twilio package is not installed")
        if not self.account_sid or not self.auth_token:
            raise ValueError("Twilio credentials are not configured")
        return Client(self.account_sid, self.auth_token)

    @staticmethod
    def _normalize_number_type(number_type: str) -> str:
        normalized = number_type.strip().lower().replace("-", "_")
        aliases = {
            "local": "local",
            "tollfree": "toll_free",
            "toll_free": "toll_free",
            "mobile": "mobile",
        }
        if normalized not in aliases:
            raise ValueError("number_type must be one of: local, toll_free, mobile")
        return aliases[normalized]

    def search_available_numbers(
        self,
        *,
        country_code: str,
        number_type: str,
        limit: int = 1,
        area_code: int | None = None,
        contains: str | None = None,
    ) -> list[dict[str, Any]]:
        client = self._sdk_client()
        normalized_type = self._normalize_number_type(number_type)
        resource = getattr(client.available_phone_numbers(country_code.upper()), normalized_type, None)
        if resource is None:
            raise ValueError(f"Unsupported number type for country {country_code}: {number_type}")

        params: dict[str, Any] = {"voice_enabled": True, "limit": limit}
        if area_code is not None:
            params["area_code"] = area_code
        if contains:
            params["contains"] = contains

        try:
            matches = resource.list(**params)
        except TwilioRestException as exc:
            raise ValueError(exc.msg or "Twilio rejected the number search request") from exc

        return [
            {
                "phone_number": match.phone_number,
                "friendly_name": getattr(match, "friendly_name", "") or match.phone_number,
                "locality": getattr(match, "locality", None),
                "region": getattr(match, "region", None),
                "iso_country": getattr(match, "iso_country", country_code.upper()),
                "capabilities": getattr(match, "capabilities", {}) or {},
            }
            for match in matches
        ]

    def provision_incoming_number(
        self,
        *,
        agent_id: str,
        public_base_url: str,
        country_code: str,
        number_type: str,
        area_code: int | None = None,
        contains: str | None = None,
        phone_number: str | None = None,
        friendly_name: str | None = None,
        address_sid: str | None = None,
        bundle_sid: str | None = None,
        identity_sid: str | None = None,
    ) -> dict[str, Any]:
        client = self._sdk_client()
        normalized_type = self._normalize_number_type(number_type)

        chosen_number = phone_number
        if not chosen_number:
            matches = self.search_available_numbers(
                country_code=country_code,
                number_type=normalized_type,
                limit=1,
                area_code=area_code,
                contains=contains,
            )
            if not matches:
                raise ValueError("No voice-capable phone numbers are currently available for the requested criteria")
            chosen_number = str(matches[0]["phone_number"])

        base = public_base_url.rstrip("/")
        encoded_agent_id = quote(agent_id, safe="")
        voice_url = f"{base}/api/v1/twilio/webhook/incoming?agent_id={encoded_agent_id}"
        status_callback = f"{base}/api/v1/twilio/webhook/status?agent_id={encoded_agent_id}"

        create_params: dict[str, Any] = {
            "phone_number": chosen_number,
            "voice_url": voice_url,
            "voice_method": "POST",
            "status_callback": status_callback,
            "status_callback_method": "POST",
            "friendly_name": friendly_name or f"agent:{agent_id}:{chosen_number}",
        }
        if address_sid:
            create_params["address_sid"] = address_sid
        if bundle_sid:
            create_params["bundle_sid"] = bundle_sid
        if identity_sid:
            create_params["identity_sid"] = identity_sid

        try:
            purchased = client.incoming_phone_numbers.create(**create_params)
        except TwilioRestException as exc:
            raise ValueError(exc.msg or "Twilio rejected the number provisioning request") from exc

        return {
            "agent_id": agent_id,
            "phone_number_sid": purchased.sid,
            "phone_number": purchased.phone_number,
            "friendly_name": getattr(purchased, "friendly_name", create_params["friendly_name"]),
            "voice_url": getattr(purchased, "voice_url", voice_url),
            "status_callback": getattr(purchased, "status_callback", status_callback),
            "capabilities": getattr(purchased, "capabilities", {}) or {},
            "country_code": country_code.upper(),
            "number_type": normalized_type,
            "account_sid": getattr(purchased, "account_sid", self.account_sid),
        }

    def update_incoming_number_binding(
        self,
        *,
        phone_number_sid: str,
        agent_id: str,
        public_base_url: str,
        friendly_name: str | None = None,
    ) -> dict[str, Any]:
        client = self._sdk_client()
        base = public_base_url.rstrip("/")
        encoded_agent_id = quote(agent_id, safe="")
        voice_url = f"{base}/api/v1/twilio/webhook/incoming?agent_id={encoded_agent_id}"
        status_callback = f"{base}/api/v1/twilio/webhook/status?agent_id={encoded_agent_id}"

        update_params: dict[str, Any] = {
            "voice_url": voice_url,
            "voice_method": "POST",
            "status_callback": status_callback,
            "status_callback_method": "POST",
        }
        if friendly_name:
            update_params["friendly_name"] = friendly_name

        try:
            updated = client.incoming_phone_numbers(phone_number_sid).update(**update_params)
        except TwilioRestException as exc:
            raise ValueError(exc.msg or "Twilio rejected the number rebind request") from exc

        return {
            "agent_id": agent_id,
            "phone_number_sid": updated.sid,
            "phone_number": updated.phone_number,
            "friendly_name": getattr(updated, "friendly_name", friendly_name or updated.phone_number),
            "voice_url": getattr(updated, "voice_url", voice_url),
            "status_callback": getattr(updated, "status_callback", status_callback),
            "capabilities": getattr(updated, "capabilities", {}) or {},
            "account_sid": getattr(updated, "account_sid", self.account_sid),
        }

    def release_incoming_number(self, *, phone_number_sid: str) -> None:
        client = self._sdk_client()
        try:
            client.incoming_phone_numbers(phone_number_sid).delete()
        except TwilioRestException as exc:
            raise ValueError(exc.msg or "Twilio rejected the number release request") from exc

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
        play_url: str | None = None,
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
        if play_url:
            lines.append(f'    <Play>{self._safe_text(play_url)}</Play>')
        elif safe_text:
            lines.append(f'    <Say voice="{voice}">{safe_text}</Say>')
        lines.append("  </Gather>")
        lines.append(f'  <Say voice="{voice}">I did not catch that. Please try again.</Say>')
        lines.append(f"  <Redirect>{action_url}</Redirect>")
        lines.append("</Response>")
        return "\n".join(lines)

    def build_hangup_twiml(self, text: str = "", voice: str = "Polly.Joanna", play_url: str | None = None) -> str:
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<Response>"]
        if play_url:
            lines.append(f'  <Play>{self._safe_text(play_url)}</Play>')
        elif text:
            lines.append(f'  <Say voice="{voice}">{self._safe_text(text)}</Say>')
        lines.append("  <Hangup/>")
        lines.append("</Response>")
        return "\n".join(lines)

    def build_transfer_twiml(
        self,
        text: str,
        transfer_number: str,
        voice: str = "Polly.Joanna",
        play_url: str | None = None,
    ) -> str:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<Response>",
        ]
        if play_url:
            lines.append(f'  <Play>{self._safe_text(play_url)}</Play>')
        else:
            lines.append(f'  <Say voice="{voice}">{self._safe_text(text)}</Say>')
        lines.extend([
            f"  <Dial>{transfer_number}</Dial>",
            "</Response>",
        ])
        return "\n".join(lines)

    def build_action_twiml(
        self,
        action: dict[str, str | None],
        action_url: str,
        voice: str = "Polly.Joanna",
        play_url: str | None = None,
    ) -> str:
        action_type = action.get("action", "speak")
        text = action.get("text_to_speak", "")
        transfer_number = action.get("transfer_number")

        if action_type == "hangup":
            return self.build_hangup_twiml(text or "", voice=voice, play_url=play_url)
        if action_type == "transfer" and transfer_number:
            return self.build_transfer_twiml(text or "", transfer_number, voice=voice, play_url=play_url)
        return self.build_gather_twiml(text or "", action_url, voice=voice, play_url=play_url)

    def build_diagnostic_twiml(
        self,
        text: str,
        voice: str = "Polly.Joanna",
    ) -> str:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<Response>",
            f'  <Say voice="{voice}">{self._safe_text(text)}</Say>',
            "  <Pause length=\"1\"/>",
            f'  <Say voice="{voice}">{self._safe_text(text)}</Say>',
            "  <Hangup/>",
            "</Response>",
        ]
        return "\n".join(lines)

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
