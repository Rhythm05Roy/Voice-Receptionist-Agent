import json
import re
from typing import Any, Sequence

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
                temperature=0.55,
                max_tokens=320,
            )
            reply = completion.choices[0].message.content or ""
            logger.debug("LLM reply generated", model=self.model)
            return reply.strip()
        except OpenAIError as exc:
            logger.exception("OpenAI call failed")
            raise ConversationEngineError(str(exc)) from exc

    async def _generate_json(self, messages: Sequence[dict[str, str]]) -> dict[str, Any]:
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=0,
            max_tokens=320,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        return json.loads(raw)

    async def detect_call_intent(self, user_input: str, context: dict[str, Any] | None = None) -> dict[str, str]:
        prompt = (
            "Classify the caller request and return strict JSON with keys: intent, booking_id, confidence. "
            "intent must be one of: new_booking, track_booking, business_info, pricing_info, transfer_request, end_call, unclear. "
            "Use business_info when caller asks what services, business details, policies, or general help. "
            "Use new_booking only when caller clearly wants to book now. "
            "Use track_booking when caller asks status/reschedule/cancel for an existing booking. "
            "Extract booking_id when present (example DUMMY-10001)."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "text": user_input,
                        "context": context or {},
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            parsed = await self._generate_json(messages)
            intent = str(parsed.get("intent", "unclear")).strip().lower()
            if intent not in {
                "new_booking",
                "track_booking",
                "business_info",
                "pricing_info",
                "transfer_request",
                "end_call",
                "unclear",
            }:
                intent = "unclear"
            return {
                "intent": intent,
                "booking_id": str(parsed.get("booking_id", "")).strip(),
                "confidence": str(parsed.get("confidence", "medium")).strip().lower() or "medium",
            }
        except Exception:  # noqa: BLE001
            return self._heuristic_intent(user_input)

    async def analyze_turn(
        self,
        question: str,
        user_input: str,
        collected_answers: dict[str, str],
        question_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = (
            "You route one caller turn during intake. Return strict JSON with keys: "
            "intent, extracted_answer, assistant_reply, normalized_answer. "
            "intent must be one of: answer, info_request, unclear, off_topic, cancel. "
            "Use answer only if user answered the current intake question. "
            "If caller asks a side question, use info_request and provide short assistant_reply. "
            "If caller response is unrelated/noise, use off_topic."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "question_meta": question_meta or {},
                        "user_input": user_input,
                        "collected_answers": collected_answers,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            parsed = await self._generate_json(messages)
            intent = str(parsed.get("intent", "unclear")).strip().lower()
            if intent not in {"answer", "info_request", "unclear", "off_topic", "cancel"}:
                intent = "unclear"
            return {
                "intent": intent,
                "extracted_answer": str(parsed.get("extracted_answer", "")).strip(),
                "normalized_answer": str(parsed.get("normalized_answer", "")).strip(),
                "assistant_reply": str(parsed.get("assistant_reply", "")).strip(),
            }
        except Exception:  # noqa: BLE001
            cleaned = user_input.strip()
            if cleaned.lower() in {"cancel", "stop", "hang up", "end call"}:
                return {
                    "intent": "cancel",
                    "extracted_answer": "",
                    "normalized_answer": "",
                    "assistant_reply": "Understood. I can end the call now.",
                }
            if len(cleaned) < 2:
                return {
                    "intent": "unclear",
                    "extracted_answer": "",
                    "normalized_answer": "",
                    "assistant_reply": "Could you please repeat that a bit more clearly?",
                }
            return {
                "intent": "answer",
                "extracted_answer": cleaned,
                "normalized_answer": cleaned,
                "assistant_reply": "",
            }

    async def detect_language_preference(
        self,
        user_input: str,
        supported_languages: list[str],
        default_language: str,
    ) -> str:
        prompt = (
            "Return strict JSON with key language. Choose only one code from supported_languages. "
            "Detect language from user text and explicit requests like 'speak in English'."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "text": user_input,
                        "supported_languages": supported_languages,
                        "default_language": default_language,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            parsed = await self._generate_json(messages)
            detected = str(parsed.get("language", default_language)).strip().lower()
            if detected in supported_languages:
                return detected
        except Exception:  # noqa: BLE001
            pass

        if re.search(r"\benglish\b", user_input.lower()):
            return "en" if "en" in supported_languages else default_language
        if re.search(r"\barabic\b", user_input.lower()) and "ar" in supported_languages:
            return "ar"
        if any("\u0600" <= ch <= "\u06FF" for ch in user_input) and "ar" in supported_languages:
            return "ar"
        return default_language

    async def rewrite_confirmation(self, text: str, caller_language_hint: str = "en") -> str:
        prompt = (
            "Rewrite this confirmation for phone call speech. Keep it short, clear, and human. "
            "Do not add any facts."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"language": caller_language_hint, "text": text},
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            return await self.generate_reply(messages)
        except Exception:  # noqa: BLE001
            return text

    def _heuristic_intent(self, user_input: str) -> dict[str, str]:
        text = user_input.lower().strip()
        booking_id_match = re.search(r"\b([a-z]{2,}-\d{4,})\b", text)
        booking_id = booking_id_match.group(1).upper() if booking_id_match else ""

        if any(word in text for word in ["bye", "goodbye", "end call", "hang up", "thank you bye"]):
            return {"intent": "end_call", "booking_id": booking_id, "confidence": "high"}
        if any(word in text for word in ["track", "booking status", "booking id", "reschedule", "cancel booking"]):
            return {"intent": "track_booking", "booking_id": booking_id, "confidence": "high"}
        if any(word in text for word in ["transfer", "human", "agent", "representative"]):
            return {"intent": "transfer_request", "booking_id": booking_id, "confidence": "medium"}
        if any(word in text for word in ["price", "pricing", "cost", "rate", "quote"]):
            return {"intent": "pricing_info", "booking_id": booking_id, "confidence": "medium"}
        if any(word in text for word in ["service", "services", "business", "about", "company"]):
            return {"intent": "business_info", "booking_id": booking_id, "confidence": "medium"}
        if any(word in text for word in ["book", "appointment", "visit", "need service"]):
            return {"intent": "new_booking", "booking_id": booking_id, "confidence": "medium"}
        return {"intent": "unclear", "booking_id": booking_id, "confidence": "low"}
