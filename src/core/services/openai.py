import json
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
                temperature=0.6,
                max_tokens=256,
            )
            reply = completion.choices[0].message.content or ""
            logger.debug("LLM reply generated", model=self.model)
            return reply.strip()
        except OpenAIError as exc:
            logger.exception("OpenAI call failed")
            raise ConversationEngineError(str(exc))

    async def _generate_json(self, messages: Sequence[dict[str, str]]) -> dict[str, Any]:
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        return json.loads(raw)

    async def classify_intake_answer(self, question: str, answer: str) -> dict[str, Any]:
        prompt = (
            "You validate a caller answer for an intake question. "
            "Return strict JSON with keys: is_relevant (bool), normalized_answer (string), follow_up (string). "
            "If answer is off-topic or unclear, set is_relevant=false and provide a short follow_up. "
            "Keep follow_up in the same language as the answer."
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"},
        ]
        try:
            parsed = await self._generate_json(messages)
            return {
                "is_relevant": bool(parsed.get("is_relevant", False)),
                "normalized_answer": str(parsed.get("normalized_answer", "")).strip(),
                "follow_up": str(parsed.get("follow_up", "")).strip(),
            }
        except Exception:  # noqa: BLE001
            logger.warning("Falling back to heuristic intake validation")
            cleaned = answer.strip()
            return {
                "is_relevant": len(cleaned) >= 3,
                "normalized_answer": cleaned,
                "follow_up": "Could you please answer that question a bit more clearly?",
            }

    async def rewrite_confirmation(self, text: str, caller_language_hint: str = "en") -> str:
        prompt = (
            "Rewrite this booking confirmation to sound natural and short for phone conversation. "
            "Use the caller language hint when possible. Do not add new facts."
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Language hint: {caller_language_hint}\nText: {text}"},
        ]
        try:
            return await self.generate_reply(messages)
        except Exception:  # noqa: BLE001
            return text

    async def analyze_turn(
        self,
        question: str,
        user_input: str,
        collected_answers: dict[str, str],
    ) -> dict[str, Any]:
        prompt = (
            "You route a phone-call user turn for an intake workflow. "
            "Return strict JSON with keys: intent, extracted_answer, assistant_reply. "
            "intent must be one of: answer, info_request, unclear, cancel. "
            "Use answer only if the user directly answers the current question. "
            "If the user asks a side question, set intent=info_request and provide a short helpful reply. "
            "If unclear/off-topic, set intent=unclear and ask to refocus. "
            "Keep tone concise and natural."
        )
        context = json.dumps(collected_answers, ensure_ascii=False)
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Current question: {question}\n"
                    f"Collected answers: {context}\n"
                    f"User said: {user_input}"
                ),
            },
        ]
        try:
            parsed = await self._generate_json(messages)
            intent = str(parsed.get("intent", "unclear")).strip().lower()
            if intent not in {"answer", "info_request", "unclear", "cancel"}:
                intent = "unclear"
            return {
                "intent": intent,
                "extracted_answer": str(parsed.get("extracted_answer", "")).strip(),
                "assistant_reply": str(parsed.get("assistant_reply", "")).strip(),
            }
        except Exception:  # noqa: BLE001
            cleaned = user_input.strip()
            if cleaned.lower() in {"cancel", "stop", "hang up"}:
                return {"intent": "cancel", "extracted_answer": "", "assistant_reply": "Understood. Ending the call."}
            if len(cleaned) < 3:
                return {
                    "intent": "unclear",
                    "extracted_answer": "",
                    "assistant_reply": "Could you answer that with a bit more detail?",
                }
            return {"intent": "answer", "extracted_answer": cleaned, "assistant_reply": ""}

    async def detect_call_intent(self, user_input: str) -> dict[str, str]:
        prompt = (
            "Classify this caller request. Return strict JSON with keys: intent, booking_id, reply_hint. "
            "intent must be one of: new_booking, track_booking, business_info, pricing_info, unclear. "
            "booking_id should be extracted when present (example DUMMY-1772598927), otherwise empty."
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]
        try:
            parsed = await self._generate_json(messages)
            intent = str(parsed.get("intent", "unclear")).strip().lower()
            if intent not in {"new_booking", "track_booking", "business_info", "pricing_info", "unclear"}:
                intent = "unclear"
            return {
                "intent": intent,
                "booking_id": str(parsed.get("booking_id", "")).strip(),
                "reply_hint": str(parsed.get("reply_hint", "")).strip(),
            }
        except Exception:  # noqa: BLE001
            text = user_input.lower()
            if "booking id" in text or "track" in text or "status" in text:
                return {"intent": "track_booking", "booking_id": "", "reply_hint": ""}
            if any(k in text for k in ["service", "book", "appointment", "visit"]):
                return {"intent": "new_booking", "booking_id": "", "reply_hint": ""}
            if any(k in text for k in ["price", "pricing", "cost", "quote", "rate"]):
                return {"intent": "pricing_info", "booking_id": "", "reply_hint": ""}
            if any(k in text for k in ["business", "company", "about you", "what do you do"]):
                return {"intent": "business_info", "booking_id": "", "reply_hint": ""}
            return {"intent": "unclear", "booking_id": "", "reply_hint": ""}
