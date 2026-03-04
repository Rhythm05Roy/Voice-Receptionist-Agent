from __future__ import annotations

import difflib
import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.core.services.backend_client import BackendClient
from src.core.services.openai import OpenAIClient
from src.core.types import AgentConfig

from . import prompts

if TYPE_CHECKING:
    from .engine import CallSession


def _contains_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def _detect_language(text: str, fallback: str = "en") -> str:
    if not text.strip():
        return fallback
    return "ar" if _contains_arabic(text) else "en"


def _cancel_terms() -> tuple[str, ...]:
    return (
        "cancel",
        "stop",
        "not interested",
        "later",
        "\u0645\u0634\u063a\u0648\u0644",
        "\u0627\u0644\u063a\u0627\u0621",
        "\u0625\u0644\u063a\u0627\u0621",
        "\u0648\u0642\u0641",
        "\u0633\u0643\u0631",
    )


def _is_unknown(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {
        "i don't know",
        "idk",
        "not sure",
        "\u0645\u0627 \u0627\u062f\u0631\u064a",
        "\u0645\u062f\u0631\u064a",
    }


def _language_hint(text: str) -> str:
    return "ar" if _contains_arabic(text) else "en"


def _localize(language: str, en_text: str, ar_text: str) -> str:
    return ar_text if language == "ar" else en_text


def _is_out_of_coverage(answer: str) -> bool:
    lowered = answer.lower()
    in_bahrain_terms = {
        "bahrain",
        "bahrian",
        "barain",
        "baran",
        "bahran",
        "bahrein",
        "bh",
        "\u0627\u0644\u0628\u062d\u0631\u064a\u0646",
        "\u0627\u0644\u0645\u0646\u0627\u0645\u0629",
        "manama",
        "muharraq",
        "riffa",
        "hamad",
    }
    if any(term in lowered for term in in_bahrain_terms):
        return False

    likely_foreign = {
        "bangladesh",
        "dhaka",
        "india",
        "pakistan",
        "saudi",
        "ksa",
        "qatar",
        "uae",
        "oman",
        "kuwait",
        "egypt",
        "canada",
        "uk",
        "usa",
    }
    return any(term in lowered for term in likely_foreign)


def _is_business_info_query(text: str) -> bool:
    lowered = text.lower()
    triggers = [
        "know about your",
        "about your",
        "what kind of business",
        "about your business",
        "about your company",
        "what services",
        "which services",
        "services you provide",
        "do you provide",
        "service details",
    ]
    return any(t in lowered for t in triggers)


def _is_pricing_query(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in ["price", "pricing", "cost", "charge", "rate", "quotation", "quote"])


def _is_booking_support_query(text: str) -> bool:
    lowered = text.lower()
    triggers = [
        "previously booked",
        "previous booking",
        "previously i have booked",
        "i have booked",
        "i booked",
        "already booked",
        "booking status",
        "status of booking",
        "reschedule",
        "cancel booking",
        "update booking",
        "my booking",
    ]
    if any(t in lowered for t in triggers):
        return True
    has_booking_word = "book" in lowered
    has_support_intent = any(t in lowered for t in ["previous", "status", "reschedule", "cancel", "update"])
    return has_booking_word and has_support_intent


def _service_catalog_summary(agent: AgentConfig) -> str:
    if not agent.service_catalog:
        return "AC repair, home cleaning, salon at home, and general maintenance"
    return ", ".join(service.name for service in agent.service_catalog)


def _service_price_summary(agent: AgentConfig) -> str:
    if not agent.service_catalog:
        return "Pricing depends on service type and location."
    chunks = [f"{service.name}: {service.base_price_bhd}" for service in agent.service_catalog[:4]]
    return " | ".join(chunks)


def _normalized_tokens(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _match_service(candidate: str, agent: AgentConfig) -> str | None:
    lowered = candidate.lower()
    for service in agent.service_catalog:
        tokens = [service.name, service.service_id, *service.keywords]
        if any(token.lower() in lowered for token in tokens if token):
            return service.name

    candidate_norm = _normalized_tokens(candidate)
    if not candidate_norm:
        return None

    best_name: str | None = None
    best_score = 0.0
    for service in agent.service_catalog:
        labels = [service.name, service.service_id, *service.keywords]
        for label in labels:
            label_norm = _normalized_tokens(label)
            if not label_norm:
                continue
            score = difflib.SequenceMatcher(None, candidate_norm, label_norm).ratio()
            if score > best_score:
                best_score = score
                best_name = service.name

    if best_name and best_score >= 0.62:
        return best_name
    return None


def build_system_messages(agent: AgentConfig, caller_language: str | None = None) -> list[dict[str, str]]:
    language_hint = caller_language or agent.language or "en"
    language_rule = (
        "Reply fully in Arabic unless caller asks for English."
        if language_hint == "ar"
        else "Reply fully in English unless caller asks for Arabic."
    )
    return [
        {"role": "system", "content": f"{prompts.SYSTEM_PROMPT}\n\nLanguage rule: {language_rule}"},
        {"role": "assistant", "content": agent.greeting or prompts.GREETING_TEMPLATE},
    ]


async def _analyze_turn(
    llm_client: OpenAIClient,
    question: str,
    user_input: str,
    collected_answers: dict[str, str],
) -> dict[str, Any]:
    if hasattr(llm_client, "analyze_turn"):
        try:
            result = await llm_client.analyze_turn(question, user_input, collected_answers)
            if isinstance(result, dict):
                logger.bind(question=question).debug("LLM intake routing result", intent=result.get("intent"))
                return result
        except Exception:  # noqa: BLE001
            logger.exception("LLM analyze_turn failed; using heuristic fallback")

    cleaned = user_input.strip()
    if cleaned.lower() in {"cancel", "stop", "hang up"}:
        return {"intent": "cancel", "extracted_answer": "", "assistant_reply": "Understood."}
    if len(cleaned) < 3 or _is_unknown(cleaned):
        return {
            "intent": "unclear",
            "extracted_answer": "",
            "assistant_reply": "Could you answer that question more clearly?",
        }
    return {"intent": "answer", "extracted_answer": cleaned, "assistant_reply": ""}


async def _natural_ask(llm_client: OpenAIClient, question: str, language: str) -> str:
    system = (
        "You are a phone intake agent. Rephrase the given question naturally in one sentence. "
        "Keep the exact meaning and keep it short."
    )
    user = f"Language: {language}\nQuestion: {question}"
    try:
        text = await llm_client.generate_reply([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        return (text or question).strip()
    except Exception:  # noqa: BLE001
        return question


async def handle_intake(session: CallSession, user_input: str, llm_client: OpenAIClient) -> dict[str, str]:
    language = session.preferred_language or session.agent_config.language or "en"
    cleaned = user_input.strip()
    if cleaned:
        language = _detect_language(cleaned, fallback=language)
        session.preferred_language = language

    lowered = cleaned.lower()
    if any(term in lowered for term in _cancel_terms()):
        text = _localize(language, "No problem. I will end the call now. Thank you.", "No problem. I will end the call now. Thank you.")
        return {"action": "disqualify", "text": text}

    if session.current_state == "greeting":
        session.awaiting_answer = True
        if not cleaned:
            question = session.next_question or prompts.INTAKE_FALLBACK
            ask_text = await _natural_ask(llm_client, question, language)
            return {"action": "next", "text": ask_text}
        if _is_booking_support_query(cleaned):
            return {
                "action": "next",
                "text": (
                    "I can help with existing bookings too. In this local demo, share your booking reference or phone number, "
                    "or tell me the new service you want."
                ),
            }
        if _is_business_info_query(cleaned):
            business_name = session.agent_config.business_name
            services = _service_catalog_summary(session.agent_config)
            coverage = session.agent_config.coverage_country or "Bahrain"
            prices = _service_price_summary(session.agent_config)
            question = session.next_question or "Which service do you need?"
            return {
                "action": "next",
                "text": (
                    f"We are {business_name}. We provide {services}. "
                    f"Our current service coverage is {coverage}. "
                    f"Typical price ranges: {prices}. "
                    f"{question}"
                ),
            }
        if _is_pricing_query(cleaned):
            prices = _service_price_summary(session.agent_config)
            question = session.next_question or "Which service do you need?"
            return {
                "action": "next",
                "text": f"Typical price ranges are: {prices}. {question}",
            }

    if cleaned and not session.awaiting_answer and session.current_question_index == 0:
        session.awaiting_answer = True

    if session.awaiting_answer:
        if _is_unknown(cleaned):
            question = session.next_question or "Could you clarify?"
            follow = await _natural_ask(llm_client, question, language)
            return {"action": "next", "text": follow}

        if len(cleaned) < 3:
            retry_text = _localize(language, "Could you share a little more detail?", "Could you share a little more detail?")
            return {"action": "next", "text": retry_text}

        question = session.next_question or ""

        if session.current_question_index == 0 and _is_business_info_query(cleaned):
            business_name = session.agent_config.business_name
            services = _service_catalog_summary(session.agent_config)
            coverage = session.agent_config.coverage_country or "Bahrain"
            prices = _service_price_summary(session.agent_config)
            return {
                "action": "next",
                "text": (
                    f"We are {business_name}. We provide {services}. "
                    f"Our coverage is {coverage}. Typical price ranges are: {prices}. "
                    "Tell me which service you want and I will book it for you."
                ),
            }

        if session.current_question_index == 0 and _is_booking_support_query(cleaned):
            return {
                "action": "next",
                "text": (
                    "Understood. For previous booking support, share your booking reference or phone number. "
                    "If you want a new booking now, tell me the service."
                ),
            }

        if session.current_question_index == 0 and _is_pricing_query(cleaned):
            prices = _service_price_summary(session.agent_config)
            return {
                "action": "next",
                "text": f"Typical price ranges are: {prices}. Tell me which service you need and I can narrow the estimate.",
            }

        analysis = await _analyze_turn(llm_client, question, cleaned, session.collected_answers)
        intent = str(analysis.get("intent") or "unclear")

        if intent == "cancel":
            text = _localize(language, "Understood. We can stop here. Thank you for your time.", "Understood. We can stop here. Thank you for your time.")
            return {"action": "disqualify", "text": text}

        if intent in {"info_request", "unclear"}:
            assistant_reply = str(analysis.get("assistant_reply") or "").strip()
            if session.current_question_index == 0 and _is_business_info_query(cleaned):
                business_name = session.agent_config.business_name
                services = _service_catalog_summary(session.agent_config)
                coverage = session.agent_config.coverage_country or "Bahrain"
                prices = _service_price_summary(session.agent_config)
                assistant_reply = (
                    f"We are {business_name}. We provide {services}. "
                    f"Our coverage is {coverage}. Typical price ranges are: {prices}. "
                    "Tell me which service you want and I will continue."
                )
            elif session.current_question_index == 0 and _is_pricing_query(cleaned):
                prices = _service_price_summary(session.agent_config)
                assistant_reply = f"Typical price ranges are: {prices}. Tell me which service you want and I can continue with booking."
            elif session.current_question_index == 0 and _is_booking_support_query(cleaned):
                assistant_reply = (
                    "I can help with booking support too. Share booking reference or phone number, "
                    "or tell me the new service you need."
                )

            if not assistant_reply:
                assistant_reply = _localize(
                    language,
                    "I can help with that, and I still need this answer to continue.",
                    "I can help with that, and I still need this answer to continue.",
                )
            reask = await _natural_ask(llm_client, question, language)
            return {"action": "next", "text": f"{assistant_reply} {reask}".strip()}

        candidate_answer = str(analysis.get("extracted_answer") or cleaned).strip()
        if len(candidate_answer) < 2:
            return {"action": "next", "text": "Could you answer that specific question?"}

        normalized = candidate_answer

        if session.current_question_index == 0 and session.agent_config.service_catalog:
            matched_service = _match_service(normalized, session.agent_config)
            if not matched_service:
                session.service_prompt_retries += 1
                supported = ", ".join(s.name for s in session.agent_config.service_catalog)
                if session.service_prompt_retries >= 2:
                    return {
                        "action": "next",
                        "text": "No rush. Tell me in simple words if you need new booking, pricing details, or help with existing booking.",
                    }
                return {
                    "action": "next",
                    "text": f"I can help with these services: {supported}. Which one do you need?",
                }
            session.service_prompt_retries = 0

        question_lower = question.lower()
        if any(
            term in question_lower
            for term in [
                "where",
                "area",
                "location",
                "\u0648\u064a\u0646",
                "\u0627\u0644\u0645\u0646\u0637\u0642\u0629",
                "\u0627\u0644\u0645\u0648\u0642\u0639",
            ]
        ):
            if _is_out_of_coverage(normalized):
                session.out_of_coverage = True
                disq = _localize(
                    language,
                    "At the moment we only serve locations in Bahrain. I can still answer your questions and arrange a human follow-up if you want.",
                    "At the moment we only serve locations in Bahrain. I can still answer your questions and arrange a human follow-up if you want.",
                )
                key = f"q{session.current_question_index}"
                session.collected_answers[key] = normalized
                session.current_question_index += 1
                session.awaiting_answer = False
                return {"action": "out_of_coverage", "text": disq}

        key = f"q{session.current_question_index}"
        session.collected_answers[key] = normalized
        session.current_question_index += 1
        session.awaiting_answer = False

    if session.has_more_questions:
        question = session.next_question or prompts.INTAKE_FALLBACK
        session.awaiting_answer = True
        ask_text = await _natural_ask(llm_client, question, language)
        return {"action": "next", "text": ask_text}

    return {"action": "complete", "text": "intake-complete"}


async def handle_booking(
    session: CallSession,
    backend_client: BackendClient,
    llm_client: OpenAIClient,
) -> dict[str, str | None]:
    try:
        booking_response = await backend_client.book_service(
            agent_id=session.agent_config.agent_id,
            answers=session.collected_answers,
        )
        confirmation = booking_response.get("message") or "Your request has been registered."
        language_hint = _language_hint(" ".join(session.collected_answers.values()))
        if hasattr(llm_client, "rewrite_confirmation"):
            natural_confirmation = await llm_client.rewrite_confirmation(
                str(confirmation), caller_language_hint=language_hint
            )
        else:
            natural_confirmation = str(confirmation)
        return {"action": "speak", "text_to_speak": natural_confirmation, "transfer_number": None}
    except Exception:  # noqa: BLE001
        if session.agent_config.fallback_phone:
            transfer_text = _localize(
                session.preferred_language or session.agent_config.language or "en",
                "I will transfer you to a human agent to complete your request.",
                "I will transfer you to a human agent to complete your request.",
            )
            return {
                "action": "transfer",
                "text_to_speak": transfer_text,
                "transfer_number": session.agent_config.fallback_phone,
            }
        fail_text = _localize(
            session.preferred_language or session.agent_config.language or "en",
            "A temporary error happened. Please call again shortly.",
            "A temporary error happened. Please call again shortly.",
        )
        return {
            "action": "hangup",
            "text_to_speak": fail_text,
            "transfer_number": None,
        }
