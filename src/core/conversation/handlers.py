from __future__ import annotations

import difflib
import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.core.services.backend_client import BackendClient
from src.core.services.openai import OpenAIClient
from src.core.types import AgentConfig, IntakeQuestion

from . import prompts

if TYPE_CHECKING:
    from .engine import CallSession


def _contains_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def _localize(language: str, en_text: str, ar_text: str) -> str:
    return ar_text if language == "ar" else en_text


def _normalize(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _cancel_terms() -> tuple[str, ...]:
    return (
        "cancel",
        "stop",
        "not interested",
        "end call",
        "hang up",
        "later",
    )


def _is_unknown(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {"i don't know", "idk", "not sure", "no idea", "unknown"}


def _match_service_name(candidate: str, agent: AgentConfig) -> str | None:
    lowered = candidate.lower()
    for service in agent.service_catalog:
        labels = [service.name, service.service_id, *service.keywords]
        if any(label.lower() in lowered for label in labels if label):
            return service.name

    candidate_norm = _normalize(candidate)
    best_name: str | None = None
    best_score = 0.0
    for service in agent.service_catalog:
        labels = [service.name, service.service_id, *service.keywords]
        for label in labels:
            label_norm = _normalize(label)
            if not label_norm:
                continue
            score = difflib.SequenceMatcher(None, candidate_norm, label_norm).ratio()
            if score > best_score:
                best_score = score
                best_name = service.name

    if best_name and best_score >= 0.62:
        return best_name
    return None


def _is_out_of_coverage(answer: str, agent: AgentConfig) -> bool:
    normalized = answer.strip().lower()
    if not normalized:
        return False

    if agent.coverage_country and agent.coverage_country.lower() in normalized:
        return False

    for area in agent.coverage_areas:
        if area.lower() in normalized:
            return False

    foreign_markers = {
        "bangladesh",
        "dhaka",
        "india",
        "pakistan",
        "usa",
        "uk",
        "canada",
        "qatar",
        "uae",
        "oman",
        "kuwait",
        "saudi",
    }
    return any(marker in normalized for marker in foreign_markers)


def _to_bool_answer(text: str) -> str | None:
    normalized = text.strip().lower()
    compact = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    compact = re.sub(r"\s+", " ", compact).strip()

    yes_terms = (
        "yes",
        "yeah",
        "yep",
        "true",
        "sure",
        "affirmative",
        "i have",
        "have allergy",
        "allergy yes",
    )
    no_terms = (
        "no",
        "nope",
        "false",
        "negative",
        "do not",
        "don't",
        "dont",
        "no allergy",
    )

    if compact in {"y", "yes"}:
        return "yes"
    if compact in {"n", "no"}:
        return "no"

    if any(term in compact for term in no_terms):
        return "no"
    if any(term in compact for term in yes_terms):
        return "yes"

    # Handle short transcribed variations like "yas", "ya", "yess".
    if re.fullmatch(r"y(e|a)?s+", compact):
        return "yes"
    if re.fullmatch(r"n+o+", compact):
        return "no"

    return None


def _active_intake_questions(session: CallSession) -> list[IntakeQuestion]:
    questions = session.agent_config.intake_questions
    if not questions:
        return []

    selected_service = (session.selected_service or "").strip().lower()
    active: list[IntakeQuestion] = []
    for question in questions:
        if question.ask_when == "all_bookings":
            active.append(question)
            continue
        if not selected_service:
            continue
        service_tags = [tag.lower() for tag in question.service_tags]
        if any(tag in selected_service for tag in service_tags):
            active.append(question)
    return active


def _format_reask(question: IntakeQuestion, language: str) -> str:
    if question.retry_prompt:
        return question.retry_prompt
    return _localize(
        language,
        f"Sorry, I still need this to continue: {question.question}",
        f"Sorry, I still need this to continue: {question.question}",
    )


def build_system_messages(agent: AgentConfig, caller_language: str | None = None) -> list[dict[str, str]]:
    hint = (caller_language or agent.default_greeting_language or agent.language or "en").lower()
    if hint not in set(agent.supported_languages):
        hint = agent.default_greeting_language or "en"
    language_rule = (
        "Continue in Arabic only if caller explicitly uses Arabic."
        if hint == "ar"
        else "Continue in English unless caller requests another supported language."
    )
    return [
        {"role": "system", "content": f"{prompts.SYSTEM_PROMPT}\n\nLanguage policy: {language_rule}"},
        {"role": "assistant", "content": agent.greeting or prompts.GREETING_TEMPLATE},
    ]


def _render_question_prompt(question: IntakeQuestion) -> str:
    base = question.question.strip()
    if question.answer_type == "yes_no":
        return f"{base} Please answer yes or no."
    if question.answer_type == "multiple_choice" and question.options:
        return f"{base} Options are: {', '.join(question.options)}."
    return base


async def _natural_ask(llm_client: OpenAIClient, question: IntakeQuestion, language: str) -> str:
    # Deterministic rendering keeps style stable and avoids one extra LLM hop per turn.
    _ = llm_client
    _ = language
    return _render_question_prompt(question)


async def _validate_answer(
    session: CallSession,
    question: IntakeQuestion,
    user_input: str,
    llm_client: OpenAIClient,
) -> tuple[bool, str, str | None]:
    language = session.preferred_language or session.agent_config.default_greeting_language or "en"
    clean = user_input.strip()

    if question.answer_type == "yes_no":
        normalized_bool = _to_bool_answer(clean)
        if normalized_bool is None:
            return False, "", _format_reask(question, language)
        if normalized_bool == "yes":
            # Preserve useful detail for booking notes while still matching yes/no logic.
            return True, f"yes: {clean}", None
        return True, "no", None

    if question.answer_type == "multiple_choice":
        if not question.options:
            return bool(clean), clean, None
        normalized_input = _normalize(clean)
        for option in question.options:
            if _normalize(option) in normalized_input:
                return True, option, None
        options_preview = ", ".join(question.options)
        return False, "", f"Please choose one option: {options_preview}."

    if question.answer_type == "number":
        number_match = re.search(r"\d+(?:\.\d+)?", clean)
        if number_match:
            return True, number_match.group(0), None
        return False, "", _format_reask(question, language)

    if question.validation_regex:
        try:
            if not re.search(question.validation_regex, clean):
                return False, "", _format_reask(question, language)
        except re.error:
            logger.warning("Invalid validation regex in intake question", key=question.key)

    if question.key in {"service", "service_type", "service_name"}:
        matched = _match_service_name(clean, session.agent_config)
        if not matched:
            services = ", ".join(service.name for service in session.agent_config.service_catalog)
            return False, "", f"We currently support: {services}. Which one do you need?"
        session.selected_service = matched
        return True, matched, None

    if len(clean) < 2:
        return False, "", _format_reask(question, language)

    analysis = await llm_client.analyze_turn(
        question=question.question,
        user_input=clean,
        collected_answers=session.collected_answers,
        question_meta=question.model_dump(),
    )

    intent = str(analysis.get("intent", "unclear"))
    if intent == "cancel":
        return False, "", _localize(
            language,
            "Understood. I can end the call now.",
            "Understood. I can end the call now.",
        )

    if intent in {"info_request", "off_topic", "unclear"}:
        assistant_reply = str(analysis.get("assistant_reply") or "").strip()
        return False, "", assistant_reply or _format_reask(question, language)

    normalized_answer = str(analysis.get("normalized_answer") or analysis.get("extracted_answer") or clean).strip()
    return True, normalized_answer, None


def _apply_disqualification(question: IntakeQuestion, answer: str) -> dict[str, str | None] | None:
    normalized = answer.strip().lower()
    for rule in question.disqualification_rules:
        if rule.if_answer.strip().lower() in normalized:
            if rule.action == "transfer":
                return {
                    "action": "transfer",
                    "text": rule.message_to_caller,
                    "transfer_number": rule.transfer_number,
                }
            return {"action": "disqualify", "text": rule.message_to_caller, "transfer_number": None}
    return None


async def handle_intake(session: CallSession, user_input: str, llm_client: OpenAIClient) -> dict[str, str | None]:
    language = session.preferred_language or session.agent_config.default_greeting_language or "en"
    clean = user_input.strip()

    if clean:
        if _contains_arabic(clean) and "ar" in session.agent_config.supported_languages:
            session.preferred_language = "ar"
            language = "ar"
        elif "english" in clean.lower() and "en" in session.agent_config.supported_languages:
            session.preferred_language = "en"
            language = "en"

    lowered = clean.lower()
    if any(term in lowered for term in _cancel_terms()):
        return {
            "action": "disqualify",
            "text": _localize(language, "No problem. I will end the call now.", "No problem. I will end the call now."),
            "transfer_number": None,
        }

    questions = _active_intake_questions(session)
    if not questions:
        return {"action": "complete", "text": "intake-complete", "transfer_number": None}

    if session.current_question_index >= len(questions):
        return {"action": "complete", "text": "intake-complete", "transfer_number": None}

    current_question = questions[session.current_question_index]

    if not session.awaiting_answer:
        session.awaiting_answer = True
        if not clean:
            ask_text = await _natural_ask(llm_client, current_question, language)
            return {"action": "next", "text": ask_text, "transfer_number": None}

    if not clean:
        ask_text = await _natural_ask(llm_client, current_question, language)
        return {"action": "next", "text": ask_text, "transfer_number": None}

    if _is_unknown(clean):
        session.unknown_answer_retries += 1
        if session.unknown_answer_retries >= 2 and session.agent_config.fallback_phone:
            return {
                "action": "transfer",
                "text": "Let me connect you to a human agent for faster help.",
                "transfer_number": session.agent_config.fallback_phone,
            }
        return {"action": "next", "text": _format_reask(current_question, language), "transfer_number": None}

    is_valid, normalized_answer, feedback = await _validate_answer(session, current_question, clean, llm_client)
    if not is_valid:
        if feedback and "end the call" in feedback.lower():
            return {"action": "disqualify", "text": feedback, "transfer_number": None}

        if feedback:
            # Side-question handling: answer quickly but keep intake moving.
            if any(k in clean.lower() for k in ["service", "price", "pricing", "business"]):
                return {
                    "action": "info_then_reask",
                    "text": feedback,
                    "transfer_number": None,
                }
            return {"action": "next", "text": feedback, "transfer_number": None}
        return {"action": "next", "text": _format_reask(current_question, language), "transfer_number": None}

    if current_question.key in {"location", "area", "service_area"} and _is_out_of_coverage(normalized_answer, session.agent_config):
        session.out_of_coverage = True
        return {
            "action": "out_of_coverage",
            "text": (
                f"At the moment we currently operate in {session.agent_config.coverage_country}. "
                "I can still answer questions or connect you to a human agent."
            ),
            "transfer_number": session.agent_config.fallback_phone,
        }

    disqualification = _apply_disqualification(current_question, normalized_answer)
    if disqualification:
        return disqualification

    session.collected_answers[current_question.key] = normalized_answer
    session.collected_answers[f"q{session.current_question_index}"] = normalized_answer

    # Keep frequently-used aliases for downstream booking payloads.
    if current_question.key in {"service", "service_name", "service_type"}:
        session.selected_service = normalized_answer
        session.collected_answers["service_type"] = normalized_answer
    elif current_question.key in {"location", "area", "service_area"}:
        session.collected_answers["location"] = normalized_answer
    elif current_question.key in {"preferred_time", "visit_time", "time", "schedule"}:
        session.collected_answers["preferred_time"] = normalized_answer

    session.current_question_index += 1
    session.awaiting_answer = False
    session.unknown_answer_retries = 0

    questions = _active_intake_questions(session)
    if session.current_question_index >= len(questions):
        return {"action": "complete", "text": "intake-complete", "transfer_number": None}

    next_question = questions[session.current_question_index]
    session.awaiting_answer = True
    ask_text = await _natural_ask(llm_client, next_question, language)
    return {"action": "next", "text": ask_text, "transfer_number": None}


async def handle_booking(
    session: CallSession,
    backend_client: BackendClient,
    llm_client: OpenAIClient,
) -> dict[str, str | None]:
    _ = llm_client
    try:
        booking_response = await backend_client.book_service(
            agent_id=session.agent_config.agent_id,
            answers=session.collected_answers,
        )
        status = str(booking_response.get("status") or "").lower()

        if status == "unsupported_service":
            session.current_state = "intake"
            session.current_question_index = 0
            session.awaiting_answer = False
            return {
                "action": "speak",
                "text_to_speak": str(
                    booking_response.get("message")
                    or "I can help with available services. Which service would you like?"
                ),
                "transfer_number": None,
            }

        confirmation = str(booking_response.get("message") or "Your booking request has been recorded.")
        booking_ref = (
            booking_response.get("booking_ref")
            or booking_response.get("booking_id")
            or booking_response.get("id")
        )
        normalized_confirmation = confirmation
        if booking_ref and str(booking_ref) not in confirmation:
            normalized_confirmation = f"{confirmation} Your booking ID is {booking_ref}."

        return {
            "action": "speak",
            "text_to_speak": normalized_confirmation,
            "transfer_number": None,
            "booking_ref": str(booking_ref) if booking_ref else None,
        }
    except Exception:  # noqa: BLE001
        if session.agent_config.fallback_phone:
            return {
                "action": "transfer",
                "text_to_speak": "I will transfer you to a human agent to complete this request.",
                "transfer_number": session.agent_config.fallback_phone,
            }
        return {
            "action": "hangup",
            "text_to_speak": "A temporary issue happened while creating your booking. Please call again shortly.",
            "transfer_number": None,
        }
