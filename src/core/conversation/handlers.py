from src.core.types import AgentConfig
from src.core.services.backend_client import BackendClient
from . import prompts


def build_system_messages(agent: AgentConfig) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "assistant", "content": agent.greeting or prompts.GREETING_TEMPLATE},
    ]


def _is_unknown(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {"i don't know", "idk", "not sure", "ما ادري", "مدري"}


async def handle_intake(session, user_input: str) -> dict[str, str]:
    lowered = user_input.strip().lower()
    if any(term in lowered for term in ["cancel", "not interested", "later", "مشغول"]):
        return {"action": "disqualify", "text": "ما في مشكلة، بننهي المكالمة الآن. شكراً لتواصلك."}

    if session.current_state == "greeting":
        question = session.next_question or prompts.INTAKE_FALLBACK
        session.awaiting_answer = True
        return {"action": "next", "text": question}

    if session.awaiting_answer:
        if _is_unknown(user_input):
            return {"action": "next", "text": "ما عليش، ممكن توضح أكثر أو تعطيني مثال؟"}

        cleaned = user_input.strip()
        if len(cleaned) < 3:
            return {"action": "next", "text": "ممكن تفاصيل أكثر لو سمحت؟"}

        key = f"q{session.current_question_index}"
        session.collected_answers[key] = cleaned
        session.current_question_index += 1
        session.awaiting_answer = False

    if session.has_more_questions:
        question = session.next_question or prompts.INTAKE_FALLBACK
        session.awaiting_answer = True
        return {"action": "next", "text": question}

    return {"action": "complete", "text": "intake-complete"}


async def handle_booking(session, backend_client: BackendClient) -> dict[str, str | None]:
    try:
        booking_response = await backend_client.book_service(
            agent_id=session.agent_config.agent_id,
            answers=session.collected_answers,
        )
        confirmation = booking_response.get("message") or "تم تسجيل الطلب، بنأكد معك لاحقاً."
        return {"action": "speak", "text_to_speak": confirmation, "transfer_number": None}
    except Exception:  # noqa: BLE001
        if session.agent_config.fallback_phone:
            return {
                "action": "transfer",
                "text_to_speak": "بحولك لزميلي يكمل معك الحجز.",
                "transfer_number": session.agent_config.fallback_phone,
            }
        return {
            "action": "hangup",
            "text_to_speak": "صار خطأ بسيط. تقدر تتصل فينا بعد شوي؟",
            "transfer_number": None,
        }
