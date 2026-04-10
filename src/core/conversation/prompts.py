"""Dynamic prompt generation for AI voice agents.

System prompts are built from AgentConfig data so each business
gets a tailored, context-rich prompt. The prompt includes
function-calling instructions so the LLM can naturally gather
booking details, handle corrections, and detect frustration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.types import AgentConfig


def build_system_prompt(agent: AgentConfig) -> str:
    """Generate a comprehensive system prompt for phone conversations."""
    parts: list[str] = []

    parts.append(
        f"You are an AI phone receptionist for **{agent.business_name}**. "
        "You are on a live phone call. Keep every reply SHORT (1-3 sentences max) "
        "because this will be spoken aloud. Sound warm, professional, and human."
    )

    parts.append(
        "You help callers with:\n"
        "1. Booking a new service\n"
        "2. Checking status of an existing booking (by booking ID)\n"
        "3. Answering questions about services, pricing, policies\n"
        "4. Transferring to a human agent when needed"
    )

    if agent.service_catalog:
        service_lines = []
        for svc in agent.service_catalog:
            price = svc.base_price or svc.base_price_bhd or "varies"
            service_lines.append(f"  - {svc.name}: {svc.description} ({price})")
        parts.append("Available services:\n" + "\n".join(service_lines))

    if agent.business_description:
        parts.append(f"About: {agent.business_description}")
    if agent.business_hours:
        parts.append(f"Hours: {agent.business_hours}")

    if agent.coverage_areas:
        parts.append(
            f"Coverage: {agent.coverage_country or 'Our service area'}, "
            f"including: {', '.join(agent.coverage_areas[:10])}."
        )
    elif agent.coverage_country:
        parts.append(f"Coverage: {agent.coverage_country}.")

    policy_parts = []
    if agent.cancellation_policy:
        policy_parts.append(f"Cancellation: {agent.cancellation_policy}")
    if agent.payment_policy:
        policy_parts.append(f"Payment: {agent.payment_policy}")
    if agent.deposit_policy:
        policy_parts.append(f"Deposit: {agent.deposit_policy}")
    if policy_parts:
        parts.append("Policies: " + " | ".join(policy_parts))

    if agent.faqs:
        faq_lines = [f"  Q: {k}  A: {v}" for k, v in list(agent.faqs.items())[:8]]
        parts.append("Common questions:\n" + "\n".join(faq_lines))

    intake_fields = []
    for q in agent.intake_questions:
        if isinstance(q, dict):
            if not q.get("is_active", True):
                continue
            key = q.get("key", "")
            question = q.get("question", key)
            required = q.get("required", True)
            ask_when = q.get("ask_when", "all_bookings")
            service_tags = q.get("service_tags", [])
            specific_categories = q.get("specific_categories", [])
        else:
            if not getattr(q, "is_active", True):
                continue
            key = getattr(q, "key", "")
            question = getattr(q, "question", key)
            required = getattr(q, "required", True)
            ask_when = getattr(q, "ask_when", "all_bookings")
            service_tags = getattr(q, "service_tags", [])
            specific_categories = getattr(q, "specific_categories", [])
        metadata = ["required" if required else "optional", f"ask_when={ask_when}"]
        if service_tags:
            metadata.append(f"service_tags={', '.join(service_tags)}")
        if specific_categories:
            metadata.append(f"categories={', '.join(specific_categories)}")
        intake_fields.append(f"  - {key}: ask '{question}' ({'; '.join(metadata)})")

    parts.append(
        "BOOKING FLOW RULES:\n"
        "- Gather these fields naturally through conversation (do NOT interrogate):\n"
        + "\n".join(intake_fields)
        + "\n"
        "- Do NOT ask all questions at once. Ask one at a time conversationally.\n"
        "- If the caller already mentioned info (e.g. 'I need AC repair'), extract it and do not re-ask.\n"
        "- Respect ask_when, service_tags, and categories from the configured intake questions. Only ask service-specific questions when they apply.\n"
        "- When ALL required fields are collected, call the `submit_booking` tool.\n"
        "- If caller wants to CHANGE a previous answer (e.g. 'update my location'), acknowledge it and update your understanding. Do NOT ignore correction requests.\n"
        "- If a field is unclear, ask for clarification naturally.\n\n"
        "SERVICE-SPECIFIC INTAKE (collect these additional details based on service type):\n\n"
        "For AC Repair / Maintenance:\n"
        "  - How many AC units need service?\n"
        "  - What is the issue? (not cooling, leaking, noise, routine maintenance)\n"
        "  - Is the unit a split, window, or central AC?\n"
        "  - Is this urgent or can it wait for a scheduled slot?\n\n"
        "For Home Deep Cleaning:\n"
        "  - Property type (apartment, villa, office)?\n"
        "  - Number of rooms/bathrooms?\n"
        "  - Which areas to focus on (kitchen, bathroom, floors, all)?\n"
        "  - Any allergies or chemical sensitivities?\n"
        "  - Do they want the sanitization add-on?\n\n"
        "For At-Home Salon:\n"
        "  - What salon service? (haircut, beard trim, grooming, styling)\n"
        "  - How many people need service?\n"
        "  - Gender preference for stylist?\n\n"
        "For General Maintenance:\n"
        "  - What is the issue? (electrical, plumbing, fixture, painting)\n"
        "  - Describe the problem specifically\n"
        "  - Is this an emergency (e.g., water leak)?\n\n"
        "ALWAYS collect these for EVERY booking:\n"
        "  1. Service type (required)\n"
        "  2. Location / area (required)\n"
        "  3. Preferred date (required - today, tomorrow, specific day)\n"
        "  4. Preferred time window (required - morning, afternoon, specific hours)\n"
        "  5. Customer name (ask politely)\n"
        "  6. Contact phone number (ask politely for confirmation callbacks)\n"
        "  7. Service-specific details (from the lists above)\n"
        "  8. Any special instructions\n\n"
        "Before calling submit_booking, confirm all details with the caller in a summary."
    )

    supported = ", ".join(agent.supported_languages) if agent.supported_languages else "en"
    default_lang = agent.default_greeting_language or agent.language or "en"
    parts.append(
        f"Languages: {supported}. Default: {default_lang}. "
        "Mirror the caller's language. If they ask to switch language, switch immediately."
    )

    parts.append(
        "CRITICAL RULES:\n"
        "- LISTEN CAREFULLY. If the caller is trying to tell you something, acknowledge it.\n"
        "- If the caller says 'you're not listening' or shows frustration, apologize sincerely and offer to transfer to a human agent using `transfer_to_human`.\n"
        "- NEVER ignore what the caller just said to repeat your own question.\n"
        "- If the caller provides info AND asks a question in the same sentence, acknowledge the info, answer the question, then naturally continue.\n"
        "- Do NOT invent services or prices not listed above.\n"
        "- When caller says goodbye, use `end_call` tool.\n"
        "- Keep responses VERY short because this is a phone call."
    )

    if agent.fallback_phone:
        parts.append(
            f"Human agent fallback: {agent.fallback_phone}. "
            "Use `transfer_to_human` if caller requests human help or is frustrated."
        )

    return "\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are a production AI phone agent. "
    "Be conversational and natural. Keep responses short for phone calls."
)

GREETING_TEMPLATE = "Hello, this is your AI assistant. How can I help you today?"

INTAKE_FALLBACK = "Sorry, I did not catch that clearly. Could you please repeat?"

FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "I want to know what services you provide."},
    {"role": "assistant", "content": "Sure. Let me share our available services."},
]
