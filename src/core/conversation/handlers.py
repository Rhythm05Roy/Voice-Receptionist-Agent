"""Tool-call executors for the conversation engine.

Instead of managing a rigid intake flow, these handlers execute
actions requested by the LLM via function calls:
  - submit_booking  →  sends data to backend
  - track_booking   →  queries booking status
  - transfer_to_human → returns transfer instruction
  - end_call        →  returns hangup instruction
"""

from __future__ import annotations

from typing import Any

from loguru import logger


async def execute_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    backend_client: Any,
    agent_config: Any,
    session_state: dict[str, Any],
) -> dict[str, Any]:
    """Route a tool call to the appropriate handler and return structured result."""

    if tool_name == "submit_booking":
        return await _handle_submit_booking(arguments, backend_client, agent_config, session_state)

    if tool_name == "track_booking":
        return await _handle_track_booking(arguments, backend_client, agent_config)

    if tool_name == "transfer_to_human":
        return _handle_transfer(arguments, agent_config)

    if tool_name == "end_call":
        return _handle_end_call(arguments)

    logger.warning("Unknown tool call", tool_name=tool_name)
    return {"status": "error", "message": f"Unknown tool: {tool_name}"}


async def _handle_submit_booking(
    args: dict[str, Any],
    backend_client: Any,
    agent_config: Any,
    session_state: dict[str, Any],
) -> dict[str, Any]:
    """Submit a booking via the backend client."""
    service_type = args.get("service_type", "")
    location = args.get("location", "")
    preferred_time = args.get("preferred_time", "")
    preferred_date = args.get("preferred_date", "")
    customer_name = args.get("customer_name", "")
    customer_phone = args.get("customer_phone", "")

    # Validate coverage
    if agent_config.coverage_areas:
        location_lower = location.lower()
        country_match = agent_config.coverage_country and agent_config.coverage_country.lower() in location_lower
        area_match = any(area.lower() in location_lower for area in agent_config.coverage_areas)
        if not country_match and not area_match:
            return {
                "status": "out_of_coverage",
                "message": (
                    f"The location '{location}' appears to be outside our service area. "
                    f"We currently serve {agent_config.coverage_country or 'select areas'}, "
                    f"including {', '.join(agent_config.coverage_areas[:5])}."
                ),
            }

    # Validate service matches catalog
    matched_service = service_type
    if agent_config.service_catalog:
        catalog_names = [svc.name.lower() for svc in agent_config.service_catalog]
        if service_type.lower() not in catalog_names:
            # Try fuzzy match
            for svc in agent_config.service_catalog:
                if any(kw in service_type.lower() for kw in (svc.keywords or [])):
                    matched_service = svc.name
                    break
                if service_type.lower() in svc.name.lower() or svc.name.lower() in service_type.lower():
                    matched_service = svc.name
                    break

    # Build comprehensive booking answers
    answers: dict[str, Any] = {
        "service_type": matched_service,
        "location": location,
        "preferred_date": preferred_date,
        "preferred_time": preferred_time,
    }
    # Customer info
    if customer_name:
        answers["customer_name"] = customer_name
    if customer_phone:
        answers["customer_phone"] = customer_phone
    # Service-specific details
    if args.get("property_type"):
        answers["property_type"] = args["property_type"]
    if args.get("num_rooms"):
        answers["num_rooms"] = args["num_rooms"]
    if args.get("specific_areas"):
        answers["specific_areas"] = args["specific_areas"]
    if args.get("allergy_info"):
        answers["allergy_info"] = args["allergy_info"]
    if args.get("issue_description"):
        answers["issue_description"] = args["issue_description"]
    if args.get("urgency"):
        answers["urgency"] = args["urgency"]
    if args.get("special_instructions"):
        answers["special_instructions"] = args["special_instructions"]

    try:
        result = await backend_client.book_service(
            agent_id=agent_config.agent_id,
            answers=answers,
        )

        booking_ref = result.get("booking_ref", "")
        short_id = result.get("short_booking_id", "")

        # Store in session state for later reference
        session_state["last_booking_ref"] = booking_ref
        session_state["last_short_id"] = short_id
        session_state["booking_completed"] = True

        return {
            "status": "confirmed",
            "booking_ref": booking_ref,
            "short_booking_id": short_id,
            "message": result.get("message", "Booking confirmed."),
            "service": matched_service,
            "location": location,
            "date": preferred_date,
            "time": preferred_time,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("Booking failed")
        return {
            "status": "error",
            "message": f"Booking failed: {exc}. Would you like me to try again or transfer to a human?",
        }


async def _handle_track_booking(
    args: dict[str, Any],
    backend_client: Any,
    agent_config: Any,
) -> dict[str, Any]:
    """Look up booking status from the backend."""
    booking_id = args.get("booking_id", "").strip()

    if not booking_id:
        return {"status": "missing_id", "message": "No booking ID provided."}

    try:
        result = await backend_client.track_booking(booking_id, agent_id=agent_config.agent_id)
        return {
            "status": result.get("status", "unknown"),
            "booking_ref": result.get("booking_ref", booking_id),
            "message": result.get("message", ""),
            "service_name": result.get("service_name"),
            "location": result.get("location"),
            "preferred_time": result.get("preferred_time"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tracking failed")
        return {"status": "error", "message": f"Could not look up booking: {exc}"}


def _handle_transfer(args: dict[str, Any], agent_config: Any) -> dict[str, Any]:
    """Return transfer instruction."""
    reason = args.get("reason", "caller requested")
    phone = agent_config.fallback_phone

    if not phone:
        return {
            "status": "no_agent",
            "message": "No human agent is available right now. Please try again later.",
        }

    return {
        "status": "transferring",
        "transfer_number": phone,
        "reason": reason,
        "message": f"Transferring to human agent: {reason}",
    }


def _handle_end_call(args: dict[str, Any]) -> dict[str, Any]:
    """Return call-ending instruction."""
    farewell = args.get("farewell_message", "Thank you for calling. Goodbye!")
    return {
        "status": "ended",
        "farewell_message": farewell,
    }
