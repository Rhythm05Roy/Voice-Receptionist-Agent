from __future__ import annotations

import argparse
import sys
import uuid
from typing import Iterable

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local multi-turn telephony webhook simulator for ai-voice-service."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for ai-voice-service (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--endpoint",
        default="/api/v1/telephony/webhook/inbound",
        help="Webhook endpoint path",
    )
    parser.add_argument(
        "--from-number",
        default="+10000000000",
        help="Simulated caller number",
    )
    parser.add_argument(
        "--to-number",
        default="+20000000000",
        help="Simulated business number",
    )
    parser.add_argument(
        "--call-id",
        default=None,
        help="Optional fixed call UUID. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--inputs",
        default=None,
        help='Optional scripted inputs separated by "|", e.g. "ac service|manama|today 6pm"',
    )
    return parser.parse_args()


def print_ncco(ncco: list[dict]) -> None:
    for idx, action in enumerate(ncco, start=1):
        kind = action.get("action")
        if kind == "talk":
            print(f"agent[{idx}] talk: {action.get('text', '')}")
        elif kind == "listen":
            print(f"agent[{idx}] listen: speechTimeout={action.get('speechTimeout', 'n/a')}")
        elif kind == "connect":
            endpoint = action.get("endpoint", [])
            target = ""
            if endpoint and isinstance(endpoint, list):
                target = endpoint[0].get("number", "")
            print(f"agent[{idx}] connect: {target}")
        elif kind == "hangup":
            print(f"agent[{idx}] hangup")
        else:
            print(f"agent[{idx}] {kind}: {action}")


def scripted_inputs(raw: str | None) -> Iterable[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split("|") if part.strip()]


def post_turn(
    client: httpx.Client,
    url: str,
    from_number: str,
    to_number: str,
    call_id: str,
    speech_text: str | None = None,
) -> dict:
    payload: dict = {
        "from": from_number,
        "to": to_number,
        "uuid": call_id,
    }
    if speech_text is not None:
        payload["speech"] = {"results": [{"text": speech_text}]}
    response = client.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def run() -> int:
    args = parse_args()
    call_id = args.call_id or str(uuid.uuid4())
    url = f"{args.base_url.rstrip('/')}{args.endpoint}"
    queued_inputs = list(scripted_inputs(args.inputs))

    print(f"webhook: {url}")
    print(f"call_id: {call_id}")
    print("type `exit` or `quit` to stop")
    print("-" * 60)

    with httpx.Client() as client:
        try:
            first = post_turn(
                client=client,
                url=url,
                from_number=args.from_number,
                to_number=args.to_number,
                call_id=call_id,
                speech_text=None,
            )
        except httpx.HTTPError as exc:
            print(f"error: initial request failed: {exc}")
            return 1

        ncco = first.get("ncco", [])
        print_ncco(ncco)

        while True:
            if queued_inputs:
                user_text = queued_inputs.pop(0)
                print(f"you: {user_text}")
            else:
                try:
                    user_text = input("you: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nstopped")
                    return 0

            if user_text.lower() in {"exit", "quit"}:
                print("stopped")
                return 0

            try:
                turn = post_turn(
                    client=client,
                    url=url,
                    from_number=args.from_number,
                    to_number=args.to_number,
                    call_id=call_id,
                    speech_text=user_text,
                )
            except httpx.HTTPError as exc:
                print(f"error: turn request failed: {exc}")
                return 1

            ncco = turn.get("ncco", [])
            print_ncco(ncco)

            if any(action.get("action") == "hangup" for action in ncco):
                print("call ended by agent")
                return 0


if __name__ == "__main__":
    sys.exit(run())
