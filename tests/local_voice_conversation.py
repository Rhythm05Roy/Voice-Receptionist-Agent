from __future__ import annotations

import argparse
import io
import os
import sys
import uuid
import wave
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

try:
    import numpy as np
    import sounddevice as sd
except Exception:  # noqa: BLE001
    print("Missing dependencies for live voice mode.")
    print("Install with: ./.venv/Scripts/python -m pip install numpy sounddevice")
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local real voice conversation: mic input + ElevenLabs spoken output."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/api/v1/telephony/webhook/inbound")
    parser.add_argument("--from-number", default="+10000000000")
    parser.add_argument("--to-number", default="+20000000000")
    parser.add_argument("--call-id", default=None, help="Optional call ID. Auto-generated when omitted.")
    parser.add_argument("--record-seconds", type=float, default=6.0)
    parser.add_argument("--input-rate", type=int, default=16000)
    parser.add_argument("--output-rate", type=int, default=24000)
    return parser.parse_args()


def _record_wav_bytes(seconds: float, sample_rate: int) -> bytes:
    frames = int(seconds * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    mono = audio.reshape(-1)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(mono.tobytes())
    return buf.getvalue()


def _transcribe(openai_client: OpenAI, wav_bytes: bytes) -> str:
    bio = io.BytesIO(wav_bytes)
    bio.name = "input.wav"
    result = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=bio,
    )
    return (result.text or "").strip()


def _elevenlabs_pcm(
    text: str,
    api_key: str,
    voice_id: str,
    sample_rate: int,
) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_{sample_rate}"
    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/pcm",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.content


def _speak_text(text: str, api_key: str, voice_id: str, sample_rate: int) -> None:
    if not text.strip():
        return
    pcm = _elevenlabs_pcm(text, api_key=api_key, voice_id=voice_id, sample_rate=sample_rate)
    audio = np.frombuffer(pcm, dtype=np.int16)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def _print_ncco(ncco: list[dict[str, Any]]) -> None:
    for action in ncco:
        kind = action.get("action")
        if kind == "talk":
            print(f"agent: {action.get('text', '')}")
        elif kind == "listen":
            print("agent: [listening]")
        elif kind == "connect":
            endpoint = action.get("endpoint", [])
            number = endpoint[0].get("number") if endpoint else ""
            print(f"agent: [transfer] {number}")
        elif kind == "hangup":
            print("agent: [hangup]")


def _send_turn(
    base_url: str,
    endpoint: str,
    from_number: str,
    to_number: str,
    call_id: str,
    speech_text: str | None,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "from": from_number,
        "to": to_number,
        "uuid": call_id,
    }
    if speech_text is not None:
        payload["speech"] = {"results": [{"text": speech_text}]}
    url = f"{base_url.rstrip('/')}{endpoint}"
    resp = httpx.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("ncco", [])


def main() -> int:
    load_dotenv(".env")
    args = parse_args()
    call_id = args.call_id or str(uuid.uuid4())

    openai_key = os.getenv("OPENAI_API_KEY", "")
    eleven_key = os.getenv("ELEVENLABS_API_KEY", "")
    eleven_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")

    if not openai_key:
        print("OPENAI_API_KEY is missing in .env")
        return 1
    if not eleven_key or not eleven_voice_id:
        print("ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID is missing in .env")
        return 1

    openai_client = OpenAI(api_key=openai_key)

    print("Starting local voice conversation")
    print(f"call_id: {call_id}")
    print("Say 'exit' or 'quit' to stop")
    print(f"record window: {args.record_seconds:.1f}s (use --record-seconds to change)")
    print("-" * 60)

    try:
        ncco = _send_turn(
            base_url=args.base_url,
            endpoint=args.endpoint,
            from_number=args.from_number,
            to_number=args.to_number,
            call_id=call_id,
            speech_text=None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Initial webhook failed: {exc}")
        return 1

    _print_ncco(ncco)
    for action in ncco:
        if action.get("action") == "talk":
            _speak_text(
                action.get("text", ""),
                api_key=eleven_key,
                voice_id=eleven_voice_id,
                sample_rate=args.output_rate,
            )

    while True:
        print(f"\nListening for {args.record_seconds:.1f}s...")
        try:
            wav_bytes = _record_wav_bytes(seconds=args.record_seconds, sample_rate=args.input_rate)
            user_text = _transcribe(openai_client, wav_bytes)
        except KeyboardInterrupt:
            print("\nstopped by user")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"Voice input failed: {exc}")
            continue

        if not user_text:
            print("you: [no speech detected]")
            continue

        print(f"you: {user_text}")
        if user_text.lower() in {"exit", "quit"}:
            print("stopped")
            return 0

        try:
            ncco = _send_turn(
                base_url=args.base_url,
                endpoint=args.endpoint,
                from_number=args.from_number,
                to_number=args.to_number,
                call_id=call_id,
                speech_text=user_text,
            )
        except KeyboardInterrupt:
            print("\nstopped by user")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"Webhook turn failed: {exc}")
            return 1

        _print_ncco(ncco)
        for action in ncco:
            if action.get("action") == "talk":
                _speak_text(
                    action.get("text", ""),
                    api_key=eleven_key,
                    voice_id=eleven_voice_id,
                    sample_rate=args.output_rate,
                )

        if any(action.get("action") == "hangup" for action in ncco):
            print("call ended by agent")
            return 0


if __name__ == "__main__":
    sys.exit(main())
