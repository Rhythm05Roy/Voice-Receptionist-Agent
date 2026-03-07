from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
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


class PlaybackHandle:
    def __init__(self, audio: np.ndarray, sample_rate: int):
        self._audio = audio
        self._sample_rate = sample_rate
        self._finished = threading.Event()
        self._active = False
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._audio.size == 0:
            self._finished.set()
            return
        with self._lock:
            self._active = True
        sd.play(self._audio, samplerate=self._sample_rate, blocking=False)
        threading.Thread(target=self._wait_thread, daemon=True).start()

    def _wait_thread(self) -> None:
        try:
            sd.wait()
        finally:
            with self._lock:
                self._active = False
            self._finished.set()

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def stop(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
        sd.stop()
        self._finished.set()

    def wait(self) -> None:
        self._finished.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local real voice conversation: mic input + ElevenLabs spoken output."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/api/v1/telephony/webhook/inbound")
    parser.add_argument("--from-number", default="+10000000000")
    parser.add_argument("--to-number", default="+20000000000")
    parser.add_argument("--call-id", default=None, help="Optional call ID. Auto-generated when omitted.")
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=30.0,
        help="Maximum utterance duration in seconds (legacy arg, now dynamic end-of-speech).",
    )
    parser.add_argument("--input-rate", type=int, default=16000)
    parser.add_argument("--output-rate", type=int, default=16000)
    parser.add_argument("--silence-seconds", type=float, default=1.3)
    parser.add_argument("--min-utterance-seconds", type=float, default=0.5)
    parser.add_argument("--start-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--energy-threshold", type=float, default=450.0)
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="Allow user speech to interrupt TTS playback. Use headset to reduce echo-triggered interruptions.",
    )
    return parser.parse_args()


def _wav_bytes_from_pcm(mono: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(mono.astype(np.int16).tobytes())
    return buf.getvalue()


def _capture_until_silence(
    *,
    sample_rate: int,
    max_seconds: float,
    min_seconds: float,
    silence_seconds: float,
    start_timeout_seconds: float,
    energy_threshold: float,
    playback: PlaybackHandle | None = None,
) -> tuple[bytes | None, bool]:
    chunk_seconds = 0.05
    blocksize = max(1, int(sample_rate * chunk_seconds))
    min_chunks_required = max(1, int(min_seconds / chunk_seconds))
    max_chunks = max(min_chunks_required, int(max_seconds / chunk_seconds))
    start_timeout_chunks = max(1, int(start_timeout_seconds / chunk_seconds))

    # Adaptive silence thresholds (in chunks)
    # Short speech (< 2s): use base silence_seconds
    # Medium speech (2-5s): need 2.0s of silence to end
    # Long speech (> 5s): need 2.5s of silence to end
    base_silence_chunks = max(1, int(silence_seconds / chunk_seconds))
    medium_silence_chunks = max(1, int(2.0 / chunk_seconds))
    long_silence_chunks = max(1, int(2.5 / chunk_seconds))

    short_speech_threshold = int(2.0 / chunk_seconds)  # 2 seconds of speech
    long_speech_threshold = int(5.0 / chunk_seconds)    # 5 seconds of speech

    collected: list[np.ndarray] = []
    speech_started = False
    silent_chunks = 0
    speech_chunks = 0  # tracks actual speech (non-silent) chunks
    waited_chunks = 0
    barge_in_triggered = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", blocksize=blocksize) as stream:
        while True:
            chunk, _overflowed = stream.read(blocksize)
            mono = chunk.reshape(-1)
            level = float(np.abs(mono).mean())
            is_speech = level >= energy_threshold

            if not speech_started:
                if is_speech:
                    speech_started = True
                    if playback and playback.is_active():
                        playback.stop()
                        barge_in_triggered = True
                    collected.append(mono.copy())
                    speech_chunks += 1
                    silent_chunks = 0
                    continue

                if playback and playback.is_active():
                    continue

                waited_chunks += 1
                if waited_chunks >= start_timeout_chunks:
                    return None, barge_in_triggered
                continue

            collected.append(mono.copy())
            if is_speech:
                silent_chunks = 0
                speech_chunks += 1
            else:
                silent_chunks += 1

            if len(collected) >= max_chunks:
                break

            # Adaptive silence threshold based on how much speech we've heard
            if speech_chunks >= long_speech_threshold:
                required_silence = long_silence_chunks
            elif speech_chunks >= short_speech_threshold:
                required_silence = medium_silence_chunks
            else:
                required_silence = base_silence_chunks

            if len(collected) >= min_chunks_required and silent_chunks >= required_silence:
                break

    if not collected:
        return None, barge_in_triggered

    mono = np.concatenate(collected)
    return _wav_bytes_from_pcm(mono=mono, sample_rate=sample_rate), barge_in_triggered


def _transcribe(openai_client: OpenAI, wav_bytes: bytes) -> str:
    bio = io.BytesIO(wav_bytes)
    bio.name = "input.wav"
    result = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=bio,
        language="en",
        prompt=(
            "This is a phone call about booking home services like "
            "AC repair, deep cleaning, salon services, and maintenance. "
            "Common words: booking, service, Bahrain, Manama, Riffa, "
            "Muharraq, allergy, appointment, schedule, time, location."
        ),
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
        "model_id": "eleven_turbo_v2_5",  # ~50% faster than multilingual_v2
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.content


def _prepare_audio(text: str, api_key: str, voice_id: str, sample_rate: int) -> np.ndarray:
    if not text.strip():
        return np.array([], dtype=np.float32)
    pcm = _elevenlabs_pcm(text, api_key=api_key, voice_id=voice_id, sample_rate=sample_rate)
    audio_int16 = np.frombuffer(pcm, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def _speak_blocking(text: str, api_key: str, voice_id: str, sample_rate: int) -> None:
    audio = _prepare_audio(text, api_key=api_key, voice_id=voice_id, sample_rate=sample_rate)
    if audio.size == 0:
        return
    sd.play(audio, samplerate=sample_rate, blocking=True)


def _speak_non_blocking(text: str, api_key: str, voice_id: str, sample_rate: int) -> PlaybackHandle:
    audio = _prepare_audio(text, api_key=api_key, voice_id=voice_id, sample_rate=sample_rate)
    handle = PlaybackHandle(audio=audio, sample_rate=sample_rate)
    handle.start()
    return handle


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


def _extract_actions(ncco: list[dict[str, Any]]) -> tuple[list[str], bool, bool]:
    talk_texts: list[str] = []
    has_listen = False
    has_hangup = False
    for action in ncco:
        kind = action.get("action")
        if kind == "talk":
            talk_texts.append(str(action.get("text", "")))
        elif kind == "listen":
            has_listen = True
        elif kind == "hangup":
            has_hangup = True
    return talk_texts, has_listen, has_hangup


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
    print(
        f"capture mode: dynamic (max={args.record_seconds:.1f}s, silence={args.silence_seconds:.1f}s, "
        f"start-timeout={args.start_timeout_seconds:.1f}s)"
    )
    if args.barge_in:
        print("barge-in: enabled (use headset for best results)")
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

    while True:
        _print_ncco(ncco)
        talk_texts, has_listen, has_hangup = _extract_actions(ncco)

        if has_hangup:
            # Speak farewell text BEFORE hanging up
            for text in talk_texts:
                if text.strip():
                    _speak_blocking(text, api_key=eleven_key, voice_id=eleven_voice_id, sample_rate=args.output_rate)
            print("call ended by agent")
            return 0

        if not has_listen:
            for text in talk_texts:
                _speak_blocking(text, api_key=eleven_key, voice_id=eleven_voice_id, sample_rate=args.output_rate)
            # If the agent did not ask to listen, send an empty follow-up tick.
            try:
                ncco = _send_turn(
                    base_url=args.base_url,
                    endpoint=args.endpoint,
                    from_number=args.from_number,
                    to_number=args.to_number,
                    call_id=call_id,
                    speech_text=None,
                )
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"Webhook follow-up failed: {exc}")
                return 1

        playback: PlaybackHandle | None = None
        if talk_texts:
            full_text = " ".join(part for part in talk_texts if part.strip())
            if args.barge_in:
                playback = _speak_non_blocking(
                    full_text,
                    api_key=eleven_key,
                    voice_id=eleven_voice_id,
                    sample_rate=args.output_rate,
                )
            else:
                _speak_blocking(
                    full_text,
                    api_key=eleven_key,
                    voice_id=eleven_voice_id,
                    sample_rate=args.output_rate,
                )

        print("\nListening...")
        start_t = time.perf_counter()
        try:
            wav_bytes, barge_triggered = _capture_until_silence(
                sample_rate=args.input_rate,
                max_seconds=args.record_seconds,
                min_seconds=args.min_utterance_seconds,
                silence_seconds=args.silence_seconds,
                start_timeout_seconds=args.start_timeout_seconds,
                energy_threshold=args.energy_threshold,
                playback=playback,
            )
            if playback:
                playback.wait()
        except KeyboardInterrupt:
            print("\nstopped by user")
            return 0
        except Exception as exc:  # noqa: BLE001
            if playback:
                playback.stop()
            print(f"Voice input failed: {exc}")
            continue

        latency_ms = int((time.perf_counter() - start_t) * 1000)
        if barge_triggered:
            print("[barge-in detected: interrupted TTS playback]")

        if not wav_bytes:
            print(f"you: [no speech detected] ({latency_ms} ms)")
            user_text = None
        else:
            try:
                user_text = _transcribe(openai_client, wav_bytes)
            except KeyboardInterrupt:
                print("\nstopped by user")
                return 0
            except Exception as exc:  # noqa: BLE001
                print(f"Transcription failed: {exc}")
                user_text = None

            if user_text:
                print(f"you: {user_text}")
                if user_text.lower() in {"exit", "quit"}:
                    print("stopped")
                    return 0
            else:
                print("you: [empty transcript]")
                user_text = None

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


if __name__ == "__main__":
    sys.exit(main())
