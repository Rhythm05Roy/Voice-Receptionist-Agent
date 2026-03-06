"""Audio format conversion utilities for telephony.

Twilio sends/receives µ-law 8kHz audio.  Our STT and TTS use
PCM 16kHz.  These helpers handle the conversion.
"""

from __future__ import annotations

import audioop
import base64
import struct
from typing import Iterator


def mulaw_to_pcm16(data: bytes) -> bytes:
    """Convert µ-law 8kHz bytes to PCM16 8kHz."""
    return audioop.ulaw2lin(data, 2)


def pcm16_to_mulaw(data: bytes) -> bytes:
    """Convert PCM16 8kHz bytes to µ-law 8kHz."""
    return audioop.lin2ulaw(data, 2)


def resample_8k_to_16k(pcm16_8k: bytes) -> bytes:
    """Upsample PCM16 from 8kHz to 16kHz."""
    return audioop.ratecv(pcm16_8k, 2, 1, 8000, 16000, None)[0]


def resample_16k_to_8k(pcm16_16k: bytes) -> bytes:
    """Downsample PCM16 from 16kHz to 8kHz."""
    return audioop.ratecv(pcm16_16k, 2, 1, 16000, 8000, None)[0]


def decode_twilio_audio(payload: str) -> bytes:
    """Decode base64 µ-law payload from Twilio Media Stream to PCM16 16kHz."""
    mulaw_data = base64.b64decode(payload)
    pcm_8k = mulaw_to_pcm16(mulaw_data)
    return resample_8k_to_16k(pcm_8k)


def encode_for_twilio(pcm16_16k: bytes) -> str:
    """Encode PCM16 16kHz audio to base64 µ-law for Twilio Media Stream."""
    pcm_8k = resample_16k_to_8k(pcm16_16k)
    mulaw = pcm16_to_mulaw(pcm_8k)
    return base64.b64encode(mulaw).decode("ascii")


def chunk_audio(audio: bytes, chunk_size: int = 640) -> Iterator[bytes]:
    """Split audio bytes into chunks for streaming.

    Default chunk_size 640 = 20ms of 16kHz PCM16 audio.
    """
    for i in range(0, len(audio), chunk_size):
        yield audio[i : i + chunk_size]
