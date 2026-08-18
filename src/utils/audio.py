"""Deterministic PCM WAV preparation helpers."""

from __future__ import annotations

import os
import wave
from pathlib import Path


def create_wav_excerpt(
    source_path: Path | str,
    destination_path: Path | str,
    *,
    start_sec: float,
    duration_sec: float,
) -> dict[str, int | float]:
    """Copy an exact time window from a PCM WAV without altering the source."""
    if start_sec < 0:
        raise ValueError("start_sec must not be negative")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")

    source = Path(source_path)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")

    try:
        with wave.open(str(source), "rb") as reader:
            sample_rate = reader.getframerate()
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            if sample_rate <= 0:
                raise ValueError("source WAV has an invalid sample rate")
            start_frame = round(start_sec * sample_rate)
            requested_frames = round(duration_sec * sample_rate)
            if start_frame >= reader.getnframes():
                raise ValueError("excerpt starts after the source WAV ends")
            available_frames = reader.getnframes() - start_frame
            if requested_frames > available_frames:
                raise ValueError("excerpt extends past the source WAV")

            reader.setpos(start_frame)
            frames = reader.readframes(requested_frames)
            expected_bytes = requested_frames * channels * sample_width
            if len(frames) != expected_bytes:
                raise ValueError("source WAV ended before the requested excerpt was read")
            with wave.open(str(temporary), "wb") as writer:
                writer.setparams(
                    (
                        channels,
                        sample_width,
                        sample_rate,
                        requested_frames,
                        reader.getcomptype(),
                        reader.getcompname(),
                    )
                )
                writer.writeframes(frames)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "start_frame": start_frame,
        "frame_count": requested_frames,
        "duration_sec": requested_frames / sample_rate,
    }
