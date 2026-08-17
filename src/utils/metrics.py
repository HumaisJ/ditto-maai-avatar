"""Small, model-independent experiment metric calculations."""

from __future__ import annotations


def real_time_factor(inference_time_sec: float, audio_duration_sec: float) -> float:
    """Calculate inference time divided by source audio duration."""
    if inference_time_sec < 0:
        raise ValueError("inference_time_sec must not be negative")
    if audio_duration_sec <= 0:
        raise ValueError("audio_duration_sec must be positive")
    return inference_time_sec / audio_duration_sec


def effective_generation_fps(output_frame_count: int, inference_time_sec: float) -> float:
    """Calculate generated frames per second of inference time."""
    if output_frame_count < 0:
        raise ValueError("output_frame_count must not be negative")
    if inference_time_sec <= 0:
        raise ValueError("inference_time_sec must be positive")
    return output_frame_count / inference_time_sec
