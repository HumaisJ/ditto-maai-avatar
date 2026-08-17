"""Load and validate the controlled portrait/audio test manifest."""

from __future__ import annotations

import csv
import re
import wave
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

MANIFEST_FIELDS = (
    "pair_id",
    "portrait_path",
    "audio_path",
    "audio_duration_sec",
    "language",
    "notes",
)
PAIR_ID_PATTERN = re.compile(r"^P\d{3}$")
PORTRAIT_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ManifestValidationError(ValueError):
    """Raised when one or more manifest validation checks fail."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Manifest validation failed:\n- " + "\n- ".join(errors))


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One deterministic portrait/audio test pair."""

    pair_id: str
    portrait_path: str
    audio_path: str
    audio_duration_sec: float
    language: str
    notes: str = ""


def load_manifest(path: Path | str) -> list[ManifestEntry]:
    """Read a CSV manifest and validate its schema and field types."""
    manifest_path = Path(path)
    errors: list[str] = []
    entries: list[ManifestEntry] = []

    try:
        handle = manifest_path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise ManifestValidationError([f"cannot open {manifest_path}: {exc}"]) from exc

    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ManifestValidationError(
                [f"expected columns {MANIFEST_FIELDS}, got {tuple(reader.fieldnames or ())}"]
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                duration = float(row["audio_duration_sec"])
            except (TypeError, ValueError):
                errors.append(f"line {line_number}: audio_duration_sec must be a number")
                continue
            entries.append(
                ManifestEntry(
                    pair_id=row["pair_id"].strip(),
                    portrait_path=row["portrait_path"].strip(),
                    audio_path=row["audio_path"].strip(),
                    audio_duration_sec=duration,
                    language=row["language"].strip(),
                    notes=row["notes"].strip(),
                )
            )

    if errors:
        raise ManifestValidationError(errors)
    return entries


def probe_wav_duration(path: Path | str) -> float:
    """Return the duration of a PCM WAV file in seconds."""
    with wave.open(str(path), "rb") as audio:
        if audio.getframerate() <= 0:
            raise ValueError(f"invalid WAV sample rate: {path}")
        return audio.getnframes() / audio.getframerate()


def _resolve_project_path(project_root: Path, relative_path: str) -> Path:
    candidate = (project_root / relative_path).resolve()
    if not candidate.is_relative_to(project_root):
        raise ValueError("path escapes the project root")
    return candidate


def validate_manifest(
    entries: list[ManifestEntry],
    project_root: Path | str,
    *,
    expected_count: int | None = None,
    duration_tolerance_sec: float = 0.001,
) -> None:
    """Validate identifiers, files, media readability, and recorded durations."""
    root = Path(project_root).resolve()
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_portraits: set[str] = set()
    seen_audio: set[str] = set()

    if expected_count is not None and len(entries) != expected_count:
        errors.append(f"expected {expected_count} entries, found {len(entries)}")

    for entry in entries:
        label = entry.pair_id or "<blank pair_id>"
        if not PAIR_ID_PATTERN.fullmatch(entry.pair_id):
            errors.append(f"{label}: pair_id must match P followed by three digits")
        if entry.pair_id in seen_ids:
            errors.append(f"{label}: duplicate pair_id")
        seen_ids.add(entry.pair_id)

        for value, seen, kind in (
            (entry.portrait_path, seen_portraits, "portrait"),
            (entry.audio_path, seen_audio, "audio"),
        ):
            if value in seen:
                errors.append(f"{label}: duplicate {kind} path {value}")
            seen.add(value)
            if Path(value).is_absolute():
                errors.append(f"{label}: {kind} path must be repository-relative")

        if entry.audio_duration_sec <= 0:
            errors.append(f"{label}: audio_duration_sec must be positive")
        if not entry.language:
            errors.append(f"{label}: language must not be blank")

        try:
            portrait_path = _resolve_project_path(root, entry.portrait_path)
            audio_path = _resolve_project_path(root, entry.audio_path)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue

        if portrait_path.suffix.lower() not in PORTRAIT_EXTENSIONS:
            errors.append(f"{label}: unsupported portrait extension {portrait_path.suffix}")
        elif not portrait_path.is_file():
            errors.append(f"{label}: portrait not found: {entry.portrait_path}")
        else:
            try:
                with Image.open(portrait_path) as image:
                    image.verify()
            except Exception as exc:  # Pillow exposes several decoder-specific errors.
                errors.append(f"{label}: unreadable portrait: {exc}")

        if audio_path.suffix.lower() != ".wav":
            errors.append(f"{label}: audio must be a WAV file")
        elif not audio_path.is_file():
            errors.append(f"{label}: audio not found: {entry.audio_path}")
        else:
            try:
                actual_duration = probe_wav_duration(audio_path)
            except (OSError, EOFError, wave.Error, ValueError) as exc:
                errors.append(f"{label}: unreadable WAV: {exc}")
            else:
                difference = abs(actual_duration - entry.audio_duration_sec)
                if difference > duration_tolerance_sec:
                    errors.append(
                        f"{label}: duration is {actual_duration:.3f}s, "
                        f"manifest records {entry.audio_duration_sec:.3f}s"
                    )

    if errors:
        raise ManifestValidationError(errors)
