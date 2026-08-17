from __future__ import annotations

import csv
import wave
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from src.utils.manifest import (
    MANIFEST_FIELDS,
    ManifestEntry,
    ManifestValidationError,
    load_manifest,
    probe_wav_duration,
    validate_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path) -> None:
    Image.new("RGB", (4, 4), color="white").save(path)


def _write_wav(path: Path, *, frames: int = 8000, rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * frames)


def _valid_entry(tmp_path: Path) -> ManifestEntry:
    portrait = tmp_path / "portrait.jpg"
    audio = tmp_path / "audio.wav"
    _write_image(portrait)
    _write_wav(audio)
    return ManifestEntry("P001", portrait.name, audio.name, 1.0, "und")


def test_real_manifest_has_17_valid_pairs() -> None:
    entries = load_manifest(PROJECT_ROOT / "assets" / "manifest.csv")
    validate_manifest(entries, PROJECT_ROOT, expected_count=17)
    assert [entry.pair_id for entry in entries] == [f"P{index:03d}" for index in range(1, 18)]


def test_probe_wav_duration(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _write_wav(audio, frames=4000, rate=8000)
    assert probe_wav_duration(audio) == pytest.approx(0.5)


def test_load_manifest_rejects_wrong_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("pair_id,portrait_path\nP001,p.jpg\n", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="expected columns"):
        load_manifest(manifest)


def test_load_manifest_rejects_non_numeric_duration(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "P001",
                "portrait_path": "p.jpg",
                "audio_path": "a.wav",
                "audio_duration_sec": "not-a-number",
                "language": "und",
                "notes": "",
            }
        )
    with pytest.raises(ManifestValidationError, match="must be a number"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda entry: replace(entry, pair_id="bad"), "pair_id must match"),
        (lambda entry: replace(entry, portrait_path="missing.jpg"), "portrait not found"),
        (lambda entry: replace(entry, audio_path="missing.wav"), "audio not found"),
        (lambda entry: replace(entry, audio_duration_sec=2.0), "duration is"),
        (lambda entry: replace(entry, language=""), "language must not be blank"),
    ],
)
def test_validate_manifest_reports_invalid_entry(tmp_path: Path, mutation, message: str) -> None:
    entry = mutation(_valid_entry(tmp_path))
    with pytest.raises(ManifestValidationError, match=message):
        validate_manifest([entry], tmp_path)


def test_validate_manifest_reports_duplicate_assets(tmp_path: Path) -> None:
    entry = _valid_entry(tmp_path)
    duplicate = replace(entry, pair_id="P002")
    with pytest.raises(ManifestValidationError, match="duplicate portrait path"):
        validate_manifest([entry, duplicate], tmp_path)


def test_validate_manifest_reports_unreadable_media(tmp_path: Path) -> None:
    portrait = tmp_path / "portrait.jpg"
    audio = tmp_path / "audio.wav"
    portrait.write_text("not an image", encoding="utf-8")
    audio.write_text("not audio", encoding="utf-8")
    entry = ManifestEntry("P001", portrait.name, audio.name, 1.0, "und")
    with pytest.raises(ManifestValidationError) as error:
        validate_manifest([entry], tmp_path)
    assert "unreadable portrait" in str(error.value)
    assert "unreadable WAV" in str(error.value)
