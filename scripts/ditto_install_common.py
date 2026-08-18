"""Shared, side-effect-free helpers for the D3 Ditto installation checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "ditto.yaml"
D3_PREFIX = "D3-DITTO-INSTALL-"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and minimally validate the tracked D3 configuration."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("backend") != "pytorch":
        raise ValueError("Ditto configuration must select the PyTorch backend")
    required = data.get("checkpoints", {}).get("required_files")
    if not isinstance(required, list) or len(required) != 12 or len(set(required)) != 12:
        raise ValueError("Ditto configuration must contain 12 unique required checkpoint files")
    for relative in required:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe checkpoint path: {relative!r}")
    return data


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading large checkpoints into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def verify_file_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Validate every listed regular file and reject unsafe manifest paths."""
    errors: list[str] = []
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return ["manifest does not contain a non-empty files list"]

    for entry in files:
        if not isinstance(entry, dict):
            errors.append("manifest contains a non-object file entry")
            continue
        relative = entry.get("path")
        if not isinstance(relative, str):
            errors.append("manifest file entry has no path")
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"manifest contains unsafe path: {relative}")
            continue
        candidate = root / relative_path
        if not candidate.is_file():
            errors.append(f"required file is missing: {relative}")
            continue
        expected_size = entry.get("size")
        if isinstance(expected_size, int) and candidate.stat().st_size != expected_size:
            errors.append(f"file size differs: {relative}")
            continue
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or sha256_file(candidate) != expected_hash:
            errors.append(f"SHA-256 differs: {relative}")
    return errors


def allocate_report_directory(output_root: Path) -> Path:
    """Atomically allocate the next preserved D3 report directory."""
    output_root.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in output_root.iterdir():
        suffix = path.name.removeprefix(D3_PREFIX)
        if path.is_dir() and path.name.startswith(D3_PREFIX) and suffix.isdigit():
            numbers.append(int(suffix))
    number = max(numbers, default=0) + 1
    while True:
        candidate = output_root / f"{D3_PREFIX}{number:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            number += 1
            continue
        return candidate
