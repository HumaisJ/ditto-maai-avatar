"""Create durable, reproducible experiment result directories."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import traceback
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import close_experiment_logger, create_experiment_logger
from src.utils.system_info import collect_system_info, initialize_gpu_csv

KIND_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp using a trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017


def sha256_file(path: Path | str) -> str:
    """Stream a file into a SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def _audio_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return None


def _allocate_experiment_directory(
    results_root: Path, kind: str, id_width: int
) -> tuple[str, Path]:
    results_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{kind}-EXP-"
    existing_numbers = []
    for candidate in results_root.iterdir():
        if candidate.is_dir() and candidate.name.startswith(prefix):
            suffix = candidate.name.removeprefix(prefix)
            if suffix.isdigit():
                existing_numbers.append(int(suffix))
    number = max(existing_numbers, default=0) + 1
    while True:
        experiment_id = f"{prefix}{number:0{id_width}d}"
        directory = results_root / experiment_id
        try:
            directory.mkdir()
        except FileExistsError:
            number += 1
            continue
        return experiment_id, directory


@dataclass(slots=True)
class ExperimentRun:
    """Lifecycle and durable files for one experiment."""

    project_root: Path
    directory: Path
    experiment_id: str
    metadata: dict[str, Any]
    logger: logging.Logger
    _metrics: dict[str, Any] = field(default_factory=dict)
    _finished: bool = False

    @classmethod
    def create(
        cls,
        *,
        project_root: Path | str,
        results_root: Path | str,
        kind: str,
        purpose: str,
        config: dict[str, Any],
        inputs: dict[str, Path | str] | None = None,
        model: dict[str, Any] | None = None,
        id_width: int = 4,
    ) -> ExperimentRun:
        """Allocate and initialize a new experiment directory."""
        normalized_kind = kind.upper()
        if not KIND_PATTERN.fullmatch(normalized_kind):
            raise ValueError("kind must contain only uppercase letters, digits, and underscores")
        if not purpose.strip():
            raise ValueError("purpose must not be blank")
        if id_width <= 0:
            raise ValueError("id_width must be positive")

        root = Path(project_root).resolve()
        results = Path(results_root)
        if not results.is_absolute():
            results = root / results
        experiment_id, directory = _allocate_experiment_directory(
            results.resolve(), normalized_kind, id_width
        )
        logger = create_experiment_logger(experiment_id, directory / "console.log")

        input_metadata: dict[str, Any] = {}
        for name, raw_path in (inputs or {}).items():
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                close_experiment_logger(logger)
                raise FileNotFoundError(f"input does not exist: {path}")
            input_metadata[name] = {
                "path": _display_path(path, root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "duration_sec": _audio_duration(path),
            }

        started = utc_now()
        system = collect_system_info(root)
        metadata = {
            "experiment_id": experiment_id,
            "created_at_utc": started,
            "started_at_utc": started,
            "completed_at_utc": None,
            "purpose": purpose.strip(),
            "status": "running",
            "error": None,
            "git_commit": system["git_commit"],
            "model": model or {},
            "inputs": input_metadata,
            "requested_output_path": None,
            "system": system,
        }
        run = cls(root, directory, experiment_id, metadata, logger)
        _atomic_write_json(directory / "experiment.json", metadata)
        _atomic_write_json(directory / "config.json", config)
        _atomic_write_json(directory / "metrics.json", {})
        initialize_gpu_csv(directory / "gpu.csv")
        (directory / "notes.md").write_text(
            f"# {experiment_id}\n\nPurpose: {purpose.strip()}\n\n## Observations\n\n",
            encoding="utf-8",
        )
        logger.info("Experiment initialized: %s", experiment_id)
        return run

    def __enter__(self) -> ExperimentRun:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> bool:
        if exc_value is None:
            self.finish_success()
        else:
            self.logger.error(
                "Experiment failed: %s",
                exc_value,
                exc_info=(exc_type, exc_value, exc_traceback),
            )
            self.finish_failure(exc_value, exc_traceback)
        return False

    def record_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge and durably save metrics before experiment completion."""
        if self._finished:
            raise RuntimeError("cannot record metrics after experiment completion")
        self._metrics.update(metrics)
        _atomic_write_json(self.directory / "metrics.json", self._metrics)

    def set_output_path(self, output_path: Path | str) -> None:
        """Record the requested output without requiring it to be a video."""
        if self._finished:
            raise RuntimeError("cannot set output after experiment completion")
        path = Path(output_path)
        if not path.is_absolute():
            path = self.directory / path
        self.metadata["requested_output_path"] = _display_path(path, self.project_root)
        _atomic_write_json(self.directory / "experiment.json", self.metadata)

    def finish_success(self) -> None:
        """Mark the experiment successful and close its logger."""
        self._finish("succeeded", None)

    def finish_failure(self, error: BaseException, error_traceback=None) -> None:
        """Mark the experiment failed while retaining all evidence."""
        error_metadata = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error_traceback)),
        }
        self._finish("failed", error_metadata)

    def _finish(self, status: str, error: dict[str, Any] | None) -> None:
        if self._finished:
            return
        self.metadata["status"] = status
        self.metadata["error"] = error
        self.metadata["completed_at_utc"] = utc_now()
        _atomic_write_json(self.directory / "experiment.json", self.metadata)
        _atomic_write_json(self.directory / "metrics.json", self._metrics)
        self.logger.info("Experiment completed with status: %s", status)
        close_experiment_logger(self.logger)
        self._finished = True
