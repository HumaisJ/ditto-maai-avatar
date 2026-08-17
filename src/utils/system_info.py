"""Collect reproducibility metadata without requiring GPU dependencies."""

from __future__ import annotations

import csv
import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GPU_CSV_FIELDS = (
    "timestamp_utc",
    "gpu_name",
    "utilization_gpu_percent",
    "memory_used_mb",
    "memory_total_mb",
)


def _run_command(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def get_git_commit(project_root: Path | str) -> str | None:
    """Return the current Git commit, or None outside a usable repository."""
    return _run_command(["git", "rev-parse", "HEAD"], cwd=Path(project_root))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _pytorch_cuda_version() -> str | None:
    if _package_version("torch") is None:
        return None
    try:
        torch = importlib.import_module("torch")
        return torch.version.cuda
    except (AttributeError, ImportError, OSError):
        return None


def query_gpu_info() -> dict[str, Any] | None:
    """Query the first NVIDIA GPU, returning None when nvidia-smi is unavailable."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    output = _run_command(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
    )
    if not output:
        return None
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    if len(values) != 3:
        return None
    try:
        memory_total_mb = float(values[1])
    except ValueError:
        return None
    return {
        "name": values[0],
        "memory_total_mb": memory_total_mb,
        "driver_version": values[2],
    }


def collect_system_info(project_root: Path | str) -> dict[str, Any]:
    """Collect stable local runtime information for experiment metadata."""
    return {
        "git_commit": get_git_commit(project_root),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch_version": _package_version("torch"),
        "cuda_visible_to_pytorch": _pytorch_cuda_version(),
        "gpu": query_gpu_info(),
    }


def initialize_gpu_csv(path: Path | str) -> None:
    """Create a GPU CSV with a stable header and no fabricated samples."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=GPU_CSV_FIELDS).writeheader()


def append_gpu_sample(path: Path | str) -> bool:
    """Append one NVIDIA utilization sample, returning False when unavailable."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False
    output = _run_command(
        [
            executable,
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
    )
    if not output:
        return False
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    if len(values) != 4:
        return False

    row = dict(zip(GPU_CSV_FIELDS[1:], values, strict=True))
    row["timestamp_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    with Path(path).open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=GPU_CSV_FIELDS).writerow(row)
    return True
