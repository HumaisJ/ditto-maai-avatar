"""Collect reproducibility metadata without requiring GPU dependencies."""

from __future__ import annotations

import csv
import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
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


def query_gpu_info(gpu_index: int = 0) -> dict[str, Any] | None:
    """Query one NVIDIA GPU, returning None when nvidia-smi is unavailable."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    output = _run_command(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
            f"--id={gpu_index}",
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


def query_gpu_sample(gpu_index: int = 0) -> dict[str, float | str] | None:
    """Return one utilization sample for a physical NVIDIA GPU."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    output = _run_command(
        [
            executable,
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            f"--id={gpu_index}",
        ]
    )
    if not output:
        return None
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    if len(values) != 4:
        return None

    try:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),  # noqa: UP017
            "gpu_name": values[0],
            "utilization_gpu_percent": float(values[1]),
            "memory_used_mb": float(values[2]),
            "memory_total_mb": float(values[3]),
        }
    except ValueError:
        return None


def append_gpu_sample(path: Path | str, gpu_index: int = 0) -> bool:
    """Append one NVIDIA utilization sample, returning False when unavailable."""
    row = query_gpu_sample(gpu_index)
    if row is None:
        return False
    with Path(path).open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=GPU_CSV_FIELDS).writerow(row)
    return True


class GpuSampler:
    """Record one physical GPU at a fixed interval during an experiment."""

    def __init__(self, path: Path | str, *, gpu_index: int = 0, interval_sec: float = 1.0):
        if interval_sec <= 0:
            raise ValueError("interval_sec must be positive")
        self.path = Path(path)
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self.samples: list[dict[str, float | str]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuSampler:
        self._sample()
        self._thread = threading.Thread(target=self._run, name="ditto-gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self.interval_sec + 10.0))
        self._sample()
        return False

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._sample()

    def _sample(self) -> None:
        sample = query_gpu_sample(self.gpu_index)
        if sample is None:
            return
        self.samples.append(sample)
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=GPU_CSV_FIELDS).writerow(sample)

    def summary(self) -> dict[str, float | int | None]:
        """Summarize recorded samples without fabricating unavailable GPU data."""
        if not self.samples:
            return {
                "gpu_sample_count": 0,
                "baseline_vram_mb": None,
                "peak_vram_mb": None,
                "peak_vram_increase_mb": None,
                "average_gpu_utilization_percent": None,
                "maximum_gpu_utilization_percent": None,
            }
        memory = [float(sample["memory_used_mb"]) for sample in self.samples]
        utilization = [float(sample["utilization_gpu_percent"]) for sample in self.samples]
        baseline = memory[0]
        peak = max(memory)
        return {
            "gpu_sample_count": len(self.samples),
            "baseline_vram_mb": baseline,
            "peak_vram_mb": peak,
            "peak_vram_increase_mb": max(0.0, peak - baseline),
            "average_gpu_utilization_percent": sum(utilization) / len(utilization),
            "maximum_gpu_utilization_percent": max(utilization),
        }
