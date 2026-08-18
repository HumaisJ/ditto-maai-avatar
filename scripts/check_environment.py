"""Validate and record the isolated Windows Ditto GPU environment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "environment"
BLOCKING_PROCESS_TYPES = {"C", "M", "M+C"}
PROCESS_ROW_PATTERN = re.compile(
    r"^\|\s*(?P<gpu>\d+)\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+"
    r"(?P<pid>\d+)\s+(?P<type>C\+G|M\+C|C|G|M|O)\s+"
)


def utc_now() -> str:
    # datetime.UTC does not exist in the target Python 3.10 environment.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017


def parse_version(value: str) -> tuple[int, ...]:
    """Extract a comparable numeric version tuple from a tool version string."""
    match = re.search(r"\d+(?:\.\d+)+", value)
    if match is None:
        raise ValueError(f"no numeric version found in {value!r}")
    return tuple(int(part) for part in match.group(0).split("."))


def run_command(command: list[str], *, timeout: int = 30) -> tuple[int, str]:
    """Run a diagnostic command without raising for ordinary command failure."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return result.returncode, output


def find_blocking_compute_processes(process_table: str, gpu_index: int) -> list[str]:
    """Return compute-only process rows for one GPU from standard nvidia-smi output."""
    blocking: list[str] = []
    for line in process_table.splitlines():
        match = PROCESS_ROW_PATTERN.match(line)
        if match is None or int(match.group("gpu")) != gpu_index:
            continue
        if match.group("type") in BLOCKING_PROCESS_TYPES:
            blocking.append(line.strip())
    return blocking


def query_nvidia_smi(gpu_index: int = 0) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Return selected-GPU facts, raw output, and its active compute-process rows."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, "nvidia-smi not found", []

    query = [
        executable,
        "--query-gpu=name,memory.total,memory.free,driver_version,utilization.gpu",
        "--format=csv,noheader,nounits",
        f"--id={gpu_index}",
    ]
    code, raw_gpu = run_command(query)
    if code != 0 or not raw_gpu:
        return None, raw_gpu or "nvidia-smi GPU query failed", []
    values = [value.strip() for value in raw_gpu.splitlines()[0].split(",")]
    if len(values) != 5:
        return None, raw_gpu, []

    process_code, raw_processes = run_command([executable, f"--id={gpu_index}"])
    processes = find_blocking_compute_processes(raw_processes, gpu_index)
    if process_code != 0:
        processes.append("selected GPU process inspection failed")
    try:
        gpu = {
            "name": values[0],
            "memory_total_mb": float(values[1]),
            "memory_free_mb": float(values[2]),
            "driver_version": values[3],
            "utilization_gpu_percent": float(values[4]),
        }
    except ValueError:
        return None, raw_gpu, processes
    combined = raw_gpu
    if processes:
        combined += "\n\nCompute processes:\n" + "\n".join(processes)
    return gpu, combined, processes


def collect_torch_info() -> dict[str, Any]:
    """Import PyTorch and execute a small real CUDA operation."""
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"import_error": str(exc), "cuda_operation_error": None}

    result: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_operation_error": None,
        "device_name": None,
        "device_capability": None,
        "compiled_architectures": [],
    }
    if not result["cuda_available"]:
        return result

    try:
        result["device_name"] = torch.cuda.get_device_name(0)
        result["device_capability"] = list(torch.cuda.get_device_capability(0))
        result["compiled_architectures"] = torch.cuda.get_arch_list()
        left = torch.arange(256, dtype=torch.float32, device="cuda").reshape(16, 16)
        product = left @ left.T
        torch.cuda.synchronize()
        result["cuda_test_sum"] = float(product.sum().cpu().item())
    except Exception as exc:  # CUDA failures surface through several PyTorch exception types.
        result["cuda_operation_error"] = f"{type(exc).__name__}: {exc}"
    return result


def evaluate_environment(
    snapshot: dict[str, Any],
    *,
    expected_environment: str = "avatar-ditto",
    expected_gpu: str = "RTX 5060 Ti",
    min_vram_mb: float = 15_000,
    min_driver: str = "570.65",
    max_utilization_percent: float = 20,
    min_free_vram_mb: float = 12_000,
) -> list[str]:
    """Return every failed D2 acceptance condition."""
    errors: list[str] = []
    if snapshot.get("conda_environment") != expected_environment:
        errors.append(
            f"active Conda environment must be {expected_environment!r}, "
            f"got {snapshot.get('conda_environment')!r}"
        )
    if snapshot.get("operating_system") != "Windows":
        errors.append("D2 environment must run on Windows")
    if snapshot.get("ffmpeg") is None:
        errors.append("ffmpeg is not available")

    gpu = snapshot.get("gpu")
    if not isinstance(gpu, dict):
        errors.append("NVIDIA GPU information is unavailable")
    else:
        if expected_gpu.casefold() not in str(gpu.get("name", "")).casefold():
            errors.append(f"expected GPU name containing {expected_gpu!r}, got {gpu.get('name')!r}")
        if float(gpu.get("memory_total_mb", 0)) < min_vram_mb:
            errors.append(f"GPU VRAM must be at least {min_vram_mb:.0f} MiB")
        if float(gpu.get("memory_free_mb", 0)) < min_free_vram_mb:
            errors.append(f"GPU free VRAM must be at least {min_free_vram_mb:.0f} MiB")
        try:
            driver_ok = parse_version(str(gpu.get("driver_version", ""))) >= parse_version(
                min_driver
            )
        except ValueError:
            driver_ok = False
        if not driver_ok:
            errors.append(f"NVIDIA driver must be at least {min_driver}")
        if float(gpu.get("utilization_gpu_percent", 100)) > max_utilization_percent:
            errors.append(
                f"GPU utilization exceeds the {max_utilization_percent:.0f}% safety threshold"
            )
    if snapshot.get("compute_processes"):
        errors.append("another compute process is using the GPU")

    torch_info = snapshot.get("pytorch")
    if not isinstance(torch_info, dict) or torch_info.get("import_error"):
        errors.append("PyTorch could not be imported")
    else:
        if not str(torch_info.get("version", "")).startswith("2.8.0"):
            errors.append("PyTorch must be version 2.8.0")
        if torch_info.get("cuda_build") != "12.8":
            errors.append("PyTorch must use the CUDA 12.8 build")
        if not torch_info.get("cuda_available"):
            errors.append("torch.cuda.is_available() is false")
        if torch_info.get("device_capability") != [12, 0]:
            errors.append("GPU CUDA capability must be [12, 0]")
        if "sm_120" not in torch_info.get("compiled_architectures", []):
            errors.append("PyTorch binaries do not include sm_120")
        if torch_info.get("cuda_operation_error"):
            errors.append(f"CUDA tensor operation failed: {torch_info['cuda_operation_error']}")
        if torch_info.get("cuda_test_sum") is None:
            errors.append("CUDA tensor operation did not produce a result")
    return errors


def allocate_report_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    numbers = []
    prefix = "D2-GPU-ENV-"
    for path in output_root.iterdir():
        suffix = path.name.removeprefix(prefix)
        if path.is_dir() and path.name.startswith(prefix) and suffix.isdigit():
            numbers.append(int(suffix))
    number = max(numbers, default=0) + 1
    while True:
        candidate = output_root / f"{prefix}{number:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            number += 1
            continue
        return candidate


def write_report(
    directory: Path,
    snapshot: dict[str, Any],
    errors: list[str],
    *,
    nvidia_output: str,
) -> None:
    status = "passed" if not errors else "failed"
    report = {
        "checkpoint_id": directory.name,
        "stage": "D2",
        "status": status,
        "created_at_utc": utc_now(),
        "errors": errors,
        "environment": snapshot,
    }
    (directory / "environment.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "nvidia-smi.txt").write_text(nvidia_output + "\n", encoding="utf-8")

    _, conda_output = run_command(["conda", "list", "-n", "avatar-ditto"])
    (directory / "conda.txt").write_text(conda_output + "\n", encoding="utf-8")
    _, pip_output = run_command([sys.executable, "-m", "pip", "freeze"])
    (directory / "pip-freeze.txt").write_text(pip_output + "\n", encoding="utf-8")

    lines = [f"D2 environment verification: {status.upper()}"]
    lines.extend(f"ERROR: {error}" for error in errors)
    if not errors:
        lines.append("All D2 acceptance checks passed.")
    (directory / "verification.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-gpu", default="RTX 5060 Ti")
    parser.add_argument("--min-vram-mb", type=float, default=15_000)
    parser.add_argument("--min-driver", default="570.65")
    parser.add_argument("--max-utilization", type=float, default=20)
    parser.add_argument("--min-free-vram-mb", type=float, default=12_000)
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gpu, nvidia_output, processes = query_nvidia_smi(args.gpu_index)
    ffmpeg = shutil.which("ffmpeg")
    snapshot = {
        "operating_system": platform.system(),
        "windows_version": platform.version(),
        "python_version": platform.python_version(),
        "python_executable": Path(sys.executable).name,
        "conda_prefix_name": Path(os.environ.get("CONDA_PREFIX", "")).name or None,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "ffmpeg": ffmpeg,
        "selected_gpu_index": args.gpu_index,
        "gpu": gpu,
        "compute_processes": processes,
        "pytorch": collect_torch_info(),
    }
    errors = evaluate_environment(
        snapshot,
        expected_gpu=args.expected_gpu,
        min_vram_mb=args.min_vram_mb,
        min_driver=args.min_driver,
        max_utilization_percent=args.max_utilization,
        min_free_vram_mb=args.min_free_vram_mb,
    )
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    directory = allocate_report_directory(output_root.resolve())
    write_report(directory, snapshot, errors, nvidia_output=nvidia_output)

    print(f"D2 report: {directory}")
    print("D2 environment validation passed." if not errors else "D2 validation failed.")
    for error in errors:
        print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
