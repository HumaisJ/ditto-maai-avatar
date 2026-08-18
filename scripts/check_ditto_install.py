"""Validate the pinned Ditto source/checkpoints and initialize its models without inference."""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import io
import json
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMPORT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(IMPORT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_PROJECT_ROOT))

from src.avatar.ditto_windows_compat import install_windows_blend_fallback  # noqa: E402

try:
    from scripts.check_environment import query_nvidia_smi
    from scripts.ditto_install_common import (
        DEFAULT_CONFIG,
        PROJECT_ROOT,
        allocate_report_directory,
        load_config,
        read_json,
        verify_file_manifest,
    )
except ModuleNotFoundError:
    # Direct execution adds scripts/, rather than the repo root, to sys.path.
    from check_environment import query_nvidia_smi
    from ditto_install_common import (
        DEFAULT_CONFIG,
        PROJECT_ROOT,
        allocate_report_directory,
        load_config,
        read_json,
        verify_file_manifest,
    )

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "environment"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017


def package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "torchvision",
        "torchaudio",
        "onnxruntime-gpu",
        "mediapipe",
        "einops",
        "numpy",
        "librosa",
        "opencv-python-headless",
        "imageio",
        "scikit-image",
        "huggingface-hub",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def validate_environment(
    config: dict[str, Any], gpu_index: int
) -> tuple[dict[str, Any], list[str]]:
    """Validate identity, pinned packages, and the selected GPU before loading models."""
    errors: list[str] = []
    environment = config["environment"]
    safety = config["safety"]
    prefix_name = Path(os.environ.get("CONDA_PREFIX", "")).name
    default_environment = os.environ.get("CONDA_DEFAULT_ENV")
    if environment["name"] not in {prefix_name, default_environment}:
        errors.append(
            f"active Conda environment must be {environment['name']!r}, "
            f"got {default_environment!r} / {prefix_name!r}"
        )

    versions = package_versions()
    expected_versions = {
        "torch": environment["torch"],
        "torchvision": environment["torchvision"],
        "torchaudio": environment["torchaudio"],
        "onnxruntime-gpu": environment["onnxruntime_gpu"],
        "mediapipe": environment["mediapipe"],
        "einops": environment["einops"],
    }
    for package, expected in expected_versions.items():
        if versions.get(package) != expected:
            errors.append(f"{package} must be {expected}, got {versions.get(package)!r}")

    gpu, nvidia_output, compute_processes = query_nvidia_smi(gpu_index)
    if gpu is None:
        errors.append("selected GPU information is unavailable")
    else:
        if safety["expected_gpu"].casefold() not in str(gpu["name"]).casefold():
            errors.append(f"unexpected selected GPU: {gpu['name']}")
        if gpu["memory_free_mb"] < float(safety["minimum_free_vram_mb"]):
            errors.append("selected GPU has less than the required free VRAM")
        if gpu["utilization_gpu_percent"] > float(safety["maximum_utilization_percent"]):
            errors.append("selected GPU exceeds the utilization safety threshold")
    if compute_processes:
        errors.append("another compute-only process is using the selected GPU")

    snapshot = {
        "operating_system": platform.system(),
        "windows_version": platform.version(),
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable)),
        "conda_default_environment": default_environment,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "selected_gpu_index": gpu_index,
        "gpu": gpu,
        "compute_processes": compute_processes,
        "packages": versions,
        "nvidia_smi": nvidia_output,
    }
    return snapshot, errors


def validate_runtime(
    config: dict[str, Any],
    source_dir: Path,
    checkpoint_dir: Path,
    source_manifest_path: Path,
    checkpoint_manifest_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Validate revisions and every source/checkpoint file before deserialization."""
    errors: list[str] = []
    details: dict[str, Any] = {}
    try:
        source_manifest = read_json(source_manifest_path)
        details["source_revision"] = source_manifest.get("revision")
        if source_manifest.get("revision") != config["source"]["revision"]:
            errors.append("Ditto source revision does not match config/ditto.yaml")
        errors.extend(
            f"source: {error}"
            for error in verify_file_manifest(source_dir, source_manifest)
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"source manifest could not be validated: {exc}")

    try:
        checkpoint_manifest = read_json(checkpoint_manifest_path)
        details["checkpoint_revision"] = checkpoint_manifest.get("revision")
        details["checkpoint_file_count"] = len(checkpoint_manifest.get("files", []))
        if checkpoint_manifest.get("revision") != config["checkpoints"]["revision"]:
            errors.append("checkpoint revision does not match config/ditto.yaml")
        if checkpoint_manifest.get("errors"):
            errors.append("checkpoint manifest contains download/verification errors")
        errors.extend(
            f"checkpoint: {error}"
            for error in verify_file_manifest(checkpoint_dir, checkpoint_manifest)
        )
        actual_paths = {entry.get("path") for entry in checkpoint_manifest.get("files", [])}
        expected_paths = set(config["checkpoints"]["required_files"])
        if actual_paths != expected_paths:
            errors.append("checkpoint manifest does not contain the exact required file set")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"checkpoint manifest could not be validated: {exc}")
    return details, errors


def load_ditto_models(
    config: dict[str, Any], source_dir: Path, checkpoint_dir: Path
) -> tuple[dict[str, Any], str]:
    """Import the CUDA providers and construct StreamSDK once; do not process inputs."""
    output = io.StringIO()
    started = time.perf_counter()
    original_cwd = Path.cwd()
    sys.path.insert(0, str(source_dir))
    compatibility = None
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() is false")
            torch.cuda.init()

            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(f"CUDAExecutionProvider is unavailable: {providers}")

            if platform.system() == "Windows":
                compatibility = install_windows_blend_fallback()
            os.chdir(source_dir)
            from stream_pipeline_online import StreamSDK

            cfg_path = checkpoint_dir / config["checkpoints"]["config_file"]
            data_root = checkpoint_dir / config["checkpoints"]["data_root"]
            sdk = StreamSDK(str(cfg_path), str(data_root))
            torch.cuda.synchronize()
            del sdk
            gc.collect()
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        return {
            "status": "passed",
            "model_load_time_sec": round(elapsed, 3),
            "onnxruntime_providers": providers,
            "ditto_entrypoint": "stream_pipeline_online.StreamSDK",
            "compatibility_overlay": compatibility,
            "inference_executed": False,
        }, output.getvalue()
    finally:
        os.chdir(original_cwd)
        if sys.path and sys.path[0] == str(source_dir):
            sys.path.pop(0)


def write_report(
    directory: Path,
    report: dict[str, Any],
    model_log: str,
    source_manifest_path: Path,
    checkpoint_manifest_path: Path,
) -> None:
    (directory / "installation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    errors = report["errors"]
    lines = [f"D3 Ditto installation verification: {report['status'].upper()}"]
    lines.extend(f"ERROR: {error}" for error in errors)
    if not errors:
        lines.append("Pinned Ditto source, checkpoints, CUDA providers, and model load passed.")
        lines.append("No portrait/audio inference was executed.")
    (directory / "verification.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "model-load.log").write_text(model_log, encoding="utf-8")
    for source, name in (
        (source_manifest_path, "source-manifest.json"),
        (checkpoint_manifest_path, "checkpoint-manifest.json"),
    ):
        if source.is_file():
            shutil.copyfile(source, directory / name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else PROJECT_ROOT / args.output_root
    )
    directory = allocate_report_directory(output_root.resolve())
    errors: list[str] = []
    model_log = ""
    report: dict[str, Any] = {
        "checkpoint_id": directory.name,
        "stage": "D3",
        "status": "failed",
        "created_at_utc": utc_now(),
        "backend": "pytorch",
        "errors": errors,
        "runtime": {},
        "environment": {},
        "model_load": {"status": "not_attempted", "inference_executed": False},
    }
    try:
        config = load_config(args.config.resolve())
        environment, environment_errors = validate_environment(config, args.gpu_index)
        report["environment"] = environment
        errors.extend(environment_errors)
        runtime, runtime_errors = validate_runtime(
            config,
            args.source_dir.resolve(),
            args.checkpoint_dir.resolve(),
            args.source_manifest.resolve(),
            args.checkpoint_manifest.resolve(),
        )
        report["runtime"] = runtime
        errors.extend(runtime_errors)
        if not errors:
            report["model_load"], model_log = load_ditto_models(
                config, args.source_dir.resolve(), args.checkpoint_dir.resolve()
            )
    except Exception as exc:  # Preserve third-party import/CUDA failures in the D3 report.
        errors.append(f"{type(exc).__name__}: {exc}")
        model_log += traceback.format_exc()

    report["status"] = "passed" if not errors else "failed"
    write_report(
        directory,
        report,
        model_log,
        args.source_manifest.resolve(),
        args.checkpoint_manifest.resolve(),
    )
    print(f"D3 report: {directory}")
    print("D3 Ditto validation passed." if not errors else "D3 Ditto validation failed.")
    for error in errors:
        print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
