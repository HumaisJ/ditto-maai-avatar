from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_environment


def _valid_snapshot() -> dict:
    return {
        "operating_system": "Windows",
        "conda_environment": "avatar-ditto",
        "ffmpeg": "ffmpeg.exe",
        "gpu": {
            "name": "NVIDIA GeForce RTX 5060 Ti",
            "memory_total_mb": 16303.0,
            "memory_free_mb": 15100.0,
            "driver_version": "580.10",
            "utilization_gpu_percent": 0.0,
        },
        "compute_processes": [],
        "pytorch": {
            "version": "2.8.0+cu128",
            "cuda_build": "12.8",
            "cuda_available": True,
            "device_capability": [12, 0],
            "compiled_architectures": ["sm_80", "sm_90", "sm_120"],
            "cuda_operation_error": None,
            "cuda_test_sum": 123.0,
        },
    }


@pytest.mark.parametrize(
    ("version", "expected"),
    [("570.65", (570, 65)), ("2.8.0+cu128", (2, 8, 0)), ("driver 580.10.2", (580, 10, 2))],
)
def test_parse_version(version: str, expected: tuple[int, ...]) -> None:
    assert check_environment.parse_version(version) == expected


def test_utc_now_is_python_310_compatible_utc_timestamp() -> None:
    timestamp = check_environment.utc_now()
    assert timestamp.endswith("Z")
    assert "+00:00" not in timestamp


def test_evaluate_environment_accepts_valid_blackwell_snapshot() -> None:
    assert check_environment.evaluate_environment(_valid_snapshot()) == []


def test_query_nvidia_smi_limits_queries_to_selected_gpu(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, timeout: int = 30) -> tuple[int, str]:
        calls.append(command)
        if "--query-gpu=name,memory.total,memory.free,driver_version,utilization.gpu" in command:
            return 0, "NVIDIA GeForce RTX 5060 Ti, 16311, 15153, 591.86, 0"
        return 0, ""

    monkeypatch.setattr(check_environment.shutil, "which", lambda _: "nvidia-smi.exe")
    monkeypatch.setattr(check_environment, "run_command", fake_run)

    gpu, _, processes = check_environment.query_nvidia_smi(gpu_index=0)

    assert gpu is not None
    assert processes == []
    assert len(calls) == 2
    assert all("--id=0" in command for command in calls)


def test_wddm_graphics_processes_are_not_blocking() -> None:
    process_table = """
|    0   N/A  N/A   2444    C+G   C:\\Windows\\explorer.exe   N/A |
|    0   N/A  N/A  16304    C+G   C:\\Microsoft VS Code\\Code.exe   N/A |
"""
    assert check_environment.find_blocking_compute_processes(process_table, 0) == []


def test_compute_only_process_on_selected_gpu_is_blocking() -> None:
    process_table = """
|    0   N/A  N/A   2444    C+G   C:\\Windows\\explorer.exe   N/A |
|    0   N/A  N/A  47624      C   D:\\env\\python.exe   N/A |
|    1   N/A  N/A  57028      C   D:\\other\\python.exe   N/A |
"""
    blocking = check_environment.find_blocking_compute_processes(process_table, 0)
    assert len(blocking) == 1
    assert "47624" in blocking[0]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda value: value.update(conda_environment="base"), "active Conda environment"),
        (lambda value: value.update(operating_system="Linux"), "must run on Windows"),
        (lambda value: value.update(ffmpeg=None), "ffmpeg is not available"),
        (lambda value: value["gpu"].update(name="NVIDIA T600"), "expected GPU name"),
        (lambda value: value["gpu"].update(memory_total_mb=4096), "GPU VRAM"),
        (lambda value: value["gpu"].update(memory_free_mb=4096), "free VRAM"),
        (lambda value: value["gpu"].update(driver_version="560.00"), "NVIDIA driver"),
        (lambda value: value["gpu"].update(utilization_gpu_percent=50), "utilization"),
        (
            lambda value: value.update(compute_processes=["123, python.exe, 1000"]),
            "compute process",
        ),
        (lambda value: value["pytorch"].update(version="2.5.1+cu121"), "version 2.8.0"),
        (lambda value: value["pytorch"].update(cuda_build="12.1"), "CUDA 12.8"),
        (lambda value: value["pytorch"].update(cuda_available=False), "cuda.is_available"),
        (lambda value: value["pytorch"].update(device_capability=[8, 6]), "capability"),
        (lambda value: value["pytorch"].update(compiled_architectures=[]), "sm_120"),
        (
            lambda value: value["pytorch"].update(cuda_operation_error="kernel failed"),
            "CUDA tensor operation failed",
        ),
    ],
)
def test_evaluate_environment_reports_failures(mutation, expected_error: str) -> None:
    snapshot = _valid_snapshot()
    mutation(snapshot)
    errors = check_environment.evaluate_environment(snapshot)
    assert any(expected_error in error for error in errors)


def test_allocate_report_directory_increments(tmp_path: Path) -> None:
    first = check_environment.allocate_report_directory(tmp_path)
    second = check_environment.allocate_report_directory(tmp_path)
    assert first.name == "D2-GPU-ENV-0001"
    assert second.name == "D2-GPU-ENV-0002"


def test_write_report_preserves_failed_checkpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(check_environment, "run_command", lambda *args, **kwargs: (0, "recorded"))
    directory = check_environment.allocate_report_directory(tmp_path)
    check_environment.write_report(
        directory,
        _valid_snapshot(),
        ["controlled failure"],
        nvidia_output="GPU raw output",
    )
    report = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["errors"] == ["controlled failure"]
    assert (directory / "nvidia-smi.txt").read_text(encoding="utf-8").startswith("GPU raw")
    assert (directory / "conda.txt").is_file()
    assert (directory / "pip-freeze.txt").is_file()
