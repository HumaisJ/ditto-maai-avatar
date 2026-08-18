from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPU_SCRIPTS = (
    PROJECT_ROOT / "scripts" / "gpu" / "setup_ditto_env.ps1",
    PROJECT_ROOT / "scripts" / "gpu" / "publish_checkpoint.ps1",
)


@pytest.mark.skipif(shutil.which("powershell") is None, reason="PowerShell is unavailable")
@pytest.mark.parametrize("script_path", GPU_SCRIPTS)
def test_powershell_script_parses(script_path: Path) -> None:
    command = (
        "$errors=$null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{script_path}', "
        "[ref]$null, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_setup_script_protects_conda_base() -> None:
    content = GPU_SCRIPTS[0].read_text(encoding="utf-8")
    assert "conda list -n base --json" in content
    assert "conda install -n base" not in content
    assert "conda env update -n base" not in content
    assert "conda env remove -n base" not in content


def test_setup_script_does_not_install_ditto_or_tensorrt() -> None:
    content = GPU_SCRIPTS[0].read_text(encoding="utf-8").casefold()
    assert "tensorrt" not in content
    assert "ditto-talkinghead" not in content
    assert "huggingface" not in content


def test_setup_script_uses_selected_gpu_without_requiring_git() -> None:
    content = GPU_SCRIPTS[0].read_text(encoding="utf-8")
    assert '[int]$GpuIndex = 0' in content
    assert '--id=$GpuIndex' in content
    assert '$env:CUDA_VISIBLE_DEVICES = "$GpuIndex"' in content
    assert '@("conda", "git", "nvidia-smi")' not in content
    assert "git lfs version" not in content
    assert "memory.free" in content
    assert "compute-only process" in content
    assert "Stop-Process" not in content
    assert "taskkill" not in content.casefold()


@pytest.mark.skipif(shutil.which("powershell") is None, reason="PowerShell is unavailable")
def test_failure_report_number_accepts_measure_object_double() -> None:
    content = GPU_SCRIPTS[0].read_text(encoding="utf-8")
    assert '"D2-GPU-ENV-{0:D4}" -f ([int]$next)' in content

    command = '$next = [double]2; "D2-GPU-ENV-{0:D4}" -f ([int]$next)'
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "D2-GPU-ENV-0002"


@pytest.mark.skipif(shutil.which("powershell") is None, reason="PowerShell is unavailable")
def test_powershell_wddm_process_pattern_blocks_only_compute_types() -> None:
    command = (
        "$pattern='^\\|\\s*0\\s+(?:N/A|\\d+)\\s+(?:N/A|\\d+)\\s+\\d+\\s+"
        "(?:C|M|M\\+C)\\s+'; "
        "$desktop='| 0 N/A N/A 2444 C+G C:\\Windows\\explorer.exe N/A |'; "
        "$compute='| 0 N/A N/A 47624 C D:\\env\\python.exe N/A |'; "
        "if ($desktop -match $pattern) { exit 1 }; "
        "if ($compute -notmatch $pattern) { exit 1 }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
