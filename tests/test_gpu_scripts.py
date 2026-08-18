from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPU_SCRIPTS = (
    PROJECT_ROOT / "scripts" / "gpu" / "setup_ditto_env.ps1",
    PROJECT_ROOT / "scripts" / "gpu" / "publish_checkpoint.ps1",
    PROJECT_ROOT / "scripts" / "gpu" / "prepare_ditto_source.ps1",
    PROJECT_ROOT / "scripts" / "gpu" / "install_ditto.ps1",
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


def test_setup_script_confirms_validator_report_before_skipping_fallback() -> None:
    content = GPU_SCRIPTS[0].read_text(encoding="utf-8")
    assert "$reportCountBeforeValidation" in content
    assert "$reportCountAfterValidation" in content
    assert "$validationReportCreated = $true" not in content


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


def test_d3_installer_reuses_environment_and_protects_other_state() -> None:
    content = GPU_SCRIPTS[3].read_text(encoding="utf-8")
    folded = content.casefold()
    assert "conda env create" not in folded
    assert "conda install -n base" not in folded
    assert "stop-process" not in folded
    assert "taskkill" not in folded
    assert "cuda_visible_devices" in folded
    assert "conda run --no-capture-output -n $environmentname" in folded
    assert "get-condafingerprint -name \"base\"" in folded
    assert "get-torchfingerprint" in folded


def test_d3_uses_pinned_official_revisions_and_excludes_tensorrt() -> None:
    config = (PROJECT_ROOT / "config" / "ditto.yaml").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements" / "ditto.txt").read_text(encoding="utf-8")
    preparation = GPU_SCRIPTS[2].read_text(encoding="utf-8")
    assert "c3e47eee2e626500017a0556b470d6d4182f85e8" in config
    assert "c3e47eee2e626500017a0556b470d6d4182f85e8" in preparation
    assert "e4a2f60328ee7c32af585ac4b3cce299e4c8e254" in config
    assert "onnxruntime-gpu==1.23.2" in requirements
    assert "einops==0.8.1" in requirements
    assert "matplotlib==3.10.8" in requirements
    assert "sounddevice==0.5.5" in requirements
    assert "tensorrt" not in requirements.casefold()
    assert "cuda-python" not in requirements.casefold()
    assert "polygraphy" not in requirements.casefold()
    installer = GPU_SCRIPTS[3].read_text(encoding="utf-8").casefold()
    assert "--no-deps mediapipe==0.10.35" in installer
    assert "opencv-contrib-python" not in installer


def test_d3_installer_rejects_empty_or_outdated_requirements() -> None:
    content = GPU_SCRIPTS[3].read_text(encoding="utf-8").casefold()
    assert "requirements/ditto.txt is empty, outdated" in content
    assert '"pyyaml==6.0.2"' in content
    assert '"huggingface-hub==0.36.0"' in content
    assert '"onnxruntime-gpu==1.23.2"' in content
    assert '"einops==0.8.1"' in content
    assert '"matplotlib==3.10.8"' in content
    assert '"sounddevice==0.5.5"' in content
    assert "d3 dependency imports passed" in content
