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
