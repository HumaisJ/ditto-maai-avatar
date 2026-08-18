from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts import check_ditto_install
from scripts import ditto_install_common as common
from src.avatar.ditto_windows_compat import (
    BLEND_MODULE,
    blend_images_numpy,
    install_windows_blend_fallback,
)


def _manifest(path: str, content: bytes) -> dict:
    import hashlib

    return {
        "revision": "revision",
        "files": [
            {
                "path": path,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
        "errors": [],
    }


def test_ditto_config_is_pytorch_only_and_has_exact_checkpoint_inventory() -> None:
    config = common.load_config()
    assert config["backend"] == "pytorch"
    assert len(config["checkpoints"]["required_files"]) == 12
    assert config["checkpoints"]["config_file"].endswith("_pytorch.pkl")
    assert config["checkpoints"]["data_root"] == "ditto_pytorch"


def test_verify_file_manifest_accepts_matching_file(tmp_path: Path) -> None:
    content = b"pinned source"
    (tmp_path / "module.py").write_bytes(content)
    assert common.verify_file_manifest(tmp_path, _manifest("module.py", content)) == []


def test_verify_file_manifest_reports_missing_and_changed_files(tmp_path: Path) -> None:
    content = b"original"
    manifest = _manifest("module.py", content)
    assert "missing" in common.verify_file_manifest(tmp_path, manifest)[0]
    (tmp_path / "module.py").write_bytes(b"changed!")
    assert "SHA-256" in common.verify_file_manifest(tmp_path, manifest)[0]


def test_verify_file_manifest_rejects_parent_traversal(tmp_path: Path) -> None:
    errors = common.verify_file_manifest(tmp_path, _manifest("../outside.py", b"data"))
    assert errors == ["manifest contains unsafe path: ../outside.py"]


def test_allocate_d3_report_directory_increments(tmp_path: Path) -> None:
    assert common.allocate_report_directory(tmp_path).name == "D3-DITTO-INSTALL-0001"
    assert common.allocate_report_directory(tmp_path).name == "D3-DITTO-INSTALL-0002"


def test_validate_runtime_accepts_matching_revisions_and_files(tmp_path: Path) -> None:
    config = common.load_config()
    source_root = tmp_path / "source"
    checkpoint_root = tmp_path / "checkpoints"
    source_root.mkdir()
    checkpoint_root.mkdir()
    (source_root / "module.py").write_bytes(b"source")
    source_manifest = _manifest("module.py", b"source")
    source_manifest["revision"] = config["source"]["revision"]

    checkpoint_files = []
    for relative in config["checkpoints"]["required_files"]:
        path = checkpoint_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        checkpoint_files.extend(_manifest(relative, relative.encode())["files"])
    checkpoint_manifest = {
        "revision": config["checkpoints"]["revision"],
        "files": checkpoint_files,
        "errors": [],
    }
    source_manifest_path = tmp_path / "source.json"
    checkpoint_manifest_path = tmp_path / "checkpoints.json"
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    checkpoint_manifest_path.write_text(json.dumps(checkpoint_manifest), encoding="utf-8")

    details, errors = check_ditto_install.validate_runtime(
        config,
        source_root,
        checkpoint_root,
        source_manifest_path,
        checkpoint_manifest_path,
    )
    assert errors == []
    assert details["checkpoint_file_count"] == 12


def test_validate_environment_blocks_busy_gpu_and_changed_torch(monkeypatch) -> None:
    config = common.load_config()
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "avatar-ditto")
    monkeypatch.setattr(
        check_ditto_install,
        "package_versions",
        lambda: {
            "torch": "2.5.1+cu121",
            "torchvision": "0.23.0+cu128",
            "torchaudio": "2.8.0+cu128",
            "onnxruntime-gpu": "1.23.2",
            "mediapipe": "0.10.35",
            "einops": "0.8.1",
        },
    )
    monkeypatch.setattr(
        check_ditto_install,
        "query_nvidia_smi",
        lambda _: (
            {
                "name": "NVIDIA GeForce RTX 5060 Ti",
                "memory_free_mb": 15000.0,
                "utilization_gpu_percent": 0.0,
            },
            "recorded",
            ["compute process"],
        ),
    )
    _, errors = check_ditto_install.validate_environment(config, 0)
    assert any("torch must be" in error for error in errors)
    assert any("compute-only process" in error for error in errors)


def test_model_load_code_does_not_accept_portrait_or_audio_inputs() -> None:
    parser = check_ditto_install.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "portrait" not in destinations
    assert "audio" not in destinations
    assert "output_video" not in destinations


def test_windows_blend_fallback_matches_upstream_kernel_math() -> None:
    mask = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    warped = np.array(
        [[[300.0, -5.0, 50.0], [100.0, 80.0, 60.0]], [[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]]],
        dtype=np.float32,
    )
    rgb = np.array(
        [[[4, 5, 6], [20, 40, 60]], [[100, 120, 140], [200, 210, 220]]], dtype=np.uint8
    )
    result = np.empty((2, 2, 3), dtype=np.uint8)
    blend_images_numpy(mask, warped, rgb, result)

    expected = np.empty_like(result)
    for row in range(2):
        for column in range(2):
            for channel in range(3):
                value = (
                    mask[row, column] * warped[row, column, channel]
                    + (1.0 - mask[row, column]) * rgb[row, column, channel]
                )
                expected[row, column, channel] = int(min(255, max(0, value)))
    np.testing.assert_array_equal(result, expected)


def test_windows_blend_fallback_supplies_ditto_import_name() -> None:
    previous = sys.modules.get(BLEND_MODULE)
    try:
        assert install_windows_blend_fallback() == "numpy_vectorized_windows_fallback"
        assert sys.modules[BLEND_MODULE].blend_images_cy is blend_images_numpy
    finally:
        if previous is None:
            sys.modules.pop(BLEND_MODULE, None)
        else:
            sys.modules[BLEND_MODULE] = previous


def test_ditto_verifier_supports_direct_script_execution() -> None:
    result = subprocess.run(
        [sys.executable, str(common.PROJECT_ROOT / "scripts" / "check_ditto_install.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=common.PROJECT_ROOT.parent,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-dir" in result.stdout
