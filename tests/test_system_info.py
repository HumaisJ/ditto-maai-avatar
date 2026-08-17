from pathlib import Path

from src.utils import system_info


def test_gpu_csv_remains_header_only_when_nvidia_smi_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "gpu.csv"
    system_info.initialize_gpu_csv(path)
    monkeypatch.setattr(system_info.shutil, "which", lambda _: None)
    assert system_info.append_gpu_sample(path) is False
    assert path.read_text(encoding="utf-8").splitlines() == [
        "timestamp_utc,gpu_name,utilization_gpu_percent,memory_used_mb,memory_total_mb"
    ]
    assert system_info.query_gpu_info() is None


def test_collect_system_info_survives_missing_external_tools(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(system_info, "_run_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(system_info.shutil, "which", lambda _: None)
    result = system_info.collect_system_info(tmp_path)
    assert result["git_commit"] is None
    assert result["gpu"] is None
    assert result["python_version"]
