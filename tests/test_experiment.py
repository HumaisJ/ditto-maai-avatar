from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_mock_experiment import main as run_mock
from src.utils.experiment import ExperimentRun, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = {
    "experiment.json",
    "config.json",
    "metrics.json",
    "console.log",
    "gpu.csv",
    "notes.md",
}


def _new_run(tmp_path: Path, *, purpose: str = "Test run") -> ExperimentRun:
    input_path = tmp_path / "input.txt"
    input_path.write_text("controlled input", encoding="utf-8")
    return ExperimentRun.create(
        project_root=tmp_path,
        results_root="results",
        kind="MOCK",
        purpose=purpose,
        config={"test": True},
        inputs={"document": input_path},
        model={"name": "mock"},
    )


def test_successful_experiment_writes_complete_record(tmp_path: Path) -> None:
    run = _new_run(tmp_path)
    with run:
        output = run.directory / "mock_output.txt"
        output.write_text("done", encoding="utf-8")
        run.set_output_path(output)
        run.record_metrics({"inference_time_sec": 0.1})

    metadata = json.loads((run.directory / "experiment.json").read_text(encoding="utf-8"))
    metrics = json.loads((run.directory / "metrics.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "succeeded"
    assert metadata["completed_at_utc"] is not None
    assert metadata["inputs"]["document"]["sha256"] == sha256_file(tmp_path / "input.txt")
    assert metrics == {"inference_time_sec": 0.1}
    assert EXPECTED_FILES <= {path.name for path in run.directory.iterdir()}


def test_experiment_ids_increment_without_overwrite(tmp_path: Path) -> None:
    first = _new_run(tmp_path)
    first.finish_success()
    second = _new_run(tmp_path)
    second.finish_success()
    assert first.experiment_id == "MOCK-EXP-0001"
    assert second.experiment_id == "MOCK-EXP-0002"
    assert first.directory.is_dir()
    assert second.directory.is_dir()


def test_failed_experiment_is_retained(tmp_path: Path) -> None:
    run = _new_run(tmp_path)
    with pytest.raises(RuntimeError, match="controlled failure"):
        with run:
            raise RuntimeError("controlled failure")

    metadata = json.loads((run.directory / "experiment.json").read_text(encoding="utf-8"))
    assert run.directory.is_dir()
    assert metadata["status"] == "failed"
    assert metadata["error"]["type"] == "RuntimeError"
    assert "controlled failure" in metadata["error"]["traceback"]
    assert "Experiment failed" in (run.directory / "console.log").read_text(encoding="utf-8")


def test_create_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="purpose"):
        ExperimentRun.create(
            project_root=tmp_path,
            results_root="results",
            kind="MOCK",
            purpose=" ",
            config={},
        )
    with pytest.raises(FileNotFoundError):
        ExperimentRun.create(
            project_root=tmp_path,
            results_root="results",
            kind="MOCK",
            purpose="Missing input",
            config={},
            inputs={"missing": "missing.txt"},
        )


def test_mock_cli_creates_a_successful_experiment(tmp_path: Path) -> None:
    assert run_mock(["--pair-id", "P001", "--results-root", str(tmp_path)]) == 0
    directories = list(tmp_path.glob("MOCK-EXP-*"))
    assert len(directories) == 1
    metadata = json.loads((directories[0] / "experiment.json").read_text(encoding="utf-8"))
    metrics = json.loads((directories[0] / "metrics.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "succeeded"
    assert metadata["inputs"]["audio"]["path"].startswith("assets/audio/")
    assert metrics["mock"] is True
    assert (directories[0] / "mock_output.txt").is_file()


def test_mock_cli_rejects_unknown_pair(tmp_path: Path) -> None:
    assert run_mock(["--pair-id", "P999", "--results-root", str(tmp_path)]) == 2
    assert not list(tmp_path.iterdir())
