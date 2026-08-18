from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_ditto_batch, test_ditto_file
from src.utils.manifest import load_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_base_evidence(experiments_root: Path, *, d5_score: float = 3.0) -> None:
    for experiment_id in ("DITTO-EXP-0001", "DITTO-EXP-0002"):
        directory = experiments_root / experiment_id
        directory.mkdir(parents=True)
        _write_json(
            directory / "experiment.json",
            {"experiment_id": experiment_id, "status": "succeeded"},
        )
    notes = "\n".join(
        f"- {field}: {d5_score}" for field in test_ditto_file.VISUAL_FIELDS
    )
    (experiments_root / "DITTO-EXP-0002" / "notes.md").write_text(
        notes + "\n", encoding="utf-8"
    )


def _arguments(tmp_path: Path, *, resume: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        config=run_ditto_batch.DEFAULT_CONFIG,
        manifest=run_ditto_batch.DEFAULT_MANIFEST,
        runtime_root=tmp_path / "runtime",
        experiments_root=tmp_path / "experiments",
        batches_root=tmp_path / "batches",
        gpu_index=0,
        resume=resume,
    )


def _install_preflight_mocks(monkeypatch) -> None:
    monkeypatch.setattr(run_ditto_batch, "validate_runtime", lambda *args: ({}, []))
    monkeypatch.setattr(run_ditto_batch, "validate_environment", lambda *args: ({}, []))


def _create_child_result(command: list[str], *, succeeded: bool = True) -> None:
    def option(name: str) -> str:
        return command[command.index(name) + 1]

    experiments_root = Path(option("--results-root"))
    experiment_id = option("--expected-experiment-id")
    pair_id = option("--pair-id")
    directory = experiments_root / experiment_id
    directory.mkdir()
    error = None if succeeded else {"type": "RuntimeError", "message": "simulated failure"}
    _write_json(
        directory / "experiment.json",
        {
            "experiment_id": experiment_id,
            "status": "succeeded" if succeeded else "failed",
            "error": error,
            "requested_output_path": f"results/experiments/{experiment_id}/generated.mp4",
        },
    )
    _write_json(
        directory / "config.json",
        {"stage": "D7", "pair_id": pair_id, "batch_id": "DITTO-BATCH-0001"},
    )
    metrics = {}
    if succeeded:
        (directory / "generated.mp4").write_bytes(b"validated video")
        metrics = {
            "output_validated": True,
            "output_audio_stream_decoded": True,
            "output_size_bytes": 15,
            "output_frame_count": 100,
            "output_fps": 25.0,
            "output_video_duration_sec": 4.0,
            "inference_time_sec": 8.0,
            "total_time_sec": 9.0,
            "real_time_factor": 2.0,
            "peak_vram_mb": 5400.0,
            "average_gpu_utilization_percent": 70.0,
            "maximum_gpu_utilization_percent": 95.0,
        }
    _write_json(directory / "metrics.json", metrics)


def test_d7_configuration_contains_all_pairs_once_in_duration_order() -> None:
    config = run_ditto_batch.load_config(run_ditto_batch.DEFAULT_CONFIG)
    entries = load_manifest(run_ditto_batch.DEFAULT_MANIFEST)
    d7, plan = run_ditto_batch.validate_d7_plan(config, entries)

    assert d7["batch_id"] == "DITTO-BATCH-0001"
    assert d7["execution"] == "sequential"
    assert d7["failure_policy"] == "fail_fast"
    assert [item["experiment_id"] for item in plan] == [
        f"DITTO-EXP-{number:04d}" for number in range(3, 20)
    ]
    assert {item["pair_id"] for item in plan} == {
        f"P{number:03d}" for number in range(1, 18)
    }
    durations = [item["audio_duration_sec"] for item in plan]
    assert durations == sorted(durations)
    assert sum(durations) == pytest.approx(3064.867)


def test_d7_prerequisites_require_complete_nonblocking_d5_review(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    _write_base_evidence(experiments)
    scores = run_ditto_batch.validate_d7_prerequisites(experiments)
    assert set(scores) == set(test_ditto_file.VISUAL_FIELDS)

    notes = experiments / "DITTO-EXP-0002" / "notes.md"
    notes.write_text(notes.read_text(encoding="utf-8").replace("3.0", "1", 1), encoding="utf-8")
    with pytest.raises(RuntimeError, match="score of 1"):
        run_ditto_batch.validate_d7_prerequisites(experiments)


def test_d7_batch_records_all_17_successes_sequentially(tmp_path: Path, monkeypatch) -> None:
    args = _arguments(tmp_path)
    _write_base_evidence(args.experiments_root)
    _install_preflight_mocks(monkeypatch)
    calls: list[str] = []

    def successful_child(command: list[str], batch_directory: Path) -> int:
        del batch_directory
        calls.append(command[command.index("--pair-id") + 1])
        _create_child_result(command)
        return 0

    monkeypatch.setattr(run_ditto_batch, "_run_child", successful_child)
    assert run_ditto_batch.execute_batch(args) == 0
    assert len(calls) == 17

    batch_directory = args.batches_root / "DITTO-BATCH-0001"
    batch = json.loads((batch_directory / "batch.json").read_text(encoding="utf-8"))
    assert batch["status"] == "technical_succeeded"
    assert batch["completed_run_count"] == 17
    assert batch["failed_run_count"] == 0
    assert batch["remaining_run_count"] == 0
    assert batch["aggregate_metrics"]["successful_run_count"] == 17
    with (batch_directory / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 17
    assert all(row["status"] == "succeeded" for row in summary)
    with (batch_directory / "visual_review.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        review = list(csv.DictReader(handle))
    assert len(review) == 17
    assert all(row["overall_realism"] == "pending" for row in review)


def test_d7_batch_stops_at_first_failure_and_preserves_it(tmp_path: Path, monkeypatch) -> None:
    args = _arguments(tmp_path)
    _write_base_evidence(args.experiments_root)
    _install_preflight_mocks(monkeypatch)
    calls = 0

    def failed_child(command: list[str], batch_directory: Path) -> int:
        nonlocal calls
        del batch_directory
        calls += 1
        _create_child_result(command, succeeded=False)
        return 1

    monkeypatch.setattr(run_ditto_batch, "_run_child", failed_child)
    assert run_ditto_batch.execute_batch(args) == 1
    assert calls == 1

    batch_directory = args.batches_root / "DITTO-BATCH-0001"
    batch = json.loads((batch_directory / "batch.json").read_text(encoding="utf-8"))
    assert batch["status"] == "failed"
    assert batch["failed_run_count"] == 1
    with (batch_directory / "failures.csv").open(encoding="utf-8", newline="") as handle:
        failures = list(csv.DictReader(handle))
    assert failures[0]["experiment_id"] == "DITTO-EXP-0003"
    assert "simulated failure" in failures[0]["error"]


def test_d7_resume_skips_verified_success_and_continues_pending_items(
    tmp_path: Path, monkeypatch
) -> None:
    first_args = _arguments(tmp_path)
    _write_base_evidence(first_args.experiments_root)
    _install_preflight_mocks(monkeypatch)
    first_calls = 0

    def interrupted_child(command: list[str], batch_directory: Path) -> int:
        nonlocal first_calls
        del batch_directory
        first_calls += 1
        if first_calls == 2:
            raise KeyboardInterrupt
        _create_child_result(command)
        return 0

    monkeypatch.setattr(run_ditto_batch, "_run_child", interrupted_child)
    assert run_ditto_batch.execute_batch(first_args) == 130
    assert first_calls == 2

    resumed_pairs: list[str] = []

    def resumed_child(command: list[str], batch_directory: Path) -> int:
        del batch_directory
        resumed_pairs.append(command[command.index("--pair-id") + 1])
        _create_child_result(command)
        return 0

    monkeypatch.setattr(run_ditto_batch, "_run_child", resumed_child)
    assert run_ditto_batch.execute_batch(_arguments(tmp_path, resume=True)) == 0
    assert len(resumed_pairs) == 16
    assert "P007" not in resumed_pairs


def test_d7_worker_sequence_refuses_missing_or_later_experiments(tmp_path: Path) -> None:
    config = test_ditto_file.load_config(test_ditto_file.DEFAULT_CONFIG)
    d7 = test_ditto_file._validate_stage_config(config, "D7")
    _write_base_evidence(tmp_path)
    test_ditto_file.ensure_d7_experiment_sequence(
        tmp_path,
        d7["runs"],
        pair_id="P007",
        expected_experiment_id="DITTO-EXP-0003",
    )
    (tmp_path / "DITTO-EXP-0019").mkdir()
    with pytest.raises(RuntimeError, match="requires exactly"):
        test_ditto_file.ensure_d7_experiment_sequence(
            tmp_path,
            d7["runs"],
            pair_id="P007",
            expected_experiment_id="DITTO-EXP-0003",
        )
