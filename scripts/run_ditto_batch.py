"""Run the approved D7 17-pair Ditto baseline sequentially with durable batch evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_ditto_install import validate_environment, validate_runtime  # noqa: E402
from scripts.ditto_install_common import load_config  # noqa: E402
from scripts.test_ditto_file import (  # noqa: E402
    VISUAL_FIELDS,
    _validate_stage_config,
)
from src.utils.experiment import sha256_file, utc_now  # noqa: E402
from src.utils.manifest import ManifestEntry, load_manifest, validate_manifest  # noqa: E402
from src.utils.system_info import get_git_commit  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "ditto.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "assets" / "manifest.csv"
DEFAULT_RUNTIME = PROJECT_ROOT / ".runtime" / "ditto"
DEFAULT_EXPERIMENTS = PROJECT_ROOT / "results" / "experiments"
DEFAULT_BATCHES = PROJECT_ROOT / "results" / "batches"
SINGLE_RUNNER = PROJECT_ROOT / "scripts" / "test_ditto_file.py"

SUMMARY_FIELDS = (
    "position",
    "pair_id",
    "experiment_id",
    "status",
    "portrait_path",
    "audio_path",
    "audio_duration_sec",
    "output_path",
    "output_size_bytes",
    "output_frame_count",
    "output_fps",
    "output_video_duration_sec",
    "inference_time_sec",
    "total_time_sec",
    "real_time_factor",
    "peak_vram_mb",
    "average_gpu_utilization_percent",
    "maximum_gpu_utilization_percent",
    "error",
)
FAILURE_FIELDS = ("position", "pair_id", "experiment_id", "error")
VISUAL_REVIEW_FIELDS = (
    "position",
    "pair_id",
    "experiment_id",
    "video_path",
    *VISUAL_FIELDS,
    "observations",
)


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _log(batch_directory: Path, message: str) -> None:
    line = f"{utc_now()} | {message}"
    print(line, flush=True)
    with (batch_directory / "console.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _parse_complete_d5_review(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise RuntimeError("D7 requires the retained D5 notes.md visual review")
    scores: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        for field in VISUAL_FIELDS:
            prefix = f"- {field}:"
            if line.startswith(prefix):
                value = line.removeprefix(prefix).strip().split(maxsplit=1)[0]
                try:
                    scores[field] = float(value)
                except ValueError as exc:
                    raise RuntimeError(f"D5 visual score {field} is not numeric") from exc
    if set(scores) != set(VISUAL_FIELDS):
        raise RuntimeError("D7 requires all seven numeric D5 visual-review scores")
    if any(score < 1 or score > 5 for score in scores.values()):
        raise RuntimeError("D5 visual-review scores must be between 1 and 5")
    if any(score == 1 for score in scores.values()):
        raise RuntimeError("a D5 visual-review score of 1 blocks D7")
    return scores


def _validate_successful_experiment(directory: Path, experiment_id: str) -> None:
    report = _read_json(directory / "experiment.json")
    if report.get("experiment_id") != experiment_id or report.get("status") != "succeeded":
        raise RuntimeError(f"D7 requires retained successful experiment {experiment_id}")


def validate_d7_prerequisites(experiments_root: Path) -> dict[str, float]:
    """Require the reviewed D4/D5 evidence and no unrelated Ditto attempts."""
    for experiment_id in ("DITTO-EXP-0001", "DITTO-EXP-0002"):
        directory = experiments_root / experiment_id
        if not directory.is_dir():
            raise RuntimeError(f"D7 requires retained experiment {experiment_id}")
        _validate_successful_experiment(directory, experiment_id)
    return _parse_complete_d5_review(experiments_root / "DITTO-EXP-0002" / "notes.md")


def validate_d7_plan(
    config: dict[str, Any], entries: list[ManifestEntry]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve and validate the immutable shortest-to-longest D7 plan."""
    d7 = _validate_stage_config(config, "D7")
    by_id = {entry.pair_id: entry for entry in entries}
    ordered: list[dict[str, Any]] = []
    previous_duration = -1.0
    for position, configured in enumerate(d7["runs"], start=1):
        entry = by_id[configured["pair_id"]]
        if entry.audio_duration_sec < previous_duration:
            raise ValueError("D7 runs must be ordered by non-decreasing audio duration")
        previous_duration = entry.audio_duration_sec
        ordered.append(
            {
                "position": position,
                "pair_id": entry.pair_id,
                "experiment_id": configured["experiment_id"],
                "portrait_path": entry.portrait_path,
                "audio_path": entry.audio_path,
                "audio_duration_sec": entry.audio_duration_sec,
            }
        )
    return d7, ordered


def _blank_summary_rows(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in plan:
        row = {field: "" for field in SUMMARY_FIELDS}
        row.update(item)
        row["status"] = "pending"
        rows.append(row)
    return rows


def _blank_visual_rows(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in plan:
        row = {field: "pending" for field in VISUAL_REVIEW_FIELDS}
        row.update(
            {
                "position": item["position"],
                "pair_id": item["pair_id"],
                "experiment_id": item["experiment_id"],
                "video_path": (
                    f"results/experiments/{item['experiment_id']}/generated.mp4"
                ),
                "observations": "",
            }
        )
        rows.append(row)
    return rows


def _sync_batch_files(
    batch_directory: Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    payload["updated_at_utc"] = utc_now()
    payload["completed_run_count"] = sum(row["status"] == "succeeded" for row in rows)
    payload["failed_run_count"] = sum(row["status"] == "failed" for row in rows)
    payload["remaining_run_count"] = sum(row["status"] == "pending" for row in rows)
    payload["runs"] = [
        {
            "position": row["position"],
            "pair_id": row["pair_id"],
            "experiment_id": row["experiment_id"],
            "status": row["status"],
        }
        for row in rows
    ]
    _atomic_write_json(batch_directory / "batch.json", payload)
    _atomic_write_csv(batch_directory / "summary.csv", SUMMARY_FIELDS, rows)
    failures = [
        {field: row[field] for field in FAILURE_FIELDS}
        for row in rows
        if row["status"] == "failed"
    ]
    _atomic_write_csv(batch_directory / "failures.csv", FAILURE_FIELDS, failures)


def _new_batch(
    *,
    batch_directory: Path,
    config_path: Path,
    manifest_path: Path,
    d7: dict[str, Any],
    plan: list[dict[str, Any]],
    d5_scores: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if batch_directory.exists():
        raise RuntimeError(
            f"{d7['batch_id']} already exists; use --resume only for an interrupted batch"
        )
    batch_directory.mkdir(parents=True)
    rows = _blank_summary_rows(plan)
    payload = {
        "batch_id": d7["batch_id"],
        "stage": "D7",
        "purpose": "Complete documented offline Ditto baseline across all 17 paired inputs.",
        "status": "running",
        "created_at_utc": utc_now(),
        "started_at_utc": utc_now(),
        "updated_at_utc": None,
        "completed_at_utc": None,
        "git_commit": get_git_commit(PROJECT_ROOT),
        "execution": d7["execution"],
        "failure_policy": d7["failure_policy"],
        "resume_policy": d7["resume_policy"],
        "d6_disposition": "skipped_by_explicit_user_decision",
        "expected_run_count": 17,
        "total_audio_duration_sec": sum(item["audio_duration_sec"] for item in plan),
        "config_path": _relative(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "d5_visual_scores": d5_scores,
        "completed_run_count": 0,
        "failed_run_count": 0,
        "remaining_run_count": 17,
        "aggregate_metrics": {},
        "error": None,
        "runs": [],
    }
    (batch_directory / "console.log").write_text("", encoding="utf-8")
    _atomic_write_csv(
        batch_directory / "visual_review.csv",
        VISUAL_REVIEW_FIELDS,
        _blank_visual_rows(plan),
    )
    _sync_batch_files(batch_directory, payload, rows)
    return payload, rows


def _load_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 17:
        raise RuntimeError("D7 summary.csv must retain exactly 17 rows")
    return rows


def _resume_batch(
    *,
    batch_directory: Path,
    config_path: Path,
    manifest_path: Path,
    d7: dict[str, Any],
    plan: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not batch_directory.is_dir():
        raise RuntimeError(f"cannot resume missing batch {d7['batch_id']}")
    payload = _read_json(batch_directory / "batch.json")
    if payload.get("batch_id") != d7["batch_id"] or payload.get("stage") != "D7":
        raise RuntimeError("existing batch identity does not match D7 configuration")
    if payload.get("status") != "interrupted":
        raise RuntimeError("only a cleanly interrupted D7 batch may be resumed")
    if payload.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("D7 config changed after the batch started")
    if payload.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("asset manifest changed after the batch started")
    rows = _load_summary(batch_directory / "summary.csv")
    expected_identity = [(item["pair_id"], item["experiment_id"]) for item in plan]
    actual_identity = [(row["pair_id"], row["experiment_id"]) for row in rows]
    if actual_identity != expected_identity:
        raise RuntimeError("existing D7 summary order or experiment IDs changed")
    for row, planned in zip(rows, plan, strict=True):
        row.update(
            {
                "position": planned["position"],
                "portrait_path": planned["portrait_path"],
                "audio_path": planned["audio_path"],
                "audio_duration_sec": planned["audio_duration_sec"],
            }
        )
    if any(row["status"] not in {"pending", "succeeded"} for row in rows):
        raise RuntimeError("a failed or ambiguous D7 experiment requires review, not resume")
    payload["status"] = "running"
    payload["error"] = None
    _sync_batch_files(batch_directory, payload, rows)
    return payload, rows


def _experiment_summary(experiments_root: Path, planned: dict[str, Any]) -> dict[str, Any]:
    directory = experiments_root / planned["experiment_id"]
    report = _read_json(directory / "experiment.json")
    run_config = _read_json(directory / "config.json")
    metrics = _read_json(directory / "metrics.json")
    if report.get("experiment_id") != planned["experiment_id"]:
        raise RuntimeError("experiment report identity does not match its directory")
    if run_config.get("stage") != "D7" or run_config.get("pair_id") != planned["pair_id"]:
        raise RuntimeError("D7 experiment configuration does not match the batch plan")
    if run_config.get("batch_id") != "DITTO-BATCH-0001":
        raise RuntimeError("D7 experiment is not linked to DITTO-BATCH-0001")
    error = report.get("error")
    error_text = ""
    if isinstance(error, dict):
        error_text = str(error.get("message") or error.get("type") or error)
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(planned)
    row.update(
        {
            "status": report.get("status", "unknown"),
            "output_path": report.get("requested_output_path") or "",
            "output_size_bytes": metrics.get("output_size_bytes", ""),
            "output_frame_count": metrics.get("output_frame_count", ""),
            "output_fps": metrics.get("output_fps", ""),
            "output_video_duration_sec": metrics.get("output_video_duration_sec", ""),
            "inference_time_sec": metrics.get("inference_time_sec", ""),
            "total_time_sec": metrics.get("total_time_sec", ""),
            "real_time_factor": metrics.get("real_time_factor", ""),
            "peak_vram_mb": metrics.get("peak_vram_mb", ""),
            "average_gpu_utilization_percent": metrics.get(
                "average_gpu_utilization_percent", ""
            ),
            "maximum_gpu_utilization_percent": metrics.get(
                "maximum_gpu_utilization_percent", ""
            ),
            "error": error_text,
        }
    )
    if row["status"] == "succeeded":
        output = directory / "generated.mp4"
        required = (
            metrics.get("output_validated") is True,
            metrics.get("output_audio_stream_decoded") is True,
            isinstance(metrics.get("output_frame_count"), int),
            output.is_file() and output.stat().st_size > 0,
        )
        if not all(required):
            raise RuntimeError(f"{planned['experiment_id']} lacks validated output evidence")
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "successful_run_count": len(rows),
        "total_audio_duration_sec": sum(float(row["audio_duration_sec"]) for row in rows),
        "total_output_frames": sum(int(row["output_frame_count"]) for row in rows),
        "total_output_bytes": sum(int(row["output_size_bytes"]) for row in rows),
        "total_inference_time_sec": sum(float(row["inference_time_sec"]) for row in rows),
        "total_runtime_sec": sum(float(row["total_time_sec"]) for row in rows),
        "maximum_peak_vram_mb": max(float(row["peak_vram_mb"]) for row in rows),
    }


def _run_child(command: list[str], batch_directory: Path) -> int:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    with (batch_directory / "console.log").open(
        "a", encoding="utf-8", newline="\n"
    ) as transcript:
        for line in process.stdout:
            print(line, end="", flush=True)
            transcript.write(line)
            transcript.flush()
    return process.wait()


def _ensure_disk_space(root: Path, minimum_free_mb: int) -> None:
    free_mb = shutil.disk_usage(root).free / (1024 * 1024)
    if free_mb < minimum_free_mb:
        raise RuntimeError(
            f"D7 requires {minimum_free_mb} MiB free disk space; only {free_mb:.0f} MiB remains"
        )


def execute_batch(args: argparse.Namespace) -> int:
    if args.gpu_index != 0:
        raise RuntimeError("D7 is approved only for physical GPU 0")
    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    runtime_root = args.runtime_root.resolve()
    experiments_root = args.experiments_root.resolve()
    batches_root = args.batches_root.resolve()
    config = load_config(config_path)
    entries = load_manifest(manifest_path)
    validate_manifest(entries, PROJECT_ROOT, expected_count=17)
    d7, plan = validate_d7_plan(config, entries)
    d5_scores = validate_d7_prerequisites(experiments_root)
    batch_directory = batches_root / d7["batch_id"]

    source_dir = runtime_root / config["runtime"]["source_dir"]
    checkpoint_dir = runtime_root / config["runtime"]["checkpoint_dir"]
    source_manifest = runtime_root / "source-manifest.json"
    checkpoint_manifest = runtime_root / "checkpoint-manifest.json"
    runtime, runtime_errors = validate_runtime(
        config,
        source_dir,
        checkpoint_dir,
        source_manifest,
        checkpoint_manifest,
    )
    if runtime_errors:
        raise RuntimeError("D7 runtime preflight failed: " + "; ".join(runtime_errors))

    if args.resume:
        payload, rows = _resume_batch(
            batch_directory=batch_directory,
            config_path=config_path,
            manifest_path=manifest_path,
            d7=d7,
            plan=plan,
        )
    else:
        unexpected = [
            path.name
            for path in experiments_root.glob("DITTO-EXP-*")
            if path.is_dir() and path.name not in {"DITTO-EXP-0001", "DITTO-EXP-0002"}
        ]
        if unexpected:
            raise RuntimeError(
                "fresh D7 requires no later Ditto experiments; found " + ", ".join(unexpected)
            )
        payload, rows = _new_batch(
            batch_directory=batch_directory,
            config_path=config_path,
            manifest_path=manifest_path,
            d7=d7,
            plan=plan,
            d5_scores=d5_scores,
        )

    payload["runtime_preflight"] = runtime
    _sync_batch_files(batch_directory, payload, rows)
    _log(batch_directory, f"D7 batch initialized with {len(plan)} sequential complete pairs")

    try:
        for index, planned in enumerate(plan):
            if rows[index]["status"] == "succeeded":
                verified = _experiment_summary(experiments_root, planned)
                if verified["status"] != "succeeded":
                    raise RuntimeError(
                        f"retained {planned['experiment_id']} is not a verified success"
                    )
                rows[index] = verified
                _log(batch_directory, f"Resume verified and skipped {planned['experiment_id']}")
                continue

            _ensure_disk_space(PROJECT_ROOT, int(d7["minimum_free_disk_mb"]))
            environment, environment_errors = validate_environment(config, args.gpu_index)
            if environment_errors:
                payload["status"] = "interrupted"
                payload["error"] = "; ".join(environment_errors)
                payload["last_environment_preflight"] = environment
                _sync_batch_files(batch_directory, payload, rows)
                _log(batch_directory, "Stopped safely before allocating the next experiment")
                return 2

            _log(
                batch_directory,
                f"Starting {planned['experiment_id']} ({planned['pair_id']}, "
                f"{planned['audio_duration_sec']:.3f}s)",
            )
            command = [
                sys.executable,
                str(SINGLE_RUNNER),
                "--stage",
                "D7",
                "--config",
                str(config_path),
                "--manifest",
                str(manifest_path),
                "--runtime-root",
                str(runtime_root),
                "--results-root",
                str(experiments_root),
                "--gpu-index",
                str(args.gpu_index),
                "--pair-id",
                planned["pair_id"],
                "--expected-experiment-id",
                planned["experiment_id"],
                "--batch-id",
                d7["batch_id"],
                "--batch-directory",
                str(batch_directory),
            ]
            return_code = _run_child(command, batch_directory)
            try:
                result = _experiment_summary(experiments_root, planned)
            except Exception as exc:
                result = dict(rows[index])
                result["status"] = "failed"
                result["error"] = f"batch validation failed: {exc}"
            if return_code != 0 and result["status"] != "failed":
                result["status"] = "failed"
                result["error"] = f"single-runner exited with code {return_code}"
            rows[index] = result
            _sync_batch_files(batch_directory, payload, rows)
            if result["status"] != "succeeded":
                payload["status"] = "failed"
                payload["error"] = result["error"] or "D7 experiment failed"
                payload["completed_at_utc"] = utc_now()
                _sync_batch_files(batch_directory, payload, rows)
                _log(batch_directory, f"Fail-fast stop after {planned['experiment_id']}")
                return 1
            _log(batch_directory, f"Validated {planned['experiment_id']}")

        payload["status"] = "technical_succeeded"
        payload["completed_at_utc"] = utc_now()
        payload["aggregate_metrics"] = _aggregate(rows)
        payload["error"] = None
        _sync_batch_files(batch_directory, payload, rows)
        _log(batch_directory, "All 17 D7 experiments passed technical validation")
        return 0
    except KeyboardInterrupt:
        payload["status"] = "interrupted"
        payload["error"] = "operator interrupted the batch"
        _sync_batch_files(batch_directory, payload, rows)
        _log(batch_directory, "D7 interrupted; evidence was preserved")
        return 130
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = str(exc)
        payload["completed_at_utc"] = utc_now()
        _sync_batch_files(batch_directory, payload, rows)
        _log(batch_directory, f"D7 failed: {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--batches-root", type=Path, default=DEFAULT_BATCHES)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute_batch(args)
    except Exception as exc:
        print(f"D7 batch preflight failed: {exc}", file=sys.stderr)
        return 1
    if result == 0:
        print("D7 technical batch completed: DITTO-BATCH-0001")
        print("Copy the batch directory and DITTO-EXP-0003 through DITTO-EXP-0019 to the PC.")
        print("Do not begin another model stage before all 17 videos are reviewed.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
