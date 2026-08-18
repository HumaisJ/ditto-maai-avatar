"""Run one approved controlled offline Ditto inference for D4, D5, or a D7 batch item."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_ditto_install import validate_environment, validate_runtime  # noqa: E402
from scripts.ditto_install_common import load_config  # noqa: E402
from src.avatar.ditto_adapter import run_ditto_file  # noqa: E402
from src.utils.audio import create_wav_excerpt  # noqa: E402
from src.utils.experiment import ExperimentRun  # noqa: E402
from src.utils.manifest import load_manifest, validate_manifest  # noqa: E402
from src.utils.metrics import effective_generation_fps, real_time_factor  # noqa: E402
from src.utils.system_info import GpuSampler  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "ditto.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "assets" / "manifest.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "experiments"
DEFAULT_RUNTIME = PROJECT_ROOT / ".runtime" / "ditto"
VISUAL_FIELDS = (
    "identity_preservation",
    "lip_sync_quality",
    "facial_naturalness",
    "head_motion_naturalness",
    "upper_portrait_motion_naturalness",
    "artifact_level",
    "overall_realism",
)


class _Tee:
    """Copy third-party Python output to the terminal and experiment log."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


STAGE_PURPOSES = {
    "D4": "D4: one portrait and one short audio produce one valid video.",
    "D5": "D5: another portrait and one complete longer audio produce one valid video.",
    "D7": "D7: one item in the complete 17-pair offline Ditto baseline.",
}


def _read_experiment_status(directory: Path) -> str | None:
    try:
        import json

        payload = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("experiment_id") != directory.name:
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def ensure_experiment_sequence(results_root: Path, stage: str) -> None:
    """Enforce one retained attempt per controlled Ditto stage."""
    existing = {
        path.name: path
        for path in results_root.glob("DITTO-EXP-*")
        if path.is_dir() and path.name.removeprefix("DITTO-EXP-").isdigit()
    } if results_root.is_dir() else {}
    if stage == "D4":
        if existing:
            raise RuntimeError(
                "A DITTO experiment already exists. Preserve and review it before any further run."
            )
        return
    if stage != "D5":
        raise ValueError(f"unsupported controlled Ditto stage: {stage}")

    if set(existing) != {"DITTO-EXP-0001"}:
        raise RuntimeError(
            "D5 requires exactly one prior DITTO-EXP-0001 result and no later attempt."
        )
    if _read_experiment_status(existing["DITTO-EXP-0001"]) != "succeeded":
        raise RuntimeError("D5 requires a successful DITTO-EXP-0001 result.")


def ensure_d7_experiment_sequence(
    results_root: Path,
    runs: list[dict[str, str]],
    *,
    pair_id: str,
    expected_experiment_id: str,
) -> None:
    """Require all earlier D7 items to have succeeded and forbid any later attempt."""
    expected = [(run["pair_id"], run["experiment_id"]) for run in runs]
    try:
        position = expected.index((pair_id, expected_experiment_id))
    except ValueError as exc:
        raise ValueError("D7 pair and experiment ID do not match the approved batch") from exc

    required_ids = ["DITTO-EXP-0001", "DITTO-EXP-0002"]
    required_ids.extend(experiment_id for _, experiment_id in expected[:position])
    existing = {
        path.name: path
        for path in results_root.glob("DITTO-EXP-*")
        if path.is_dir() and path.name.removeprefix("DITTO-EXP-").isdigit()
    } if results_root.is_dir() else {}
    if set(existing) != set(required_ids):
        raise RuntimeError(
            f"D7 {expected_experiment_id} requires exactly the retained successful experiments: "
            + ", ".join(required_ids)
        )
    for experiment_id in required_ids:
        if _read_experiment_status(existing[experiment_id]) != "succeeded":
            raise RuntimeError(f"D7 requires successful prior experiment {experiment_id}")


def _validate_stage_config(config: dict[str, Any], stage: str) -> dict[str, Any]:
    expected_by_stage = {
        "D4": {
            "experiment_id": "DITTO-EXP-0001",
            "pair_id": "P007",
            "audio_mode": "excerpt",
            "clip_start_sec": 0.0,
            "clip_duration_sec": 5.0,
            "derived_filename": "P007_first_5_seconds.wav",
            "seed": 1024,
            "fps": 25,
            "pipeline": "offline",
            "gpu_sample_interval_sec": 1.0,
            "output_filename": "generated.mp4",
        },
        "D5": {
            "experiment_id": "DITTO-EXP-0002",
            "pair_id": "P015",
            "audio_mode": "full",
            "seed": 1024,
            "fps": 25,
            "pipeline": "offline",
            "gpu_sample_interval_sec": 1.0,
            "output_filename": "generated.mp4",
        },
    }
    if stage == "D7":
        stage_config = config.get("d7")
        if not isinstance(stage_config, dict):
            raise ValueError("config/ditto.yaml is missing the D7 configuration")
        expected_common = {
            "batch_id": "DITTO-BATCH-0001",
            "execution": "sequential",
            "failure_policy": "fail_fast",
            "resume_policy": "verified_successes_only",
            "audio_mode": "full",
            "seed": 1024,
            "fps": 25,
            "pipeline": "offline",
            "gpu_sample_interval_sec": 1.0,
            "output_filename": "generated.mp4",
            "minimum_free_disk_mb": 5120,
        }
        for key, expected_value in expected_common.items():
            if stage_config.get(key) != expected_value:
                raise ValueError(
                    f"D7 {key} must be {expected_value!r}, got {stage_config.get(key)!r}"
                )
        runs = stage_config.get("runs")
        if not isinstance(runs, list) or len(runs) != 17:
            raise ValueError("D7 must define exactly 17 ordered runs")
        if any(
            not isinstance(run, dict)
            or set(run) != {"pair_id", "experiment_id"}
            or not isinstance(run["pair_id"], str)
            or not isinstance(run["experiment_id"], str)
            for run in runs
        ):
            raise ValueError("each D7 run must contain only pair_id and experiment_id strings")
        pair_ids = [run["pair_id"] for run in runs]
        experiment_ids = [run["experiment_id"] for run in runs]
        expected_pair_ids = {f"P{number:03d}" for number in range(1, 18)}
        if len(set(pair_ids)) != 17 or set(pair_ids) != expected_pair_ids:
            raise ValueError("D7 runs must contain every manifest pair exactly once")
        if experiment_ids != [f"DITTO-EXP-{number:04d}" for number in range(3, 20)]:
            raise ValueError("D7 experiment IDs must be DITTO-EXP-0003 through DITTO-EXP-0019")
        return stage_config

    expected = expected_by_stage.get(stage)
    if expected is None:
        raise ValueError(f"unsupported controlled Ditto stage: {stage}")
    stage_config = config.get(stage.casefold())
    if not isinstance(stage_config, dict):
        raise ValueError(f"config/ditto.yaml is missing the {stage} configuration")
    for key, expected_value in expected.items():
        if stage_config.get(key) != expected_value:
            raise ValueError(
                f"{stage} {key} must be {expected_value!r}, got {stage_config.get(key)!r}"
            )
    return stage_config


def _prepare_inference_audio(
    *,
    original_audio: Path,
    runtime_root: Path,
    stage: str,
    stage_config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    mode = stage_config["audio_mode"]
    if mode == "full":
        return original_audio, {"mode": "full", "source_unchanged": True}
    if mode != "excerpt":
        raise ValueError(f"unsupported {stage} audio mode: {mode}")
    derived_audio = runtime_root / "d4-inputs" / stage_config["derived_filename"]
    clip = create_wav_excerpt(
        original_audio,
        derived_audio,
        start_sec=float(stage_config["clip_start_sec"]),
        duration_sec=float(stage_config["clip_duration_sec"]),
    )
    return derived_audio, {"mode": "excerpt", **clip}


def _append_visual_review_template(path: Path, stage: str = "D4") -> None:
    if stage in {"D5", "D7"}:
        instruction = (
            "Use 1 = unacceptable, 2 = poor, 3 = usable, 4 = good, and 5 = excellent. "
            "Complete every field after watching the full video."
        )
        stop_message = (
            "Any score of 1 blocks D6 until the D5 result is diagnosed."
            if stage == "D5"
            else "Any score of 1 must be diagnosed before the D7 baseline is concluded."
        )
    else:
        instruction = "Use 1 = unacceptable, 2 = poor, 3 = usable, 4 = good, 5 = excellent, or N/A."
        stop_message = "Any score of 1 blocks D5 until the D4 result is diagnosed."
    lines = [
        f"## {stage} visual review",
        "",
        instruction,
        "",
    ]
    lines.extend(f"- {field}: pending" for field in VISUAL_FIELDS)
    lines.extend(["", stop_message, ""])
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("D4", "D5", "D7"), default="D4")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--pair-id")
    parser.add_argument("--expected-experiment-id")
    parser.add_argument("--batch-id")
    parser.add_argument("--batch-directory", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage = args.stage
    config = load_config(args.config.resolve())
    stage_config = _validate_stage_config(config, stage)
    results_root = args.results_root.resolve()
    if stage == "D7":
        if (
            not args.pair_id
            or not args.expected_experiment_id
            or not args.batch_id
            or args.batch_directory is None
        ):
            raise ValueError(
                "D7 requires --pair-id, --expected-experiment-id, --batch-id, "
                "and --batch-directory"
            )
        if args.batch_id != stage_config["batch_id"]:
            raise ValueError("D7 batch ID does not match the approved configuration")
        batch_directory = args.batch_directory.resolve()
        batch_report = json.loads((batch_directory / "batch.json").read_text(encoding="utf-8"))
        if batch_report.get("batch_id") != args.batch_id or batch_report.get("status") != "running":
            raise RuntimeError("D7 item requires its active approved batch record")
        approved_item = next(
            (
                item
                for item in batch_report.get("runs", [])
                if item.get("pair_id") == args.pair_id
                and item.get("experiment_id") == args.expected_experiment_id
            ),
            None,
        )
        if approved_item is None or approved_item.get("status") != "pending":
            raise RuntimeError("D7 item is not the pending run in its batch record")
        ensure_d7_experiment_sequence(
            results_root,
            stage_config["runs"],
            pair_id=args.pair_id,
            expected_experiment_id=args.expected_experiment_id,
        )
        selected_pair_id = args.pair_id
        expected_experiment_id = args.expected_experiment_id
    else:
        ensure_experiment_sequence(results_root, stage)
        selected_pair_id = stage_config["pair_id"]
        expected_experiment_id = stage_config["experiment_id"]

    entries = load_manifest(args.manifest.resolve())
    validate_manifest(entries, PROJECT_ROOT, expected_count=17)
    selected = next(
        (entry for entry in entries if entry.pair_id == selected_pair_id), None
    )
    if selected is None:
        raise RuntimeError(f"manifest does not contain {stage} pair {selected_pair_id}")

    portrait = (PROJECT_ROOT / selected.portrait_path).resolve()
    original_audio = (PROJECT_ROOT / selected.audio_path).resolve()
    runtime_root = args.runtime_root.resolve()
    inference_audio, audio_preparation = _prepare_inference_audio(
        original_audio=original_audio,
        runtime_root=runtime_root,
        stage=stage,
        stage_config=stage_config,
    )

    source_dir = runtime_root / config["runtime"]["source_dir"]
    checkpoint_dir = runtime_root / config["runtime"]["checkpoint_dir"]
    source_manifest = runtime_root / "source-manifest.json"
    checkpoint_manifest = runtime_root / "checkpoint-manifest.json"
    run_config = {
        "stage": stage,
        "pair_id": selected.pair_id,
        "portrait_path": selected.portrait_path,
        "original_audio_path": selected.audio_path,
        "inference_audio_path": str(inference_audio.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "audio_preparation": audio_preparation,
        "inference": stage_config,
        "ditto": config,
    }
    if stage == "D7":
        run_config["batch_id"] = args.batch_id
        run_config["batch_path"] = str(batch_directory.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        )
        run_config["expected_experiment_id"] = expected_experiment_id
    run = ExperimentRun.create(
        project_root=PROJECT_ROOT,
        results_root=results_root,
        kind="DITTO",
        purpose=STAGE_PURPOSES[stage],
        config=run_config,
        inputs={
            "portrait": portrait,
            "original_audio": original_audio,
            "inference_audio": inference_audio,
        },
        model={
            "name": "ditto-talkinghead",
            "backend": config["backend"],
            "source_revision": config["source"]["revision"],
            "checkpoint_revision": config["checkpoints"]["revision"],
            "pipeline": "stream_pipeline_offline.StreamSDK",
        },
    )
    if run.experiment_id != expected_experiment_id:
        with run:
            raise RuntimeError(
                f"{stage} allocated {run.experiment_id}; expected {expected_experiment_id}"
            )
    output = run.directory / stage_config["output_filename"]
    run.set_output_path(output)
    _append_visual_review_template(run.directory / "notes.md", stage)

    try:
        with run:
            environment, environment_errors = validate_environment(config, args.gpu_index)
            runtime, runtime_errors = validate_runtime(
                config,
                source_dir,
                checkpoint_dir,
                source_manifest,
                checkpoint_manifest,
            )
            errors = environment_errors + runtime_errors
            if errors:
                raise RuntimeError(f"{stage} preflight failed: " + "; ".join(errors))
            run.logger.info("%s environment and pinned runtime preflight passed", stage)

            sampler = GpuSampler(
                run.directory / "gpu.csv",
                gpu_index=args.gpu_index,
                interval_sec=float(stage_config["gpu_sample_interval_sec"]),
            )
            try:
                with sampler:
                    with (run.directory / "console.log").open(
                        "a", encoding="utf-8", newline="\n"
                    ) as transcript:
                        stdout = _Tee(sys.stdout, transcript)
                        stderr = _Tee(sys.stderr, transcript)
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            metrics = run_ditto_file(
                                source_dir=source_dir,
                                checkpoint_dir=checkpoint_dir,
                                config_file=config["checkpoints"]["config_file"],
                                data_root=config["checkpoints"]["data_root"],
                                portrait_path=portrait,
                                audio_path=inference_audio,
                                output_path=output,
                                seed=int(stage_config["seed"]),
                                expected_fps=int(stage_config["fps"]),
                            )
            finally:
                run.record_metrics(sampler.summary())

            metrics["real_time_factor"] = real_time_factor(
                float(metrics["inference_time_sec"]), float(metrics["audio_duration_sec"])
            )
            metrics["effective_generation_fps"] = effective_generation_fps(
                int(metrics["output_frame_count"]), float(metrics["inference_time_sec"])
            )
            metrics["selected_gpu_index"] = args.gpu_index
            metrics["environment_preflight"] = environment
            metrics["runtime_preflight"] = runtime
            run.record_metrics(metrics)
            run.logger.info("%s generated and validated %s", stage, output)
    except Exception as exc:
        print(f"{stage} experiment failed: {run.experiment_id}: {exc}", file=sys.stderr)
        print(f"Preserved results: {run.directory}", file=sys.stderr)
        return 1

    print(f"{stage} experiment completed: {run.experiment_id}")
    print(f"Results: {run.directory}")
    if stage == "D7":
        print("Control returns to DITTO-BATCH-0001; do not launch this item directly.")
        return 0
    next_stage = "D5" if stage == "D4" else "D6"
    print(
        "Stop here. Copy this directory to the PC and complete the visual review "
        f"before {next_stage}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
