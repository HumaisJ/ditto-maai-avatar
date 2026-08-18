"""Run the single approved D4 offline Ditto inference."""

from __future__ import annotations

import argparse
import contextlib
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


def ensure_first_ditto_experiment(results_root: Path) -> None:
    """Refuse to create a second D4 attempt, including after a retained failure."""
    if results_root.is_dir() and any(results_root.glob("DITTO-EXP-*")):
        raise RuntimeError(
            "A DITTO experiment already exists. Preserve and review it before any further run."
        )


def _validate_d4_config(config: dict[str, Any]) -> dict[str, Any]:
    d4 = config.get("d4")
    if not isinstance(d4, dict):
        raise ValueError("config/ditto.yaml is missing the D4 configuration")
    expected = {
        "pair_id": "P007",
        "clip_start_sec": 0.0,
        "clip_duration_sec": 5.0,
        "seed": 1024,
        "fps": 25,
        "pipeline": "offline",
        "gpu_sample_interval_sec": 1.0,
        "output_filename": "generated.mp4",
    }
    for key, expected_value in expected.items():
        if d4.get(key) != expected_value:
            raise ValueError(f"D4 {key} must be {expected_value!r}, got {d4.get(key)!r}")
    return d4


def _append_visual_review_template(path: Path) -> None:
    lines = [
        "## D4 visual review",
        "",
        "Use 1 = unacceptable, 2 = poor, 3 = usable, 4 = good, 5 = excellent, or N/A.",
        "",
    ]
    lines.extend(f"- {field}: pending" for field in VISUAL_FIELDS)
    lines.extend(["", "Any score of 1 blocks D5 until the D4 result is diagnosed.", ""])
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config.resolve())
    d4 = _validate_d4_config(config)
    results_root = args.results_root.resolve()
    ensure_first_ditto_experiment(results_root)

    entries = load_manifest(args.manifest.resolve())
    validate_manifest(entries, PROJECT_ROOT, expected_count=17)
    selected = next((entry for entry in entries if entry.pair_id == d4["pair_id"]), None)
    if selected is None:
        raise RuntimeError(f"manifest does not contain D4 pair {d4['pair_id']}")

    portrait = (PROJECT_ROOT / selected.portrait_path).resolve()
    original_audio = (PROJECT_ROOT / selected.audio_path).resolve()
    runtime_root = args.runtime_root.resolve()
    derived_audio = runtime_root / "d4-inputs" / "P007_first_5_seconds.wav"
    clip = create_wav_excerpt(
        original_audio,
        derived_audio,
        start_sec=float(d4["clip_start_sec"]),
        duration_sec=float(d4["clip_duration_sec"]),
    )

    source_dir = runtime_root / config["runtime"]["source_dir"]
    checkpoint_dir = runtime_root / config["runtime"]["checkpoint_dir"]
    source_manifest = runtime_root / "source-manifest.json"
    checkpoint_manifest = runtime_root / "checkpoint-manifest.json"
    run_config = {
        "stage": "D4",
        "pair_id": selected.pair_id,
        "portrait_path": selected.portrait_path,
        "original_audio_path": selected.audio_path,
        "derived_audio_path": str(derived_audio.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "clip": clip,
        "inference": d4,
        "ditto": config,
    }
    run = ExperimentRun.create(
        project_root=PROJECT_ROOT,
        results_root=results_root,
        kind="DITTO",
        purpose="D4: one portrait and one short audio produce one valid video.",
        config=run_config,
        inputs={
            "portrait": portrait,
            "original_audio": original_audio,
            "inference_audio": derived_audio,
        },
        model={
            "name": "ditto-talkinghead",
            "backend": config["backend"],
            "source_revision": config["source"]["revision"],
            "checkpoint_revision": config["checkpoints"]["revision"],
            "pipeline": "stream_pipeline_offline.StreamSDK",
        },
    )
    output = run.directory / d4["output_filename"]
    run.set_output_path(output)
    _append_visual_review_template(run.directory / "notes.md")

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
                raise RuntimeError("D4 preflight failed: " + "; ".join(errors))
            run.logger.info("D4 environment and pinned runtime preflight passed")

            sampler = GpuSampler(
                run.directory / "gpu.csv",
                gpu_index=args.gpu_index,
                interval_sec=float(d4["gpu_sample_interval_sec"]),
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
                                audio_path=derived_audio,
                                output_path=output,
                                seed=int(d4["seed"]),
                                expected_fps=int(d4["fps"]),
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
            run.logger.info("D4 generated and validated %s", output)
    except Exception as exc:
        print(f"D4 experiment failed: {run.experiment_id}: {exc}", file=sys.stderr)
        print(f"Preserved results: {run.directory}", file=sys.stderr)
        return 1

    print(f"D4 experiment completed: {run.experiment_id}")
    print(f"Results: {run.directory}")
    print("Stop here. Copy this directory to the PC and complete the visual review before D5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
