"""Create one complete local experiment record without loading an ML model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml_config  # noqa: E402
from src.utils.experiment import ExperimentRun  # noqa: E402
from src.utils.manifest import load_manifest, validate_manifest  # noqa: E402
from src.utils.metrics import real_time_factor  # noqa: E402
from src.utils.system_info import append_gpu_sample  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-id", default="P001", help="Manifest pair ID to record.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "assets" / "manifest.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "base.yaml")
    parser.add_argument(
        "--results-root", type=Path, help="Override the configured results directory."
    )
    parser.add_argument("--purpose", default="Verify local experiment recording infrastructure.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_yaml_config(args.config)
    entries = load_manifest(args.manifest)
    validate_manifest(entries, PROJECT_ROOT, expected_count=17)
    selected = next((entry for entry in entries if entry.pair_id == args.pair_id), None)
    if selected is None:
        print(f"Unknown pair ID: {args.pair_id}", file=sys.stderr)
        return 2

    configured_results = config.get("project", {}).get("results_dir", "results/experiments")
    results_root = args.results_root or Path(configured_results)
    id_width = int(config.get("experiment", {}).get("id_width", 4))
    mock_config = config.get("mock", {})

    run = ExperimentRun.create(
        project_root=PROJECT_ROOT,
        results_root=results_root,
        kind="MOCK",
        purpose=args.purpose,
        config=config,
        inputs={"portrait": selected.portrait_path, "audio": selected.audio_path},
        model={
            "name": "mock_renderer",
            "checkpoint_version": None,
            "backend": mock_config.get("backend", "mock"),
            "precision": mock_config.get("precision", "not_applicable"),
        },
        id_width=id_width,
    )

    with run:
        started = perf_counter()
        output = run.directory / "mock_output.txt"
        output.write_text(
            f"Mock output for {selected.pair_id}\n"
            f"Portrait: {selected.portrait_path}\n"
            f"Audio: {selected.audio_path}\n",
            encoding="utf-8",
        )
        inference_time = perf_counter() - started
        run.set_output_path(output)
        run.record_metrics(
            {
                "mock": True,
                "audio_duration_sec": selected.audio_duration_sec,
                "inference_time_sec": inference_time,
                "real_time_factor": real_time_factor(
                    inference_time, selected.audio_duration_sec
                ),
                "output_video_duration_sec": None,
                "output_frame_count": None,
                "effective_generation_fps": None,
                "peak_vram_mb": None,
                "average_gpu_utilization_percent": None,
                "maximum_gpu_utilization_percent": None,
            }
        )
        sampled = append_gpu_sample(run.directory / "gpu.csv")
        run.logger.info("Mock output created; GPU sample available: %s", sampled)

    print(f"Mock experiment completed: {run.experiment_id}")
    print(f"Results: {run.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
