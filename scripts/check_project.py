"""Validate the local project assets and manifest without running a model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.manifest import (  # noqa: E402
    ManifestValidationError,
    load_manifest,
    validate_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "assets" / "manifest.csv",
        help="Path to the asset manifest CSV.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        entries = load_manifest(args.manifest)
        validate_manifest(entries, PROJECT_ROOT, expected_count=17)
    except ManifestValidationError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Project validation passed: {len(entries)} portrait/audio pairs are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
