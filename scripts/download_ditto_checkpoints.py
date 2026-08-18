"""Download and verify only the pinned PyTorch Ditto checkpoint subset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.ditto_install_common import DEFAULT_CONFIG, load_config, sha256_file
except ModuleNotFoundError:
    # Direct execution adds scripts/, rather than the repo root, to sys.path.
    from ditto_install_common import DEFAULT_CONFIG, load_config, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017


def _lfs_value(sibling: Any, name: str) -> Any:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get(name)
    return getattr(lfs, name, None)


def download_checkpoints(config_path: Path, destination: Path, cache_dir: Path) -> dict[str, Any]:
    """Download the exact required files and verify their upstream LFS hashes."""
    from huggingface_hub import HfApi, snapshot_download

    config = load_config(config_path)
    checkpoint_config = config["checkpoints"]
    repository = checkpoint_config["repository"]
    revision = checkpoint_config["revision"]
    required_files = checkpoint_config["required_files"]

    destination.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(required_files)} pinned Ditto files; existing valid data is reused...")
    snapshot_download(
        repo_id=repository,
        repo_type="model",
        revision=revision,
        allow_patterns=required_files,
        local_dir=destination,
        cache_dir=cache_dir,
    )

    info = HfApi().model_info(repository, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"model revision resolved to {info.sha!r}, expected {revision!r}")
    siblings = {sibling.rfilename: sibling for sibling in info.siblings}

    files: list[dict[str, Any]] = []
    errors: list[str] = []
    for relative in required_files:
        path = destination / Path(relative)
        sibling = siblings.get(relative)
        if sibling is None:
            errors.append(f"upstream metadata is missing: {relative}")
            continue
        if not path.is_file():
            errors.append(f"downloaded file is missing: {relative}")
            continue
        expected_hash = _lfs_value(sibling, "sha256")
        expected_size = _lfs_value(sibling, "size") or getattr(sibling, "size", None)
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        if expected_hash and actual_hash != expected_hash:
            errors.append(f"SHA-256 differs from upstream: {relative}")
        if expected_size is not None and actual_size != int(expected_size):
            errors.append(f"file size differs from upstream: {relative}")
        files.append(
            {
                "path": relative,
                "size": actual_size,
                "sha256": actual_hash,
                "upstream_sha256": expected_hash,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "repository": repository,
        "revision": revision,
        "files": files,
        "errors": errors,
    }
    manifest_path = destination.parent / "checkpoint-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("; ".join(errors))
    print(f"Checkpoint manifest: {manifest_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    download_checkpoints(
        args.config.resolve(), args.destination.resolve(), args.cache_dir.resolve()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
