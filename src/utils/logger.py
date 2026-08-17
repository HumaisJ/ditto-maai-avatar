"""Per-experiment logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def create_experiment_logger(experiment_id: str, console_log_path: Path | str) -> logging.Logger:
    """Create an isolated logger that writes identical messages to console and disk."""
    path = Path(console_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"dialogue_avatar.experiment.{experiment_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    close_experiment_logger(logger)
    formatter = logging.Formatter(
        fmt="%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = __import__("time").gmtime

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def close_experiment_logger(logger: logging.Logger) -> None:
    """Flush, close, and detach every handler from an experiment logger."""
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
