"""Project-owned adapter for one offline Ditto file inference."""

from __future__ import annotations

import importlib
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from src.avatar.ditto_windows_compat import install_windows_blend_fallback


def _synchronize_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    torch.cuda.synchronize()


def _seed_everything(seed: int, np: Any, torch: Any) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mux_audio(ffmpeg: str, temporary_video: Path, audio: Path, output: Path) -> float:
    started = perf_counter()
    subprocess.run(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(temporary_video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return perf_counter() - started


def validate_output_video(
    output_path: Path | str,
    *,
    ffmpeg_executable: Path | str,
    expected_frame_count: int,
    expected_duration_sec: float,
) -> dict[str, int | float | list[int] | bool]:
    """Decode the video and audio streams and enforce the D4 media contract."""
    output = Path(output_path)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("Ditto did not produce a non-empty MP4")

    cv2 = importlib.import_module("cv2")
    capture = cv2.VideoCapture(str(output))
    if not capture.isOpened():
        raise RuntimeError("generated MP4 cannot be opened as video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded_frames = 0
    try:
        while True:
            readable, _ = capture.read()
            if not readable:
                break
            decoded_frames += 1
    finally:
        capture.release()

    if fps <= 0 or width <= 0 or height <= 0 or decoded_frames <= 0:
        raise RuntimeError("generated MP4 has invalid video properties")
    if abs(decoded_frames - expected_frame_count) > 1:
        raise RuntimeError(
            f"generated MP4 has {decoded_frames} frames; expected {expected_frame_count}"
        )
    duration = decoded_frames / fps
    if abs(duration - expected_duration_sec) > max(0.1, 1.0 / fps):
        raise RuntimeError(
            f"generated MP4 duration is {duration:.3f}s; expected {expected_duration_sec:.3f}s"
        )

    audio_check = subprocess.run(
        [
            str(ffmpeg_executable),
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if audio_check.returncode != 0:
        detail = audio_check.stderr.strip() or "audio stream could not be decoded"
        raise RuntimeError(f"generated MP4 audio validation failed: {detail}")

    return {
        "output_validated": True,
        "output_size_bytes": output.stat().st_size,
        "output_frame_count": decoded_frames,
        "output_fps": fps,
        "output_video_duration_sec": duration,
        "output_resolution": [width, height],
        "output_audio_stream_decoded": True,
    }


def run_ditto_file(
    *,
    source_dir: Path | str,
    checkpoint_dir: Path | str,
    config_file: str,
    data_root: str,
    portrait_path: Path | str,
    audio_path: Path | str,
    output_path: Path | str,
    seed: int = 1024,
    expected_fps: int = 25,
) -> dict[str, int | float | str | list[int] | bool]:
    """Run the pinned offline PyTorch pipeline and return measured D4 metrics."""
    source = Path(source_dir).resolve()
    checkpoints = Path(checkpoint_dir).resolve()
    portrait = Path(portrait_path).resolve()
    audio_path = Path(audio_path).resolve()
    output = Path(output_path).resolve()
    cfg_path = checkpoints / config_file
    model_root = checkpoints / data_root
    for required in (source, checkpoints, portrait, audio_path, cfg_path, model_root):
        if not required.exists():
            raise FileNotFoundError(f"required Ditto path does not exist: {required}")
    output.parent.mkdir(parents=True, exist_ok=True)

    np = importlib.import_module("numpy")
    librosa = importlib.import_module("librosa")
    torch = importlib.import_module("torch")
    imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    _seed_everything(seed, np, torch)
    compatibility = install_windows_blend_fallback()

    original_cwd = Path.cwd()
    sys.path.insert(0, str(source))
    total_started = perf_counter()
    try:
        os.chdir(source)
        model_started = perf_counter()
        module = importlib.import_module("stream_pipeline_offline")
        sdk = module.StreamSDK(str(cfg_path), str(model_root))
        _synchronize_cuda(torch)
        model_load_time = perf_counter() - model_started

        audio_samples, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        if sample_rate != 16000 or len(audio_samples) == 0:
            raise RuntimeError("Ditto audio preprocessing did not produce 16 kHz samples")
        audio_duration = len(audio_samples) / sample_rate
        expected_frames = math.ceil(audio_duration * expected_fps)

        setup_started = perf_counter()
        sdk.setup(str(portrait), str(output), online_mode=False)
        if sdk.online_mode:
            raise RuntimeError("D4 must use Ditto's offline file-inference mode")
        sdk.setup_Nd(N_d=expected_frames)
        _synchronize_cuda(torch)
        setup_time = perf_counter() - setup_started

        generation_started = perf_counter()
        audio_features = sdk.wav2feat.wav2feat(audio_samples)
        sdk.audio2motion_queue.put(audio_features)
        sdk.close()
        _synchronize_cuda(torch)
        generation_time = perf_counter() - generation_started

        temporary_video = Path(sdk.tmp_output_path)
        if not temporary_video.is_file():
            raise RuntimeError("Ditto did not produce its temporary video")
        ffmpeg = str(imageio_ffmpeg.get_ffmpeg_exe())
        mux_time = _mux_audio(ffmpeg, temporary_video, audio_path, output)
        temporary_video.unlink(missing_ok=True)
        media = validate_output_video(
            output,
            ffmpeg_executable=ffmpeg,
            expected_frame_count=expected_frames,
            expected_duration_sec=audio_duration,
        )
        inference_time = setup_time + generation_time
        return {
            "pipeline": "stream_pipeline_offline.StreamSDK",
            "compatibility_overlay": compatibility,
            "seed": seed,
            "audio_duration_sec": audio_duration,
            "expected_output_frame_count": expected_frames,
            "model_load_time_sec": model_load_time,
            "setup_time_sec": setup_time,
            "generation_time_sec": generation_time,
            "inference_time_sec": inference_time,
            "mux_time_sec": mux_time,
            "total_time_sec": perf_counter() - total_started,
            **media,
        }
    finally:
        os.chdir(original_cwd)
        if sys.path and sys.path[0] == str(source):
            sys.path.pop(0)
