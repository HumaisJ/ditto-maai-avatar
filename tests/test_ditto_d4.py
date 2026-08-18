from __future__ import annotations

import csv
import hashlib
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import test_ditto_file
from src.avatar import ditto_adapter
from src.utils import system_info
from src.utils.audio import create_wav_excerpt


def _write_test_wav(path: Path, *, frame_count: int = 100, sample_rate: int = 20) -> bytes:
    frames = b"".join(
        int(frame).to_bytes(2, "little", signed=True)
        + int(-frame).to_bytes(2, "little", signed=True)
        for frame in range(frame_count)
    )
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(frames)
    return frames


def test_wav_excerpt_copies_exact_frames_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "runtime" / "excerpt.wav"
    source_frames = _write_test_wav(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    details = create_wav_excerpt(source, destination, start_sec=1.0, duration_sec=2.5)

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert details == {
        "sample_rate_hz": 20,
        "channels": 2,
        "sample_width_bytes": 2,
        "start_frame": 20,
        "frame_count": 50,
        "duration_sec": 2.5,
    }
    with wave.open(str(destination), "rb") as excerpt:
        assert excerpt.getparams()[:4] == (2, 2, 20, 50)
        assert excerpt.readframes(50) == source_frames[20 * 4 : 70 * 4]


@pytest.mark.parametrize(
    ("start", "duration", "message"),
    [(-1.0, 1.0, "start_sec"), (0.0, 0.0, "duration_sec"), (5.0, 1.0, "starts")],
)
def test_wav_excerpt_rejects_invalid_windows(
    tmp_path: Path, start: float, duration: float, message: str
) -> None:
    source = tmp_path / "source.wav"
    _write_test_wav(source)
    with pytest.raises(ValueError, match=message):
        create_wav_excerpt(
            source, tmp_path / "excerpt.wav", start_sec=start, duration_sec=duration
        )


def test_d4_configuration_is_fixed_to_one_short_p007_run() -> None:
    config = test_ditto_file.load_config(test_ditto_file.DEFAULT_CONFIG)
    d4 = test_ditto_file._validate_d4_config(config)
    assert d4["pair_id"] == "P007"
    assert d4["clip_start_sec"] == 0.0
    assert d4["clip_duration_sec"] == 5.0
    assert d4["pipeline"] == "offline"
    assert d4["fps"] == 25


def test_d4_refuses_a_second_or_failed_retained_attempt(tmp_path: Path) -> None:
    test_ditto_file.ensure_first_ditto_experiment(tmp_path)
    (tmp_path / "DITTO-EXP-0001").mkdir()
    with pytest.raises(RuntimeError, match="already exists"):
        test_ditto_file.ensure_first_ditto_experiment(tmp_path)


def test_gpu_sampler_records_baseline_peak_and_final_sample(tmp_path: Path, monkeypatch) -> None:
    samples = iter(
        [
            {
                "timestamp_utc": "start",
                "gpu_name": "RTX 5060 Ti",
                "utilization_gpu_percent": 0.0,
                "memory_used_mb": 1000.0,
                "memory_total_mb": 16311.0,
            },
            {
                "timestamp_utc": "end",
                "gpu_name": "RTX 5060 Ti",
                "utilization_gpu_percent": 80.0,
                "memory_used_mb": 9000.0,
                "memory_total_mb": 16311.0,
            },
        ]
    )
    monkeypatch.setattr(system_info, "query_gpu_sample", lambda _: next(samples))
    path = tmp_path / "gpu.csv"
    system_info.initialize_gpu_csv(path)

    with system_info.GpuSampler(path, gpu_index=0, interval_sec=60) as sampler:
        pass

    assert sampler.summary() == {
        "gpu_sample_count": 2,
        "baseline_vram_mb": 1000.0,
        "peak_vram_mb": 9000.0,
        "peak_vram_increase_mb": 8000.0,
        "average_gpu_utilization_percent": 40.0,
        "maximum_gpu_utilization_percent": 80.0,
    }
    with path.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_video_validation_decodes_expected_frames_and_audio(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "generated.mp4"
    output.write_bytes(b"controlled mp4")

    class FakeCapture:
        def __init__(self) -> None:
            self.frames_left = 125

        def isOpened(self) -> bool:
            return True

        def get(self, key: int) -> float:
            return {1: 25.0, 2: 600.0, 3: 550.0}[key]

        def read(self) -> tuple[bool, None]:
            if self.frames_left == 0:
                return False, None
            self.frames_left -= 1
            return True, None

        def release(self) -> None:
            pass

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_WIDTH=2,
        CAP_PROP_FRAME_HEIGHT=3,
        VideoCapture=lambda _: FakeCapture(),
    )
    monkeypatch.setattr(ditto_adapter.importlib, "import_module", lambda _: fake_cv2)
    monkeypatch.setattr(
        ditto_adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""),
    )

    result = ditto_adapter.validate_output_video(
        output,
        ffmpeg_executable="ffmpeg.exe",
        expected_frame_count=125,
        expected_duration_sec=5.0,
    )
    assert result["output_validated"] is True
    assert result["output_frame_count"] == 125
    assert result["output_video_duration_sec"] == 5.0
    assert result["output_resolution"] == [600, 550]


def test_d4_visual_template_contains_the_complete_guide_rubric(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("# experiment\n", encoding="utf-8")
    test_ditto_file._append_visual_review_template(notes)
    content = notes.read_text(encoding="utf-8")
    for field in test_ditto_file.VISUAL_FIELDS:
        assert f"- {field}: pending" in content


def test_d4_adapter_uses_offline_pytorch_path_without_tensorrt() -> None:
    content = Path(ditto_adapter.__file__).read_text(encoding="utf-8").casefold()
    assert "stream_pipeline_offline" in content
    assert "imageio_ffmpeg.get_ffmpeg_exe" in content
    assert "tensorrt" not in content
    for relative in ("src/utils/experiment.py", "src/utils/system_info.py"):
        runtime_module = (test_ditto_file.PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "from datetime import UTC" not in runtime_module
        assert "import Any, Self" not in runtime_module
