import pytest

from src.utils.metrics import effective_generation_fps, real_time_factor


def test_real_time_factor() -> None:
    assert real_time_factor(5.0, 10.0) == pytest.approx(0.5)


@pytest.mark.parametrize("inference,audio", [(-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
def test_real_time_factor_rejects_invalid_values(inference: float, audio: float) -> None:
    with pytest.raises(ValueError):
        real_time_factor(inference, audio)


def test_effective_generation_fps() -> None:
    assert effective_generation_fps(30, 2.0) == pytest.approx(15.0)


@pytest.mark.parametrize("frames,duration", [(-1, 1.0), (1, 0.0), (1, -1.0)])
def test_effective_generation_fps_rejects_invalid_values(frames: int, duration: float) -> None:
    with pytest.raises(ValueError):
        effective_generation_fps(frames, duration)
