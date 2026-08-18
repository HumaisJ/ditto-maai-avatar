"""Windows compatibility helpers for the pinned Ditto inference source."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

BLEND_MODULE = "core.utils.blend"


def blend_images_numpy(
    mask_warped: np.ndarray,
    frame_warped: np.ndarray,
    frame_rgb: np.ndarray,
    result: np.ndarray,
) -> None:
    """Match Ditto's small Cython blend kernel using vectorized NumPy operations."""
    if mask_warped.dtype != np.float32 or mask_warped.ndim != 2:
        raise TypeError("mask_warped must be a two-dimensional float32 array")
    if frame_warped.dtype != np.float32 or frame_warped.ndim != 3:
        raise TypeError("frame_warped must be a three-dimensional float32 array")
    if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3:
        raise TypeError("frame_rgb must be a three-dimensional uint8 array")
    if result.dtype != np.uint8 or result.ndim != 3:
        raise TypeError("result must be a three-dimensional uint8 array")
    expected_shape = (*mask_warped.shape, 3)
    if frame_warped.shape != expected_shape:
        raise ValueError(f"frame_warped must have shape {expected_shape}")
    if frame_rgb.shape != expected_shape or result.shape != expected_shape:
        raise ValueError("frame_rgb and result must match frame_warped's shape")

    mask = mask_warped[..., np.newaxis]
    blended = mask * frame_warped + (1.0 - mask) * frame_rgb
    result[...] = np.clip(blended, 0, 255).astype(np.uint8)


def install_windows_blend_fallback() -> str:
    """Provide Ditto's blend import without compiling its GCC-oriented Cython extension."""
    module = ModuleType(BLEND_MODULE)
    module.__doc__ = "Project-owned NumPy fallback for Ditto's Windows blend extension."
    module.__package__ = "core.utils"
    module.blend_images_cy = blend_images_numpy  # type: ignore[attr-defined]
    sys.modules[BLEND_MODULE] = module
    return "numpy_vectorized_windows_fallback"
