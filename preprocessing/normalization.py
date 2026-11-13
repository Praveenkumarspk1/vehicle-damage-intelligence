from __future__ import annotations

from typing import Any, Tuple

from ._image import load_image

try:
    import numpy as np
except Exception:
    np = None


def _array(image: Any, color_format: str) -> Any:
    array, _ = load_image(image, color_format)
    if np is None:
        raise RuntimeError("numpy is required for normalization")
    return np.asarray(array)


def normalize_image(
    image: Any,
    max_value: float = 255.0,
    dtype: Any = np.float32 if np is not None else float,
    color_format: str = "auto",
    clip: bool = True,
    channel_first: bool = False,
) -> Any:
    array = _array(image, color_format)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image pixels must be numeric")
    work = array.astype(np.float32, copy=False)
    if work.size and float(np.nanmax(work)) <= 1.0 and float(np.nanmin(work)) >= 0.0:
        normalized = work
    else:
        divisor = float(max_value) if float(max_value) > 0 else 1.0
        normalized = work / divisor
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    normalized = normalized.astype(dtype, copy=False)
    if channel_first and normalized.ndim == 3:
        normalized = np.transpose(normalized, (2, 0, 1))
    return normalized


def denormalize_image(
    image: Any,
    max_value: float = 255.0,
    dtype: Any = np.uint8 if np is not None else int,
    color_format: str = "auto",
    channel_first: bool = False,
) -> Any:
    array = _array(image, color_format)
    if channel_first and array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
    work = array.astype(np.float32, copy=False)
    if work.size and float(np.nanmax(work)) <= 1.0 and float(np.nanmin(work)) >= 0.0:
        work = work * float(max_value)
    if float(max_value) <= 255.0 and np.issubdtype(dtype, np.integer):
        work = np.clip(work, 0.0, float(max_value))
    work = np.nan_to_num(work, nan=0.0, posinf=float(max_value), neginf=0.0)
    return work.astype(dtype, copy=False)
