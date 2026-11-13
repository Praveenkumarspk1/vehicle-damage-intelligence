from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ._image import load_image

try:
    import numpy as np
except Exception:
    np = None


def _size(value: Any, width: Optional[int] = None, height: Optional[int] = None) -> Tuple[int, int]:
    if width is not None or height is not None:
        return int(height if height is not None else (width if width is not None else 640)), int(
            width if width is not None else (height if height is not None else 640)
        )
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError("size must be an int or a two-item (height, width) sequence")


def _interpolation(name: Any, downscale: bool) -> Any:
    if name is None:
        return None
    if isinstance(name, int):
        return name
    normalized = str(name).lower()
    mapping = {
        "nearest": 0,
        "linear": 1,
        "bilinear": 1,
        "cubic": 2,
        "bicubic": 2,
        "area": 3,
        "lanczos4": 4,
    }
    return mapping.get(normalized, 1 if not downscale else 3)


def _resize_array(array: Any, width: int, height: int, interpolation: Any) -> Any:
    if np is None:
        raise RuntimeError("numpy is required for resizing")
    if width <= 0 or height <= 0:
        raise ValueError("resize dimensions must be positive")
    try:
        import cv2
    except Exception:
        cv2 = None
    if cv2 is not None:
        code = _interpolation(interpolation, height * width > array.shape[0] * array.shape[1])
        return cv2.resize(array, (width, height), interpolation=code)
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("OpenCV or Pillow is required for resizing") from exc
    image = Image.fromarray(np.asarray(array))
    return np.asarray(image.resize((width, height), resample=getattr(Image, "BILINEAR", 2))).copy()


def _pad_array(array: Any, target_height: int, target_width: int, pad_value: Any = 0) -> Any:
    if np is None:
        raise RuntimeError("numpy is required for resizing")
    height, width = array.shape[:2]
    if (height, width) == (target_height, target_width):
        return array.copy()
    pad_top = (target_height - height) // 2
    pad_bottom = target_height - height - pad_top
    pad_left = (target_width - width) // 2
    pad_right = target_width - width - pad_left
    pads = [(pad_top, pad_bottom), (pad_left, pad_right)]
    if array.ndim == 3:
        pads.append((0, 0))
    return np.pad(array, pads, mode="constant", constant_values=pad_value)


def smart_resize(
    image: Any,
    size: Any = (640, 640),
    width: Optional[int] = None,
    height: Optional[int] = None,
    interpolation: Any = None,
    color_format: str = "auto",
    return_info: bool = False,
    preserve_aspect_ratio: bool = True,
    pad_value: Any = 0,
) -> Any:
    target_height, target_width = _size(size, width, height)
    array, _ = load_image(image, color_format)
    if np is None:
        raise RuntimeError("numpy is required for resizing")
    source_height, source_width = array.shape[:2]
    if preserve_aspect_ratio:
        scale = min(target_height / max(source_height, 1), target_width / max(source_width, 1))
        resized_height = max(1, int(round(source_height * scale)))
        resized_width = max(1, int(round(source_width * scale)))
        resized = _resize_array(array, resized_width, resized_height, interpolation)
        output = _pad_array(resized, target_height, target_width, pad_value)
        info = {
            "original_shape": tuple(int(v) for v in array.shape),
            "resized_shape": tuple(int(v) for v in resized.shape),
            "output_shape": tuple(int(v) for v in output.shape),
            "scale": float(scale),
            "ratio": (float(resized_width / max(source_width, 1)), float(resized_height / max(source_height, 1))),
            "padding": (int((target_width - resized_width) // 2), int((target_height - resized_height) // 2)),
        }
    else:
        output = _resize_array(array, target_width, target_height, interpolation)
        info = {
            "original_shape": tuple(int(v) for v in array.shape),
            "resized_shape": tuple(int(v) for v in output.shape),
            "output_shape": tuple(int(v) for v in output.shape),
            "scale": 1.0,
            "ratio": (1.0, 1.0),
            "padding": (0, 0),
        }
    return (output, info) if return_info else output


def letterbox_resize(
    image: Any,
    new_shape: Any = (640, 640),
    stride: int = 32,
    auto: bool = False,
    scaleup: bool = True,
    interpolation: Any = None,
    color_format: str = "auto",
    return_info: bool = False,
    pad_value: Any = 0,
) -> Any:
    target_height, target_width = _size(new_shape)
    if auto and stride > 1:
        target_height = int(round(target_height / stride) * stride)
        target_width = int(round(target_width / stride) * stride)
    array, _ = load_image(image, color_format)
    if np is None:
        raise RuntimeError("numpy is required for resizing")
    source_height, source_width = array.shape[:2]
    scale = min(target_width / max(source_width, 1), target_height / max(source_height, 1))
    if not scaleup:
        scale = min(1.0, scale)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = _resize_array(array, resized_width, resized_height, interpolation)
    pad_left = int(round((target_width - resized_width) / 2.0))
    pad_top = int(round((target_height - resized_height) / 2.0))
    pad_right = target_width - resized_width - pad_left
    pad_bottom = target_height - resized_height - pad_top
    pads = [(pad_top, pad_bottom), (pad_left, pad_right)]
    if array.ndim == 3:
        pads.append((0, 0))
    output = np.pad(resized, pads, mode="constant", constant_values=pad_value)
    info = {
        "original_shape": tuple(int(v) for v in array.shape),
        "resized_shape": tuple(int(v) for v in resized.shape),
        "output_shape": tuple(int(v) for v in output.shape),
        "ratio": float(scale),
        "pad": (pad_left, pad_top),
        "padding": (pad_left, pad_top, pad_right, pad_bottom),
        "stride": int(stride),
    }
    return (output, info) if return_info else output
