from __future__ import annotations

import io
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Any) -> Path:
    if path is None:
        raise ValueError("path is required")
    if isinstance(path, Path):
        candidate = path
    else:
        candidate = Path(str(path)).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _import_numpy() -> Any:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required for image operations") from exc
    return np


def _import_pil() -> Any:
    try:
        import importlib
        return importlib.import_module("PIL.Image")
    except Exception as exc:
        raise RuntimeError("Pillow is required for this image operation") from exc


def _import_cv2() -> Any:
    try:
        import cv2
    except Exception:
        return None
    return cv2


def _as_rgb_array(array: Any, color_format: str, np_module: Any) -> tuple[Any, str]:
    arr = np_module.asarray(array)
    if arr.ndim == 2:
        return arr, "gray"
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError("image array must have shape HxW, HxWx1, HxWx3, or HxWx4")
    if arr.shape[2] == 1:
        return arr[:, :, 0], "gray"
    normalized_format = color_format.lower() if isinstance(color_format, str) else "rgb"
    if normalized_format not in ("rgb", "bgr", "auto"):
        raise ValueError("color_format must be 'rgb', 'bgr', or 'auto'")
    detected = "rgb" if normalized_format == "auto" else normalized_format
    if normalized_format == "bgr" or detected == "bgr":
        channels = arr.copy()
        channels[..., :3] = channels[..., 2::-1]
        return channels, "rgb"
    return arr.copy(), detected


def load_image(source: Any, color_format: str = "auto") -> tuple[Any, str]:
    np_module = _import_numpy()
    try:
        pil_module = _import_pil()
    except RuntimeError:
        pil_module = None
    if pil_module is not None and isinstance(source, pil_module.Image):
        image = source.convert("RGBA" if source.mode == "RGBA" else "RGB")
        return np_module.asarray(image).copy(), "rgba" if image.mode == "RGBA" else "rgb"
    if isinstance(source, np_module.ndarray):
        return _as_rgb_array(source, color_format, np_module)
    if hasattr(source, "read"):
        if pil_module is None:
            raise RuntimeError("Pillow is required to decode image streams")
        try:
            image = pil_module.Image.open(source)
            image.load()
            image = image.convert("RGBA" if image.mode == "RGBA" else "RGB")
            return np_module.asarray(image).copy(), "rgba" if image.mode == "RGBA" else "rgb"
        except Exception as exc:
            raise ValueError("unable to decode image stream") from exc
    if isinstance(source, bytes):
        if pil_module is None:
            raise RuntimeError("Pillow is required to decode image bytes")
        try:
            image = pil_module.Image.open(io.BytesIO(source))
            image.load()
            image = image.convert("RGBA" if image.mode == "RGBA" else "RGB")
            return np_module.asarray(image).copy(), "rgba" if image.mode == "RGBA" else "rgb"
        except Exception as exc:
            raise ValueError("unable to decode image bytes") from exc
    path = resolve_path(source)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    if pil_module is not None:
        try:
            image = pil_module.Image.open(path)
            image.load()
            image = image.convert("RGBA" if image.mode == "RGBA" else "RGB")
            return np_module.asarray(image).copy(), "rgba" if image.mode == "RGBA" else "rgb"
        except Exception:
            pass
    cv2_module = _import_cv2()
    if cv2_module is None:
        raise ValueError(f"unable to decode image: {path}")
    array = cv2_module.imread(str(path), cv2_module.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f"unable to decode image: {path}")
    if array.ndim == 2:
        return array, "gray"
    if array.ndim == 3 and array.shape[2] in (3, 4):
        rgb = array.copy()
        rgb[..., :3] = rgb[..., 2::-1]
        return rgb, "rgba" if array.shape[2] == 4 else "rgb"
    raise ValueError(f"unsupported image channels: {array.shape}")


def image_to_pil(image: Any, color_format: str = "rgb") -> Any:
    np_module = _import_numpy()
    pil_module = _import_pil()
    array, _ = _as_rgb_array(image, color_format, np_module)
    if np_module.issubdtype(array.dtype, np_module.floating):
        finite = np_module.isfinite(array)
        if not finite.all():
            raise ValueError("image contains non-finite values")
        if array.size and float(np_module.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np_module.clip(array, 0, 255).astype(np_module.uint8)
    elif np_module.issubdtype(array.dtype, np_module.integer):
        info = np_module.iinfo(array.dtype)
        if info.min >= 0 and info.max > 255:
            array = np_module.clip(array, 0, info.max)
            array = (array.astype(np_module.float32) * (255.0 / float(info.max))).astype(np_module.uint8)
        else:
            array = np_module.clip(array, 0, 255).astype(np_module.uint8)
    else:
        array = np_module.clip(array, 0, 255).astype(np_module.uint8)
    if array.ndim == 2:
        return pil_module.Image.fromarray(array, mode="L")
    if array.shape[2] == 4:
        return pil_module.Image.fromarray(array, mode="RGBA")
    return pil_module.Image.fromarray(array, mode="RGB")


def save_image(image: Any, path: Any, color_format: str = "rgb") -> Path:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np_module = _import_numpy()
    array, _ = _as_rgb_array(image, color_format, np_module)
    if color_format.lower() == "bgr" and array.ndim == 3 and array.shape[2] >= 3:
        array = array.copy()
        array[..., :3] = array[..., 2::-1]
    image_to_pil(array, "rgb").save(output_path)
    return output_path


def image_dimensions(source: Any, color_format: str = "auto") -> tuple[int, int]:
    array, _ = load_image(source, color_format)
    if array.ndim == 2:
        return int(array.shape[1]), int(array.shape[0])
    return int(array.shape[1]), int(array.shape[0])


def image_shape(source: Any, color_format: str = "auto") -> tuple[int, ...]:
    array, _ = load_image(source, color_format)
    return tuple(int(value) for value in array.shape)
