from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

from .before_after import STATUS_EXISTING, STATUS_NEW, STATUS_RESOLVED, STATUS_UNCHANGED

STATUS_COLORS_BGR = {
    STATUS_NEW: (0, 0, 255),
    STATUS_EXISTING: (0, 255, 255),
    STATUS_RESOLVED: (0, 255, 0),
    STATUS_UNCHANGED: (255, 0, 0),
}
STATUS_COLORS_RGB = {key: tuple(reversed(value)) for key, value in STATUS_COLORS_BGR.items()}
STATUS_LABELS = {
    STATUS_NEW: "New damage",
    STATUS_EXISTING: "Existing damage",
    STATUS_RESOLVED: "Resolved",
    STATUS_UNCHANGED: "Unchanged",
}


def _color(status: str, rgb: bool = False) -> tuple[int, int, int]:
    return STATUS_COLORS_RGB.get(status, (128, 128, 128)) if rgb else STATUS_COLORS_BGR.get(status, (128, 128, 128))


def _mask_from_region(region: Mapping[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    mask = region.get("mask")
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.shape[:2] == shape:
        return array.astype(bool)
    padded = np.zeros(shape, dtype=bool)
    height = min(shape[0], array.shape[0])
    width = min(shape[1], array.shape[1])
    padded[:height, :width] = array[:height, :width].astype(bool)
    return padded


def _bbox(region: Mapping[str, Any]) -> np.ndarray | None:
    value = region.get("bbox")
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)[:4]
    return array if array.size == 4 else None


def _put_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], rgb: bool) -> np.ndarray:
    if cv2 is not None:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return image
    if Image is None:
        return image
    pil_image = Image.fromarray(image[..., ::-1] if rgb else image)
    draw = ImageDraw.Draw(pil_image)
    draw.text(origin, text, fill=color)
    result = np.asarray(pil_image)
    return result[..., ::-1].copy() if rgb else result


def overlay_regions(
    image: Any,
    regions: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 0.35,
    thickness: int = 2,
    rgb: bool = False,
) -> np.ndarray:
    array = np.asarray(image).copy()
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[2] > 3:
        array = array[..., :3]
    shape = array.shape[:2]
    if cv2 is not None:
        for region in regions:
            status = str(region.get("status", STATUS_EXISTING))
            color = _color(status, rgb=False)
            display_color = tuple(reversed(color)) if rgb else color
            mask = _mask_from_region(region, shape)
            if mask is not None:
                overlay = array.astype(np.float32)
                overlay[mask] = overlay[mask] * (1.0 - alpha) + np.array(display_color, dtype=np.float32) * alpha
                array = np.clip(overlay, 0, 255).astype(np.uint8)
            bbox = _bbox(region)
            if bbox is not None:
                x1, y1, x2, y2 = [int(round(value)) for value in bbox]
                cv2.rectangle(array, (x1, y1), (x2, y2), display_color, thickness)
                label = STATUS_LABELS.get(status, status)
                cv2.rectangle(array, (x1, max(0, y1 - 16)), (min(array.shape[1], x1 + 150), y1), display_color, -1)
                cv2.putText(array, label, (x1 + 3, max(13, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return array
    if Image is None:
        return array
    pil_image = Image.fromarray(array[..., ::-1] if rgb else array)
    draw = ImageDraw.Draw(pil_image, "RGBA")
    for region in regions:
        status = str(region.get("status", STATUS_EXISTING))
        color = _color(status, rgb=True)
        mask = _mask_from_region(region, shape)
        if mask is not None:
            mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            color_layer = Image.new("RGBA", shape[::-1], color + (int(alpha * 255),))
            pil_image = Image.composite(color_layer, pil_image.convert("RGBA"), mask_image)
        bbox = _bbox(region)
        if bbox is not None:
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)
    result = np.asarray(pil_image.convert("RGB"))
    return result[..., ::-1].copy() if rgb else result


def _comparison_regions(result: Mapping[str, Any], side: str) -> list[Mapping[str, Any]]:
    regions: list[Mapping[str, Any]] = []
    for item in result.get("comparisons", []):
        status = item.get("status")
        region = item.get(f"{side}_region")
        if isinstance(region, Mapping):
            enriched = dict(region)
            enriched["status"] = status
            regions.append(enriched)
    return regions


def visualize_comparison(
    before_image: Any,
    after_image: Any,
    result: Mapping[str, Any],
    *,
    rgb: bool = False,
) -> np.ndarray:
    before = np.asarray(before_image).copy()
    after = np.asarray(after_image).copy()
    if before.shape[:2] != after.shape[:2]:
        height = min(before.shape[0], after.shape[0])
        width = min(before.shape[1], after.shape[1])
        before = before[:height, :width]
        after = after[:height, :width]
    before_view = overlay_regions(before, _comparison_regions(result, "before"), rgb=rgb)
    after_view = overlay_regions(after, _comparison_regions(result, "after"), rgb=rgb)
    if cv2 is not None:
        divider = np.full((before_view.shape[0], 4, before_view.shape[2]), 240, dtype=np.uint8)
        return np.concatenate([before_view, divider, after_view], axis=1)
    return np.concatenate([before_view, after_view], axis=1)


def save_visualization(
    image: Any,
    output_path: str | Path,
    *,
    rgb: bool = False,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if cv2 is not None:
        cv2.imwrite(str(path), array[..., ::-1] if rgb else array)
    elif Image is not None:
        Image.fromarray(array[..., ::-1] if rgb else array).save(path)
    else:
        raise ImportError("OpenCV or Pillow is required to save visualizations")
    return path


def draw_comparison(before_image: Any, after_image: Any, result: Mapping[str, Any], **kwargs: Any) -> np.ndarray:
    return visualize_comparison(before_image, after_image, result, **kwargs)


def visualize(before_image: Any, after_image: Any, result: Mapping[str, Any], **kwargs: Any) -> np.ndarray:
    return visualize_comparison(before_image, after_image, result, **kwargs)


__all__ = [
    "STATUS_COLORS_BGR",
    "STATUS_COLORS_RGB",
    "STATUS_LABELS",
    "draw_comparison",
    "overlay_regions",
    "save_visualization",
    "visualize",
    "visualize_comparison",
]
