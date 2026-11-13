from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

try:
    import cv2
except ImportError:
    cv2 = None


def _as_bool_array(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0, 0), dtype=bool)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value)
    if array.dtype == bool:
        return array
    return array > 0


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    if mask.shape == (target_h, target_w):
        return mask.astype(bool)
    if mask.size == 0:
        return np.zeros((target_h, target_w), dtype=bool)
    if Image is not None:
        image = Image.fromarray(mask.astype(np.uint8) * 255)
        image = image.resize((target_w, target_h), Image.Resampling.NEAREST)
        return np.asarray(image) > 0
    y = np.clip(np.linspace(0, mask.shape[0] - 1, target_h).astype(int), 0, mask.shape[0] - 1)
    x = np.clip(np.linspace(0, mask.shape[1] - 1, target_w).astype(int), 0, mask.shape[1] - 1)
    return mask[np.ix_(y, x)].astype(bool)


def mask_to_bbox(mask: Any) -> list[int] | None:
    array = _as_bool_array(mask)
    ys, xs = np.nonzero(array)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_area_pixels(mask: Any) -> int:
    return int(np.count_nonzero(_as_bool_array(mask)))


def bbox_to_mask(bbox: Any, shape: Sequence[int]) -> np.ndarray:
    if not bbox or len(bbox) < 4:
        return np.zeros(tuple(shape[:2]), dtype=bool)
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
    height, width = int(shape[0]), int(shape[1])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, max(x2, x1)), min(height, max(y2, y1))
    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def _decode_rle(rle: Any, shape: Sequence[int]) -> np.ndarray:
    if isinstance(rle, dict) and "counts" in rle:
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        if isinstance(counts, str):
            heights = [int(item) for item in str(rle.get("size", [shape[0]])).replace("[", "").replace("]", "").split(",") if item]
            height = int(heights[0]) if heights else int(shape[0])
            width = int(heights[1]) if len(heights) > 1 else int(shape[1])
            counts_array: list[int] = []
            index = 0
            while index < len(counts):
                end = index
                while end < len(counts) and counts[end].isdigit():
                    end += 1
                if end == index:
                    index += 1
                    continue
                counts_array.append(int(counts[index:end]))
                index = end
            mask = np.zeros(height * width, dtype=bool)
            cursor = 0
            for run_index, length in enumerate(counts_array):
                if run_index % 2 == 1:
                    mask[cursor:cursor + length] = True
                cursor += length
            return mask.reshape((height, width), order="F").astype(bool)
    if isinstance(rle, Sequence) and not isinstance(rle, (str, bytes)):
        mask = np.zeros(int(shape[0]) * int(shape[1]), dtype=bool)
        cursor = 0
        for run_index, length in enumerate(rle):
            if run_index % 2 == 1:
                mask[cursor:cursor + int(length)] = True
            cursor += int(length)
        return mask.reshape((int(shape[0]), int(shape[1])), order="F").astype(bool)
    return np.zeros(tuple(shape[:2]), dtype=bool)


def polygon_to_mask(polygon: Any, shape: Sequence[int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    if not polygon:
        return mask
    points = []
    for item in polygon:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((int(round(float(item[0]))), int(round(float(item[1])))))
    if not points:
        return mask
    if ImageDraw is not None:
        image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(image).polygon(points, fill=1)
        return np.asarray(image, dtype=np.uint8) > 0
    for y in range(height):
        intersections = []
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            if (y1 <= y < y2) or (y2 <= y < y1):
                intersections.append(int(round(x1 + (y - y1) * (x2 - x1) / (y2 - y1))))
        intersections.sort()
        for index in range(0, len(intersections) - 1, 2):
            mask[y, max(0, intersections[index]):min(width, intersections[index + 1] + 1)] = True
    return mask


def _fallback_polygon(mask: np.ndarray) -> list[list[int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def mask_to_polygon(mask: Any, simplify_epsilon: float = 1.0) -> list[list[int]]:
    array = _as_bool_array(mask)
    if cv2 is not None and array.size:
        contours, _ = cv2.findContours(array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if simplify_epsilon > 0:
                contour = cv2.approxPolyDP(contour, simplify_epsilon, True)
            return [[int(point[0][0]), int(point[0][1])] for point in contour]
    return _fallback_polygon(array)


def mask_to_rle(mask: Any) -> dict[str, Any]:
    array = _as_bool_array(mask).astype(np.uint8)
    if array.size == 0:
        return {"counts": [], "size": [0, 0]}
    flat = array.reshape(-1, order="F")
    positions = np.diff(np.concatenate(([0], flat, [0])))
    starts = np.where(positions == 1)[0]
    ends = np.where(positions == -1)[0]
    return {"counts": [int(value) for value in np.concatenate((ends, starts)).tolist()], "size": [int(array.shape[0]), int(array.shape[1])]}


def rle_to_mask(rle: Any, shape: Sequence[int]) -> np.ndarray:
    return _decode_rle(rle, shape)


def mask_iou(first: Any, second: Any) -> float:
    first_mask = _as_bool_array(first)
    second_mask = _as_bool_array(second)
    if first_mask.shape != second_mask.shape:
        height = max(first_mask.shape[0], second_mask.shape[0])
        width = max(first_mask.shape[1], second_mask.shape[1])
        first_padded = np.zeros((height, width), dtype=bool)
        second_padded = np.zeros((height, width), dtype=bool)
        first_padded[:first_mask.shape[0], :first_mask.shape[1]] = first_mask
        second_padded[:second_mask.shape[0], :second_mask.shape[1]] = second_mask
        first_mask, second_mask = first_padded, second_padded
    intersection = int(np.count_nonzero(first_mask & second_mask))
    union = int(np.count_nonzero(first_mask | second_mask))
    return float(intersection / union) if union else 1.0 if intersection else 0.0


def mask_to_serializable(mask: Any, simplify_epsilon: float = 1.0) -> dict[str, Any]:
    array = _as_bool_array(mask)
    bbox = mask_to_bbox(array)
    return {
        "polygon": mask_to_polygon(array, simplify_epsilon),
        "rle": mask_to_rle(array),
        "bbox": bbox,
        "area_pixels": mask_area_pixels(array),
    }


def extract_mask(value: Any, shape: Sequence[int]) -> np.ndarray:
    if value is None:
        return np.zeros(tuple(shape[:2]), dtype=bool)
    if isinstance(value, dict):
        if "polygon" in value:
            return _resize_mask(polygon_to_mask(value["polygon"], shape), shape)
        if "rle" in value:
            return _resize_mask(rle_to_mask(value["rle"], shape), shape)
        if "mask" in value:
            return extract_mask(value["mask"], shape)
        if "bbox" in value:
            return bbox_to_mask(value["bbox"], shape)
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)) and len(value[0]) >= 2:
            try:
                float(value[0][0])
                return _resize_mask(polygon_to_mask(value, shape), shape)
            except (TypeError, ValueError):
                pass
        array = np.asarray(value)
        if array.ndim >= 2:
            return _resize_mask(_as_bool_array(array), shape)
    if hasattr(value, "shape") and getattr(value, "ndim", 2) >= 2:
        return _resize_mask(_as_bool_array(value), shape)
    return np.zeros(tuple(shape[:2]), dtype=bool)


def extract_segmentation_mask(value: Any, shape: Sequence[int]) -> np.ndarray:
    return extract_mask(value, shape)


def compute_mask_area(mask: Any) -> int:
    return mask_area_pixels(mask)


def get_bounding_polygon(mask: Any, simplify_epsilon: float = 1.0) -> list[list[int]]:
    return mask_to_polygon(mask, simplify_epsilon)
