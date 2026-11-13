from __future__ import annotations

from typing import Any

import numpy as np

from .segmentation import mask_area_pixels


def detect_damage(image: Any, vehicle_mask: Any = None, model: Any = None, confidence: float = 0.25) -> list[dict[str, Any]]:
    if model is not None:
        try:
            if callable(getattr(model, "predict", None)):
                try:
                    results = model.predict(image, conf=confidence)
                except TypeError:
                    results = model.predict(image)
            elif callable(model):
                try:
                    results = model(image, conf=confidence)
                except TypeError:
                    results = model(image)
            else:
                results = None

            if isinstance(results, dict) and results.get("available"):
                return _regions_from_results(results)

            if isinstance(results, (list, tuple)) and len(results) > 0 and hasattr(results[0], "boxes"):
                res = results[0]
                regions: list[dict[str, Any]] = []
                boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else []
                confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else []
                classes = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, "cls") else []
                names = getattr(res, "names", {})
                masks = res.masks.data.cpu().numpy() if getattr(res, "masks", None) is not None else None
                for i, box in enumerate(boxes):
                    c_id = int(classes[i]) if i < len(classes) else None
                    c_name = names.get(c_id, "damage") if names else "damage"
                    c_score = float(confs[i]) if i < len(confs) else 0.0
                    m = masks[i].tolist() if masks is not None and i < len(masks) else None
                    regions.append({
                        "bbox": [float(v) for v in box[:4]],
                        "class_id": c_id,
                        "class_name": c_name,
                        "confidence": c_score,
                        "damage_type": c_name,
                        "mask": m,
                        "area_pixels": mask_area_pixels(m) if m is not None else 0,
                        "source": "model",
                    })
                return regions
        except Exception:
            pass
    return _heuristic_damage(image, vehicle_mask, confidence)


def _regions_from_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    boxes = results.get("boxes", [])
    classes = results.get("classes", [])
    confidences = results.get("confidences", [])
    masks = results.get("masks", [])
    names = results.get("names", {})
    regions: list[dict[str, Any]] = []
    for index in range(len(boxes)):
        box = boxes[index] if index < len(boxes) else []
        class_id = classes[index] if index < len(classes) else None
        score = confidences[index] if index < len(confidences) else 0.0
        mask = masks[index] if index < len(masks) else None
        regions.append({
            "bbox": [float(value) for value in box[:4]],
            "class_id": class_id,
            "class_name": names.get(class_id, None) if names else None,
            "confidence": float(score),
            "damage_type": "not_available",
            "mask": mask,
            "area_pixels": mask_area_pixels(mask) if mask is not None else 0,
            "source": "model",
        })
    return regions


def _heuristic_damage(image: Any, vehicle_mask: Any, confidence: float) -> list[dict[str, Any]]:
    try:
        array = _as_rgb(image)
        height, width = array.shape[:2]
    except Exception:
        return []
    if vehicle_mask is None:
        mask = np.ones((height, width), dtype=bool)
    else:
        mask = _as_bool_mask(vehicle_mask, height, width)
    gray = _to_gray(array)
    if mask.sum() == 0:
        return []
    mean_color = gray[mask].mean() if gray[mask].size else 0.0
    local_variance = (gray - gray.mean()) ** 2
    contrast = local_variance[mask]
    threshold = float(np.percentile(contrast, 95)) if contrast.size else 0.0
    anomaly_mask = contrast > max(threshold, float(np.percentile(contrast, 90))) if contrast.size else np.zeros_like(gray, dtype=bool)
    anomaly_mask = anomaly_mask & mask
    area = int(mask_area_pixels(anomaly_mask))
    panel_area = int(mask_area_pixels(mask))
    ratio = area / panel_area if panel_area > 0 else 0.0
    if area == 0:
        return []
    bbox = _mask_bbox(anomaly_mask)
    return [{
        "bbox": bbox,
        "class_id": None,
        "class_name": "damage",
        "confidence": float(min(1.0, ratio * 2.0)),
        "damage_type": "not_available",
        "mask": anomaly_mask.tolist(),
        "area_pixels": area,
        "source": "heuristic",
        "metadata": {"damage_area_ratio": float(ratio) if panel_area else 0.0},
    }]


def _as_rgb(image: Any) -> np.ndarray:
    try:
        from preprocessing._image import load_image
        array, _ = load_image(image, "rgb")
        return np.asarray(array)
    except Exception:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=-1)
        return array


def _as_bool_mask(mask: Any, height: int, width: int) -> np.ndarray:
    from .segmentation import extract_mask
    if mask is None:
        return np.ones((height, width), dtype=bool)
    array = extract_mask(mask, (height, width))
    return np.asarray(array, dtype=bool)


def _to_gray(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[2] in (3, 4):
        channels = array[:, :, :3].astype(np.float32)
        gray = 0.299 * channels[:, :, 0] + 0.587 * channels[:, :, 1] + 0.114 * channels[:, :, 2]
    else:
        gray = array.astype(np.float32)
    return gray


def _mask_bbox(mask: Any) -> list[int]:
    array = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(array)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
