from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def detect_vehicle(image: Any, model: Any = None, confidence: float = 0.25) -> dict[str, Any]:
    if model is None:
        try:
            from yolo.inference import YOLOModel
            model = YOLOModel()
            model.load()
        except Exception as exc:
            return {"detected": False, "available": False, "error": str(exc), "bbox": None, "mask": None, "confidence": 0.0}
    if not getattr(model, "available", False) and not model.is_loaded:
        model = model if model.available else None
    if model is None:
        return {"detected": False, "available": False, "bbox": None, "mask": None, "confidence": 0.0}
    try:
        results = model.get_vehicles(image) if hasattr(model, "get_vehicles") else model.predict(image)
    except Exception as exc:
        return {"detected": False, "available": False, "error": str(exc), "bbox": None, "mask": None, "confidence": 0.0}
    if not results.get("available"):
        return {"detected": False, "available": False, "bbox": None, "mask": None, "confidence": 0.0}
    boxes = results.get("boxes", [])
    if not boxes:
        return {"detected": False, "available": True, "bbox": None, "mask": None, "confidence": 0.0}
    best_index = max(range(len(boxes)), key=lambda index: float(results["confidences"][index]) if index < len(results["confidences"]) else 0.0)
    return {
        "detected": True,
        "available": True,
        "class_id": results["classes"][best_index] if best_index < len(results["classes"]) else None,
        "class_name": _coco_name(results["classes"][best_index], results.get("names")),
        "bbox": [float(value) for value in boxes[best_index][:4]],
        "confidence": float(results["confidences"][best_index]) if best_index < len(results["confidences"]) else 0.0,
        "mask": results["masks"][best_index] if best_index < len(results["masks"]) else None,
    }


def _coco_name(class_id: Any, names: dict[Any, str] | None) -> str | None:
    if names:
        return str(names.get(class_id, class_id))
    return None


def detect_vehicles(image: Any, model: Any = None, confidence: float = 0.25) -> list[dict[str, Any]]:
    single = detect_vehicle(image, model, confidence)
    return [single] if single.get("detected") else []
