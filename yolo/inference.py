from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = "yolov8n-seg.pt"
COCO_VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COCO_CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "snowboard",
    39: "billiard table",
}


def resolve_yolo_weights(reference: Any | None = None) -> Path | None:
    if reference is None:
        return None
    path = Path(str(reference)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return path
    if path.suffix.lower() in (".pt", ".pth", ".onnx"):
        return None
    candidate = PROJECT_ROOT / "models" / "yolo" / str(reference)
    if candidate.is_file():
        return candidate
    return None


def load_model(reference: Any | None = None):
    model = YOLOModel(reference)
    model.load()
    return model


class YOLOModel:
    def __init__(self, weights: Any = DEFAULT_WEIGHTS, conf: float = 0.25, iou: float = 0.45, device: Any = None) -> None:
        self.weights = weights
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self._model = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def names(self) -> dict:
        return getattr(self._model, "names", {}) if self._model else {}

    def load(self) -> YOLOModel:
        if self._model is not None:
            return self
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self._load_error = f"ultralytics is not installed: {exc}"
            return self
        local = resolve_yolo_weights(self.weights)
        try:
            if local is not None:
                self._model = YOLO(str(local))
            else:
                reference = self.weights if isinstance(self.weights, str) else DEFAULT_WEIGHTS
                self._model = YOLO(reference)
        except Exception as exc:
            self._load_error = str(exc)
            self._model = None
        return self

    def predict(self, image: Any, **kwargs: Any) -> dict[str, Any]:
        if self._model is None:
            return {"boxes": [], "classes": [], "confidences": [], "masks": [], "names": {}, "raw": None, "available": False}
        result = self._model.predict(source=image, conf=self.conf, iou=self.iou, **kwargs)
        return self._parse_result(result)

    def predict_with_masks(self, image: Any, **kwargs: Any) -> dict[str, Any]:
        return self.predict(image, **kwargs)

    def get_vehicles(self, image: Any, **kwargs: Any) -> dict[str, Any]:
        output = self.predict(image, **kwargs)
        if not output["available"]:
            return output
        vehicle_indices = [
            index for index, class_id in enumerate(output["classes"]) if class_id in COCO_VEHICLE_CLASS_IDS
        ]
        return {
            "boxes": [output["boxes"][index] for index in vehicle_indices],
            "classes": [output["classes"][index] for index in vehicle_indices],
            "confidences": [output["confidences"][index] for index in vehicle_indices],
            "masks": [output["masks"][index] for index in vehicle_indices],
            "names": output["names"],
            "available": True,
        }

    def visualize(self, image: Any, results: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if results is None:
            results = self.predict(image, **kwargs)
        if not results.get("available"):
            return image
        boxes = results.get("boxes", [])
        classes = results.get("classes", [])
        confidences = results.get("confidences", [])
        names = results.get("names", {})
        try:
            import cv2
            array = _to_array(image)
            for box, class_id, confidence in zip(boxes, classes, confidences):
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = [int(value) for value in box[:4]]
                cv2.rectangle(array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{names.get(class_id, class_id)} {confidence:.2f}"
                cv2.putText(array, label, (x1, max(y1, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return array
        except ImportError:
            return image

    def _parse_result(self, result: Any) -> dict[str, Any]:
        if not isinstance(result, Sequence) or len(result) == 0:
            return {"boxes": [], "classes": [], "confidences": [], "masks": [], "names": {}, "raw": result, "available": True}
        first = result[0]
        names = getattr(first, "names", getattr(first, "names_dict", {})) or {}
        boxes: list[list[float]] = []
        classes: list[int] = []
        confidences: list[float] = []
        masks: list[Any] = []
        if hasattr(first, "boxes") and getattr(first.boxes, "xyxy", None) is not None:
            xyxy = first.boxes.xyxy
            conf = first.boxes.conf
            cls = first.boxes.cls
            for index in range(len(xyxy)):
                boxes.append([float(value) for value in xyxy[index][:4]])
                classes.append(int(cls[index]))
                confidences.append(float(conf[index]))
        if hasattr(first, "masks") and getattr(first.masks, "data", None) is not None:
            masks = [mask.tolist() for mask in first.masks.data]
        return {
            "boxes": boxes,
            "classes": classes,
            "confidences": confidences,
            "masks": masks,
            "names": dict(names),
            "raw": result,
            "available": True,
        }


def _to_array(image: Any):
    try:
        import numpy as np
        if hasattr(image, "convert"):
            return np.asarray(image.convert("RGB"))
        return np.asarray(image)
    except Exception:
        return image
