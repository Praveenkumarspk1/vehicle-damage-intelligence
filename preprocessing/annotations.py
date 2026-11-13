from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ._image import resolve_path


def _is_coco_annotation(value: Any) -> bool:
    return isinstance(value, Mapping) and ("annotations" in value or "images" in value)


def convert_coco_to_yolo(coco: Mapping[str, Any], image_id: Any, image_width: Any, image_height: Any) -> List[List[float]]:
    annotations = coco.get("annotations", []) if isinstance(coco, Mapping) else []
    width = int(float(image_width))
    height = int(float(image_height))
    rows: List[List[float]] = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        if annotation.get("image_id") != image_id:
            continue
        category = annotation.get("category_id", annotation.get("class_id"))
        bbox = annotation.get("bbox")
        box_format = str(annotation.get("bbox_format", "xywh")).lower()
        box = _to_xyxy(bbox, box_format, width, height)
        if box is None or category is None:
            continue
        rows.append(_to_yolo_row(int(category), box, width, height))
    return rows


def convert_yolo_to_coco(yolo_rows: Sequence[Sequence[float]], class_names: Optional[Mapping[int, str]] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in yolo_rows:
        values = [float(value) for value in row]
        if len(values) < 5:
            continue
        class_id = int(values[0])
        center_x, center_y, width, height = values[1], values[2], values[3], values[4]
        records.append({
            "category_id": class_id,
            "category_name": class_names.get(class_id) if class_names is not None else None,
            "bbox": [center_x, center_y, width, height],
            "bbox_format": "xywh",
            "area": float(width * height),
        })
    return records


def validate_yolo_annotations(labels_dir: Any, num_classes: int = 80, image_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    root = resolve_path(labels_dir)
    results: List[Path] = sorted(path for path in root.glob("*.txt") if path.is_file()) if root.is_dir() else []
    if not results:
        return {"valid": False, "checked_files": 0, "errors": ["no label files found"], "warnings": []}
    width, height = image_size or (640, 640)
    errors: List[str] = []
    warnings: List[str] = []
    checked = 0
    for path in results:
        checked += 1
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        for line_index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                errors.append(f"{path.name}:{line_index}: expected >=5 fields")
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                errors.append(f"{path.name}:{line_index}: class id not integer")
                continue
            if class_id < 0 or class_id >= num_classes:
                errors.append(f"{path.name}:{line_index}: class id {class_id} out of range")
            for value in parts[1:5]:
                try:
                    fraction = float(value)
                except ValueError:
                    errors.append(f"{path.name}:{line_index}: coordinate not numeric")
                    continue
                if fraction < 0.0 or fraction > 1.0:
                    warnings.append(f"{path.name}:{line_index}: coordinate {fraction} outside [0,1]")
    return {
        "valid": not errors,
        "checked_files": checked,
        "errors": errors[:200],
        "warnings": warnings[:200],
    }


def _to_xyxy(bbox: Any, box_format: str, width: int, height: int) -> Optional[List[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    values = [float(value) for value in bbox[:4]]
    if box_format in ("xywh", "coco"):
        x_center, y_center, box_width, box_height = values
        x1 = x_center - box_width / 2.0
        y1 = y_center - box_height / 2.0
        return [x1, y1, x1 + box_width, y1 + box_height]
    if box_format in ("xyxy", "cxcy"):
        return values
    return values


def _to_yolo_row(class_id: int, box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2.0 / width
    center_y = (y1 + y2) / 2.0 / height
    norm_width = (x2 - x1) / width
    norm_height = (y2 - y1) / height
    return [float(class_id), center_x, center_y, norm_width, norm_height]
