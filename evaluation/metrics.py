from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _is_point(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple, np.ndarray))
        and len(value) >= 2
        and all(isinstance(item, (int, float, np.integer, np.floating)) for item in value[:2])
    )


def _looks_like_mask(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 2
    if not isinstance(value, (list, tuple)) or not value:
        return False
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.ndim == 2 and all(
        isinstance(item, (int, float, np.integer, np.floating, bool, np.bool_))
        for item in array.reshape(-1)
    )


def _as_box(value: Any, box_format: str = "xyxy") -> np.ndarray | None:
    if isinstance(value, Mapping):
        for key in ("xyxy", "bbox_xyxy", "box", "bbox"):
            if key in value:
                value = value[key]
                if key in ("bbox", "xywh", "bbox_xywh") and box_format == "xyxy":
                    box_format = "xywh"
                break
    if isinstance(value, np.ndarray):
        array = value.reshape(-1)
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=float).reshape(-1)
    else:
        return None
    if array.size < 4 or not np.all(np.isfinite(array[:4])):
        return None
    box = array[:4].astype(float)
    if box_format == "xywh" or (
        box_format == "auto"
        and isinstance(value, Mapping)
        and any(key in value for key in ("xywh", "bbox_xywh"))
    ):
        x, y, w, h = box
        return np.asarray([x, y, x + w, y + h], dtype=float)
    return box


def _as_mask(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            return None
        return value.astype(bool, copy=False)
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if array.ndim == 2:
            return array.astype(bool)
    return None


def _box_intersection_area(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return width * height


def _box_union_area(first: np.ndarray, second: np.ndarray) -> float:
    first_width = max(0.0, float(first[2]) - float(first[0]))
    first_height = max(0.0, float(first[3]) - float(first[1]))
    second_width = max(0.0, float(second[2]) - float(second[0]))
    second_height = max(0.0, float(second[3]) - float(second[1]))
    first_area = first_width * first_height
    second_area = second_width * second_height
    return first_area + second_area - _box_intersection_area(first, second)


def _mask_intersection_area(first: np.ndarray, second: np.ndarray) -> int:
    if first.shape != second.shape:
        raise ValueError("masks must have the same shape")
    return int(np.count_nonzero(first & second))


def iou(
    prediction: Any,
    ground_truth: Any,
    box_format: str = "xyxy",
    eps: float = 1e-12,
) -> float:
    if _looks_like_mask(prediction) or _looks_like_mask(ground_truth):
        pred_mask = _as_mask(prediction)
        gt_mask = _as_mask(ground_truth)
        if pred_mask is None or gt_mask is None:
            raise ValueError("mask IoU requires two 2D masks")
        intersection = _mask_intersection_area(pred_mask, gt_mask)
        union = int(np.count_nonzero(pred_mask)) + int(np.count_nonzero(gt_mask)) - intersection
        return 0.0 if union <= 0 else float(intersection / max(union, eps))

    pred_box = _as_box(prediction, box_format)
    gt_box = _as_box(ground_truth, box_format)
    if pred_box is None or gt_box is None:
        raise ValueError("box IoU requires two boxes with four coordinates")
    intersection = _box_intersection_area(pred_box, gt_box)
    union = _box_union_area(pred_box, gt_box)
    return 0.0 if union <= 0 else float(intersection / max(union, eps))


def precision_recall_f1(
    true_positive: Any,
    false_positive: Any,
    false_negative: Any,
) -> tuple[float, float, float]:
    def count(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (str, bytes)):
            return 0.0
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        try:
            return float(np.sum(np.asarray(value, dtype=bool)))
        except (TypeError, ValueError):
            return 0.0

    tp = count(true_positive)
    fp = count(false_positive)
    fn = count(false_negative)
    precision = 0.0 if tp + fp <= 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn <= 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _detection_key(record: Mapping[str, Any], class_agnostic: bool) -> tuple[Any, Any]:
    image_id = record.get("image_id", record.get("image", record.get("file_name", "")))
    if class_agnostic:
        class_id = None
    else:
        class_id = record.get(
            "class_id",
            record.get("category_id", record.get("class", record.get("label", None))),
        )
    return image_id, class_id


def _coerce_records(items: Any) -> list[dict[str, Any]]:
    if items is None:
        return []
    if isinstance(items, Mapping):
        if any(
            key in items
            for key in (
                "bbox",
                "box",
                "xyxy",
                "xywh",
                "bbox_xyxy",
                "bbox_xywh",
                "segmentation",
                "mask",
            )
        ):
            return [dict(items)]
        if "annotations" in items:
            return [dict(item) for item in items.get("annotations", []) if isinstance(item, Mapping)]
        if "boxes" in items or "detections" in items or "predictions" in items:
            nested = items.get("boxes", items.get("detections", items.get("predictions", [])))
            if isinstance(nested, Mapping):
                return [dict(nested)]
            return [dict(item) for item in nested if isinstance(item, Mapping)]
        records: list[dict[str, Any]] = []
        for image_id, value in items.items():
            if isinstance(value, Mapping) and "annotations" in value:
                records.extend(dict(item) for item in value.get("annotations", []) if isinstance(item, Mapping))
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
                for item in value:
                    if isinstance(item, Mapping):
                        record = dict(item)
                        record.setdefault("image_id", image_id)
                        records.append(record)
        return records
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, np.ndarray)):
        records = []
        for item in items:
            if isinstance(item, Mapping):
                records.append(dict(item))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                values = list(item)
                if len(values) >= 4:
                    records.append({"bbox_xyxy": values[:4]})
        return records
    return []


def _record_box(record: Mapping[str, Any]) -> np.ndarray | None:
    for key, box_format in (
        ("bbox_xyxy", "xyxy"),
        ("xyxy", "xyxy"),
        ("box", "xyxy"),
        ("bbox", "xywh"),
        ("bbox_xywh", "xywh"),
        ("xywh", "xywh"),
    ):
        if key in record:
            return _as_box(record[key], box_format)
    return None


def _record_score(record: Mapping[str, Any]) -> float:
    for key in ("score", "confidence", "conf", "probability"):
        if key in record:
            try:
                value = float(record[key])
            except (TypeError, ValueError):
                return 0.0
            return value if math.isfinite(value) else 0.0
    return 1.0


def _record_class(record: Mapping[str, Any]) -> Any:
    return record.get("class_id", record.get("category_id", record.get("class", record.get("label"))))


def _average_precision(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    iou_threshold: float,
    class_agnostic: bool,
) -> float | None:
    gt_groups: dict[tuple[Any, Any], list[np.ndarray | None]] = defaultdict(list)
    for record in ground_truth:
        box = _record_box(record)
        if box is not None:
            gt_groups[_detection_key(record, class_agnostic)].append(box)
    if not gt_groups:
        return None

    pred_groups: dict[tuple[Any, Any], list[tuple[float, np.ndarray | None]]] = defaultdict(list)
    for record in predictions:
        box = _record_box(record)
        if box is not None:
            pred_groups[_detection_key(record, class_agnostic)].append((_record_score(record), box))

    total_gt = sum(len(boxes) for boxes in gt_groups.values())
    if total_gt <= 0:
        return None
    true_positives = 0
    false_positives = 0
    cumulative_tp: list[float] = []
    cumulative_fp: list[float] = []
    for key, grouped_predictions in pred_groups.items():
        unmatched = list(gt_groups.get(key, []))
        for score, pred_box in sorted(grouped_predictions, key=lambda item: item[0], reverse=True):
            best_index = -1
            best_iou = -1.0
            for index, gt_box in enumerate(unmatched):
                if gt_box is None:
                    continue
                overlap = iou(pred_box, gt_box)
                if overlap >= iou_threshold and overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0:
                unmatched[best_index] = None
                true_positives += 1
            else:
                false_positives += 1
            cumulative_tp.append(float(true_positives))
            cumulative_fp.append(float(false_positives))

    if not cumulative_tp:
        return 0.0
    recalls = np.asarray(cumulative_tp, dtype=float) / total_gt
    precisions = np.asarray(cumulative_tp, dtype=float) / (
        np.asarray(cumulative_tp, dtype=float) + np.asarray(cumulative_fp, dtype=float)
    )
    recall_levels = sorted(set(float(value) for value in recalls.tolist()))
    area = 0.0
    previous_recall = 0.0
    for recall_level in recall_levels:
        eligible = precisions[recalls >= recall_level - 1e-12]
        precision_at_recall = float(np.max(eligible)) if eligible.size else 0.0
        area += (recall_level - previous_recall) * precision_at_recall
        previous_recall = recall_level
    return float(area)


def mAP(
    predictions: Any,
    ground_truth: Any,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
    iou_thresholds: Sequence[float] | None = None,
) -> float | None:
    pred_records = _coerce_records(predictions)
    gt_records = _coerce_records(ground_truth)
    thresholds = list(iou_thresholds) if iou_thresholds is not None else [float(iou_threshold)]
    if not thresholds:
        return None
    average_precisions = []
    for threshold in thresholds:
        value = _average_precision(pred_records, gt_records, float(threshold), class_agnostic)
        if value is not None:
            average_precisions.append(value)
    return float(np.mean(average_precisions)) if average_precisions else None


def count_detection_matches(
    predictions: Any,
    ground_truth: Any,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> dict[str, Any]:
    pred_records = _coerce_records(predictions)
    gt_records = _coerce_records(ground_truth)
    gt_groups: dict[tuple[Any, Any], list[np.ndarray | None]] = defaultdict(list)
    for record in gt_records:
        box = _record_box(record)
        if box is not None:
            gt_groups[_detection_key(record, class_agnostic)].append(box)
    pred_groups: dict[tuple[Any, Any], list[tuple[float, np.ndarray | None]]] = defaultdict(list)
    for record in pred_records:
        box = _record_box(record)
        if box is not None:
            pred_groups[_detection_key(record, class_agnostic)].append((_record_score(record), box))

    true_positives = 0
    false_positives = 0
    matched_pairs: list[dict[str, Any]] = []
    for key, grouped_predictions in pred_groups.items():
        unmatched = list(gt_groups.get(key, []))
        for score, pred_box in sorted(grouped_predictions, key=lambda item: item[0], reverse=True):
            best_index = -1
            best_iou = -1.0
            for index, gt_box in enumerate(unmatched):
                if gt_box is None:
                    continue
                overlap = iou(pred_box, gt_box)
                if overlap >= iou_threshold and overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0:
                unmatched[best_index] = None
                true_positives += 1
                matched_pairs.append({"prediction": pred_box.tolist(), "ground_truth": gt_groups[key][best_index].tolist(), "iou": best_iou})
            else:
                false_positives += 1
    false_negatives = sum(len(boxes) for boxes in gt_groups.values()) - true_positives
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "matched": matched_pairs,
        "ground_truth_count": sum(len(boxes) for boxes in gt_groups.values()),
        "prediction_count": len(pred_records),
    }


def detection_precision_recall(
    predictions: Any,
    ground_truth: Any,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> dict[str, Any]:
    counts = count_detection_matches(predictions, ground_truth, iou_threshold, class_agnostic)
    precision, recall, f1 = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "ground_truth_count": counts["ground_truth_count"],
        "prediction_count": counts["prediction_count"],
        "matches": counts["matched"],
    }


def map50(
    predictions: Any,
    ground_truth: Any,
    class_agnostic: bool = False,
) -> float | None:
    return mAP(predictions, ground_truth, iou_threshold=0.5, class_agnostic=class_agnostic)


def calculate_map(
    predictions: Any,
    ground_truth: Any,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> float | None:
    return mAP(predictions, ground_truth, iou_threshold, class_agnostic)


box_iou = iou
mean_average_precision = mAP
precision_recall = detection_precision_recall

__all__ = [
    "box_iou",
    "calculate_map",
    "count_detection_matches",
    "detection_precision_recall",
    "iou",
    "mAP",
    "map50",
    "mean_average_precision",
    "precision_recall",
    "precision_recall_f1",
]
