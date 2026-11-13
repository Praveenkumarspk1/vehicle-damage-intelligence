from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.metrics import detection_precision_recall, mAP, map50
from evaluation.report import generate_report


def evaluate_detection(predictions: list[dict[str, Any]], ground_truths: list[dict[str, Any]], iou_threshold: float = 0.5) -> dict[str, Any]:
    return {
        "mAP": mAP(predictions, ground_truths, iou_threshold=iou_threshold),
        "map50": map50(predictions, ground_truths),
        "precision_recall_f1": detection_precision_recall(predictions, ground_truths, iou_threshold),
    }


def evaluate_segmentation(predictions: list[dict[str, Any]], ground_truths: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mAP": mAP(predictions, ground_truths, iou_threshold=0.5),
        "precision_recall_f1": detection_precision_recall(predictions, ground_truths, iou_threshold=0.5),
    }


def evaluate_model(model: Any, dataset_path: Any, output_path: Any = None) -> dict[str, Any]:
    dataset = Path(str(dataset_path))
    if not dataset.is_dir():
        return {"success": False, "error": "dataset directory not found"}
    predictions: list[dict[str, Any]] = []
    ground_truths: list[dict[str, Any]] = []
    for sample in sorted(dataset.iterdir()):
        if not sample.is_file():
            continue
        try:
            prediction = model.predict(str(sample)) if hasattr(model, "predict") else model(str(sample))
            predictions.append(prediction if isinstance(prediction, dict) else {"boxes": prediction})
        except Exception as exc:
            predictions.append({"error": str(exc)})
        ground_truths.append({"boxes": []})
    metrics = evaluate_detection(predictions, ground_truths)
    report = generate_report({"detection": metrics}, output_path=output_path)
    return {"metrics": metrics, "report_path": report["report_path"]}


__all__ = ["evaluate_detection", "evaluate_model", "evaluate_segmentation", "generate_report", "mAP", "map50", "detection_precision_recall"]
