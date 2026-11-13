"""Unified image analysis pipeline — Korean legacy API bridge.

This module was historically the entry point for vehicle damage inspection
(``analyze_image``). It now delegates to the canonical
``UnifiedDamagePipeline.process_image`` from
``pipeline.unified_pipeline``, which uses the Indian IRDAI/GIPSA cost
engine, YOLOv8 detection, and U-Net segmentation.

The public ``analyze_image`` signature and return schema are preserved
for backward compatibility with any scripts that import from here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DAMAGE_TYPES = ("Breakage", "Crushed", "Scratched", "Separated")
SEVERITY_ORDER = {"none": 0, "not_available": 0, "minor": 1, "moderate": 2, "severe": 3, "critical": 4}


def _summary_severity(damages: list[dict[str, Any]]) -> str:
    best = "none"
    for damage in damages:
        severity = damage.get("size_bucket") or "not_available"
        severity = severity.lower() if isinstance(severity, str) else "not_available"
        if SEVERITY_ORDER.get(severity, 0) > SEVERITY_ORDER.get(best, 0):
            best = severity
    if best == "not_available":
        return "none"
    return best


def analyze_image(
    image_path: Any,
    model_path: Any = None,
    confidence: float = 0.25,
    output_dir: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Analyze a vehicle damage image.

    Delegates to UnifiedDamagePipeline.process_image and returns a
    backward-compatible schema.
    """
    from pipeline.unified_pipeline import UnifiedDamagePipeline

    pipeline = UnifiedDamagePipeline()
    if model_path is not None:
        from ultralytics import YOLO
        pipeline.model = YOLO(str(model_path))

    conf_thresh = kwargs.pop("conf_thresh", confidence)
    result = pipeline.process_image(
        str(image_path),
        conf_thresh=conf_thresh,
        **kwargs,
    )

    if not result.get("success"):
        return {
            "image": str(image_path),
            "vehicle": {"detected": False, "available": False},
            "damages": [],
            "repair_method": {"method": "not_available", "available": False, "confidence": 0.0},
            "estimated_cost": {"estimated_cost": None, "currency": "INR", "available": False, "fallback": True},
            "severity": "none",
            "severity_score": 0.0,
            "damage_count": 0,
            "metadata": {"detection_backend": "unified_pipeline", "severity_method": "IRDAI_rule_based"},
        }

    damages = result.get("damages", [])
    model_name = result.get("model_name", list(pipeline.config.get("vehicle_models", {}).keys())[0] if pipeline.config.get("vehicle_models") else "")
    total_cost = result.get("total_cost_inr", 0.0)

    if output_dir is not None:
        _save_visualization(result.get("annotated_image_path"), Path(str(output_dir)) / "inspection_result.png")

    return {
        "image": str(image_path),
        "vehicle": {"detected": True, "available": True, "class_name": model_name},
        "damages": damages,
        "repair_method": result.get("repair_methods", [{"method": "not_available", "available": False, "confidence": 0.0}])[0] if result.get("repair_methods") else {"method": "not_available", "available": False, "confidence": 0.0},
        "estimated_cost": {"estimated_cost": total_cost, "currency": "INR", "available": True, "fallback": False},
        "severity": _summary_severity(damages),
        "severity_score": max((d.get("confidence", 0) for d in damages), default=0.0),
        "damage_count": len(damages),
        "metadata": {
            "detection_backend": "unified_pipeline",
            "severity_method": "IRDAI_rule_based",
            "repair_cost": {"estimated_cost": total_cost, "currency": "INR"},
            "gbr_cross_validation": result.get("gbr_cross_validation", []),
            "insurance_claim": result.get("insurance_claim", {}),
        },
    }


def _save_visualization(source_path: str | Path | None, dest_path: Path) -> None:
    if source_path is None:
        return
    try:
        import shutil
        shutil.copy2(str(source_path), dest_path)
    except Exception:
        pass
