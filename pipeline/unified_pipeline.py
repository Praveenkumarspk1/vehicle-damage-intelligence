from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cost_engine.indian_repair_quote import (
    DamageRecord,
    extract_damage_records,
    hybrid_reid_match,
    validate_image_quality,
    evaluate_insurance_claim,
)
from cost_engine.gbr_utils import IndiaGBREstimator
from deployment.onnx_inference import ONNXModel
from evaluation.metrics import mAP
from evaluation.report import generate_report
from pipeline.segmentation import bbox_to_mask, mask_area_pixels
from comparison.visualization import overlay_regions, save_visualization, visualize_comparison
from preprocessing._image import load_image


class UnifiedDamagePipeline:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model: Any = None,
        onnx_path: str | None = None,
        unet_path: str | None = None,
    ) -> None:
        self.config = self._load_config(config)
        self.model = model
        self._onnx = ONNXModel(onnx_path) if onnx_path else None
        if self._onnx is not None:
            self._onnx.load()
        self._gbr = IndiaGBREstimator()

    def _load_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        if config:
            return config
        config_path = PROJECT_ROOT / "config.yaml"
        if config_path.is_file():
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {
            "model_path": "models/best.pt",
            "conf_threshold": 0.35,
            "match_tolerance": 0.08,
            "embedding_mode": "classical",
            "data_dir": "data",
            "slots": ["front", "rear", "left", "right"],
            "quality_checks": {
                "min_blur_score": 80.0,
                "min_brightness": 40.0,
                "max_brightness": 225.0,
                "min_contrast": 25.0,
                "glare_v_thresh": 245,
                "glare_s_thresh": 35,
            },
            "location_tiers": {
                "Tier 1": {"denting_labor_hr": 480.0, "paint_booth_panel": 2800.0, "multiplier": 1.15},
                "Tier 2": {"denting_labor_hr": 360.0, "paint_booth_panel": 2200.0, "multiplier": 1.00},
                "Tier 3": {"denting_labor_hr": 260.0, "paint_booth_panel": 1700.0, "multiplier": 0.85},
            },
            "workshop_types": {
                "Authorized OEM Dealer (ASS)": {"labor_markup": 1.20, "parts_markup": 1.00, "gst_rate": 0.18},
                "Multi-Brand Independent Bodyshop (FNG)": {"labor_markup": 0.80, "parts_markup": 0.85, "gst_rate": 0.18},
            },
            "vehicle_models": {
                "Maruti Suzuki (Swift, Baleno, Brezza, Dzire)": {
                    "segment": "Hatchback / Budget Compact",
                    "price_multiplier": 1.00,
                    "stock_lead_days": 1,
                    "availability": "Immediate Stock (24h)",
                    "parts": {"default_part": 2800.0},
                },
                "Hyundai (i20, Creta, Venue, Verna)": {
                    "segment": "Mid-tier Urban",
                    "price_multiplier": 1.25,
                    "stock_lead_days": 2,
                    "availability": "In-Stock (24h - 48h)",
                    "parts": {"default_part": 3500.0},
                },
            },
            "severity_thresholds": {"minor_max_ratio": 0.02, "moderate_max_ratio": 0.07},
            "colors": {
                "BASELINE": {"rgb": [59, 130, 246], "bgr": [246, 130, 59], "hex": "#3b82f6"},
                "NEW": {"rgb": [239, 68, 68], "bgr": [68, 68, 239], "hex": "#ef4444"},
                "EXISTING": {"rgb": [34, 197, 94], "bgr": [94, 197, 34], "hex": "#22c55e"},
                "WORSENED": {"rgb": [249, 115, 22], "bgr": [22, 115, 249], "hex": "#f97316"},
            },
        }

    @property
    def model_path(self) -> str:
        return self.config.get("model_path", "models/best.pt")

    def load_model(self) -> Any:
        if self.model is not None:
            return self.model
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            return self.model
        except Exception as exc:
            self._load_error = str(exc)
            return None

    def process_image(
        self,
        image_path: str,
        slot: str = "front",
        model_name: str | None = None,
        location_tier: str | None = None,
        workshop_type: str | None = None,
        conf_thresh: float | None = None,
        vehicle_age_years: float = 0.0,
        claim_mode: bool = False,
    ) -> dict[str, Any]:
        model = self.load_model()
        if model is None:
            return {"success": False, "error": getattr(self, "_load_error", "model_unavailable")}

        conf_thresh = conf_thresh or float(self.config.get("conf_threshold", 0.35))
        img_array, _ = load_image(image_path, "bgr")
        img_bgr = np.array(img_array)
        if img_bgr.ndim == 2:
            img_bgr = np.stack([img_bgr] * 3, axis=-1)
        elif img_bgr.ndim == 3 and img_bgr.shape[2] == 1:
            img_bgr = np.repeat(img_bgr, 3, axis=-1)
        elif img_bgr.ndim == 3 and img_bgr.shape[2] >= 3:
            img_bgr = img_bgr[:, :, :3]

        if model_name is None:
            model_name = list(self.config.get("vehicle_models", {}).keys())[0]
        if location_tier is None:
            location_tier = list(self.config.get("location_tiers", {}).keys())[0]
        if workshop_type is None:
            workshop_type = list(self.config.get("workshop_types", {}).keys())[0]

        quality = validate_image_quality(img_bgr, self.config)
        damages = extract_damage_records(
            model, img_bgr, slot, conf_thresh, self.config, model_name, location_tier, workshop_type,
            vehicle_age_years=vehicle_age_years, claim_mode=claim_mode,
        )


        unet_areas: dict[str, int] = {
            d.damage_id: int(d.rel_area * img_bgr.shape[0] * img_bgr.shape[1])
            for d in damages
        }

        repair_methods: list[dict[str, Any]] = []
        for d in damages:
            action_lower = getattr(getattr(d, "quote", None), "action", "").lower()
            if "replace" in action_lower:
                method = "replace"
            elif "tinker" in action_lower or "overhaul" in action_lower or "pull" in action_lower:
                method = "overhaul"
            else:
                method = "detach"
            repair_methods.append({
                "damage_id": d.damage_id,
                "method": method,
                "available": True,
                "confidence": round(float(d.confidence), 3),
            })


        gbr_cross: list[dict[str, Any]] = []
        if self._gbr.available:
            for d in damages:
                cross = self._gbr.compare_with_irdai(
                    d.class_name, d.size_bucket, model_name, location_tier,
                    d.quote.total_cost, d.rel_area
                )
                gbr_cross.append({"damage_id": d.damage_id, **cross})

        annotated = self._annotate_frame(img_bgr, damages)
        total_cost = sum(d.quote.total_cost for d in damages)
        insurance = evaluate_insurance_claim(
            total_repair_quote=total_cost,
            vehicle_cc=1500,
            vehicle_age_years=vehicle_age_years,
            current_ncb_pct=35.0,
            claim_mode=claim_mode,
        )
        result = {
            "success": True,
            "image": str(image_path),
            "slot": slot,
            "quality": {
                "is_acceptable": quality.is_acceptable,
                "blur_score": quality.blur_score,
                "brightness": quality.brightness,
                "glare_detected": quality.glare_detected,
                "issues": quality.issues,
            },
            "damages": [self._damage_to_dict(d) for d in damages],
            "total_cost_inr": round(total_cost, 2),
            "damage_count": len(damages),
            "severe_count": sum(1 for d in damages if d.size_bucket == "Severe"),
            "unet_segmented_areas": unet_areas,
            "repair_methods": repair_methods,
            "gbr_cross_validation": gbr_cross,
            "insurance_claim": insurance,
            "vehicle_age_years": vehicle_age_years,
            "claim_mode": claim_mode,
            "annotated_image_path": str(_save_annotation(annotated, image_path)),
        }
        return result

    def compare_before_after(
        self,
        before_path: str,
        after_path: str,
        model_name: str | None = None,
        location_tier: str | None = None,
        workshop_type: str | None = None,
        baseline_damage_ids: list[dict[str, Any]] | None = None,
        vehicle_age_years: float = 0.0,
        claim_mode: bool = False,
    ) -> dict[str, Any]:
        model = self.load_model()
        if model is None:
            return {"success": False, "error": getattr(self, "_load_error", "model_unavailable")}

        conf_thresh = float(self.config.get("conf_threshold", 0.35))
        before_array, _ = load_image(before_path, "bgr")
        after_array, _ = load_image(after_path, "bgr")
        before_bgr = np.array(before_array)
        after_bgr = np.array(after_array)
        for img in (before_bgr, after_bgr):
            if img.ndim == 2:
                img[...] = np.stack([img] * 3, axis=-1)

        if model_name is None:
            model_name = list(self.config.get("vehicle_models", {}).keys())[0]
        if location_tier is None:
            location_tier = list(self.config.get("location_tiers", {}).keys())[0]
        if workshop_type is None:
            workshop_type = list(self.config.get("workshop_types", {}).keys())[0]

        before_damages = extract_damage_records(
            model, before_bgr, "before", conf_thresh, self.config, model_name, location_tier, workshop_type,
            vehicle_age_years=vehicle_age_years, claim_mode=claim_mode,
        )
        after_damages = extract_damage_records(
            model, after_bgr, "after", conf_thresh, self.config, model_name, location_tier, workshop_type,
            vehicle_age_years=vehicle_age_years, claim_mode=claim_mode,
        )

        baseline_list = baseline_damage_ids or [self._damage_to_dict(d) for d in before_damages]
        match_tol = float(self.config.get("match_tolerance", 0.08))
        evaluated, unmatched, liability = hybrid_reid_match(
            baseline_list, after_damages, tolerance=match_tol, alpha=0.65
        )


        match_tolerance = self.config.get("video_walkaround", {}).get("iou_cluster_thresh", 0.35)
        try:
            from comparison.before_after import compare_masks
            before_mask = self._damage_mask(before_bgr, before_damages)
            after_mask = self._damage_mask(after_bgr, after_damages)
            if before_mask is not None and after_mask is not None:
                diff_result = compare_masks(before_mask, after_mask, iou_threshold=match_tolerance)
            else:
                diff_result = {"summary": {"new": 0, "existing": 0, "resolved": 0, "unchanged": 0}}
        except Exception:
            diff_result = {"summary": {"new": 0, "existing": 0, "resolved": 0, "unchanged": 0}}


        before_regions = [
            {"bbox": d.bbox, "status": "existing_damage"}
            for d in before_damages
        ]
        after_regions = [
            {"bbox": d.bbox, "status": (
                "new_damage" if d.verdict == "NEW" else
                "unchanged_damage" if d.verdict == "WORSENED" else
                "existing_damage"
            )}
            for d in evaluated
        ]

        annotated_before = overlay_regions(before_bgr, before_regions)
        annotated_after = overlay_regions(after_bgr, after_regions)


        comparison_result = {
            "comparisons": [
                {
                    "status": d.verdict,
                    "before": {"bbox": d.matched_id, "rel_area": d.rel_area},
                    "after": {"bbox": d.bbox, "rel_area": d.rel_area, "confidence": d.confidence},
                }
                for d in evaluated
            ],
        }
        combined_viz = visualize_comparison(before_bgr, after_bgr, comparison_result, rgb=False)


        output_dir = PROJECT_ROOT / "outputs" / "comparisons"
        output_dir.mkdir(parents=True, exist_ok=True)
        import time as _time
        timestamp = _time.strftime("%Y%m%d_%H%M%S")
        before_path_out = output_dir / f"before_annotated_{timestamp}.png"
        after_path_out = output_dir / f"after_annotated_{timestamp}.png"
        combined_path_out = output_dir / f"comparison_{timestamp}.png"
        save_visualization(annotated_before, before_path_out)
        save_visualization(annotated_after, after_path_out)
        save_visualization(combined_viz, combined_path_out)

        total_quote = sum(d.quote.total_cost for d in evaluated)
        return {
            "success": True,
            "before_damage_count": len(before_damages),
            "after_damage_count": len(after_damages),
            "evaluated_damages": [self._damage_to_dict(d) for d in evaluated],
            "new_count": sum(1 for d in evaluated if d.verdict == "NEW"),
            "worsened_count": sum(1 for d in evaluated if d.verdict == "WORSENED"),
            "existing_count": sum(1 for d in evaluated if d.verdict == "EXISTING"),
            "driver_liability_inr": liability,
            "unmatched_baseline_count": len(unmatched),
            "mask_diff": diff_result,
            "total_repair_quote_inr": round(total_quote, 2),
            "annotated_before_path": str(before_path_out),
            "annotated_after_path": str(after_path_out),
            "comparison_visualization_path": str(combined_path_out),
        }

    def process_video(
        self,
        video_path: str,
        output_dir: str | None = None,
        model_name: str | None = None,
        location_tier: str | None = None,
        workshop_type: str | None = None,
        progress_callback: Any | None = None,
        vehicle_age_years: float = 0.0,
        claim_mode: bool = False,
    ) -> dict[str, Any]:
        model = self.load_model()
        if model is None:
            return {"success": False, "error": getattr(self, "_load_error", "model_unavailable")}
        try:
            from cost_engine.indian_repair_quote import fast_video_walkaround
            model_name = model_name or list(self.config.get("vehicle_models", {}).keys())[0]
            location_tier = location_tier or list(self.config.get("location_tiers", {}).keys())[0]
            workshop_type = workshop_type or list(self.config.get("workshop_types", {}).keys())[0]
            damages, previews, stats = fast_video_walkaround(
                video_path, model, self.config, model_name, location_tier, workshop_type, progress_callback,
                vehicle_age_years=vehicle_age_years, claim_mode=claim_mode,
            )
            return {
                "success": True,
                "video": str(video_path),
                "damages": [self._damage_to_dict(d) for d in damages],
                "damage_count": len(damages),
                "stats": stats,
            }
        except Exception as exc:
            try:
                from tracking.video_pipeline import process_video as _process_video
                return _process_video(video_path, output_dir or "outputs", model=model)
            except Exception:
                return {"success": False, "error": str(exc)}

    def run_onnx_inference(self, image_path: str) -> dict[str, Any]:
        if self._onnx is None or not self._onnx.available:
            return {"available": False, "error": "ONNX model not loaded"}
        img_array, _ = load_image(image_path)
        return self._onnx.predict(img_array)

    def evaluate_map(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> float | None:
        return mAP(predictions, ground_truth, iou_threshold=iou_threshold)

    def generate_report(self, results: dict[str, Any], output_path: str | None = None) -> str:
        report = generate_report(results, output_path)
        return report["report_path"]

    def _annotate_frame(self, img_bgr: np.ndarray, damages: list[DamageRecord]) -> np.ndarray:
        colors = self.config.get("colors", {})
        canvas = img_bgr.copy()
        for d in damages:
            c_info = colors.get(d.verdict, colors.get("BASELINE", {"bgr": [246, 130, 59]}))
            color = tuple(c_info.get("bgr", [0, 200, 0]))
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            import cv2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            reid_str = f" ReID:{d.reid_score:.2f}" if d.reid_score is not None else ""
            tag = f"[{d.verdict}]{reid_str} {d.class_name} | {d.size_bucket} (₹{d.quote.total_cost:,.0f})"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            ty1 = max(0, y1 - th - 6)
            cv2.rectangle(canvas, (x1, ty1), (x1 + tw + 6, ty1 + th + 6), color, -1)
            cv2.putText(canvas, tag, (x1 + 3, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _damage_to_dict(self, d: DamageRecord) -> dict[str, Any]:
        return {
            "damage_id": d.damage_id,
            "class_name": d.class_name,
            "slot": d.slot,
            "rel_center": d.rel_center,
            "rel_area": d.rel_area,
            "size_bucket": d.size_bucket,
            "size_rank": d.size_rank,
            "confidence": d.confidence,
            "bbox": d.bbox,
            "quote": {
                "action": d.quote.action,
                "panel_material": d.quote.panel_material,
                "labor_hours": d.quote.labor_hours,
                "denting_labor_cost": d.quote.denting_labor_cost,
                "paint_booth_cost": d.quote.paint_booth_cost,
                "part_cost": d.quote.part_cost,
                "gst_amount": d.quote.gst_amount,
                "total_cost": d.quote.total_cost,
                "reasoning": d.quote.reasoning,
            },
            "verdict": d.verdict,
            "reid_score": d.reid_score,
            "coord_3d": d.coord_3d,
            "is_glare_candidate": d.is_glare_candidate,
        }

    def _damage_mask(self, img_bgr: np.ndarray, damages: list[DamageRecord]) -> np.ndarray | None:
        if not damages:
            return None
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for idx, d in enumerate(damages, start=1):
            region_mask = bbox_to_mask(d.bbox, (h, w))
            mask[region_mask > 0] = idx
        return mask


def _save_annotation(annotated_img: np.ndarray | None, source_path: str) -> Path | None:
    """Save an annotated image to outputs/annotations/ and return the path."""
    if annotated_img is None:
        return None
    import time as _time
    out_dir = PROJECT_ROOT / "outputs" / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _time.strftime("%Y%m%d_%H%M%S")
    stem = Path(source_path).stem
    out_path = out_dir / f"{stem}_annotated_{timestamp}.png"
    save_visualization(annotated_img, out_path, rgb=False)
    return out_path


def analyze_image(image_path: str, model_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    pipeline = UnifiedDamagePipeline()
    if model_path:
        from ultralytics import YOLO
        pipeline.model = YOLO(model_path)
    return pipeline.process_image(image_path, **kwargs)


analyze_image_unified = analyze_image


__all__ = ["UnifiedDamagePipeline", "analyze_image"]
