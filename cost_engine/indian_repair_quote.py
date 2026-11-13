from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


@dataclass
class QualityResult:
    is_acceptable: bool
    blur_score: float
    brightness: float
    contrast: float
    glare_detected: bool
    issues: List[str]


@dataclass
class RepairQuote:
    action: str
    panel_material: str
    labor_hours: float
    denting_labor_cost: float
    paint_booth_cost: float
    part_cost: float
    gst_amount: float
    total_cost: float
    reasoning: str


@dataclass
class DamageRecord:
    damage_id: str
    class_name: str
    slot: str
    rel_center: List[float]
    rel_area: float
    size_bucket: str
    size_rank: int
    confidence: float
    bbox: List[float]
    quote: RepairQuote
    is_glare_candidate: bool = False
    verdict: str = "BASELINE"
    matched_id: Optional[str] = None
    visual_embedding: Optional[List[float]] = None
    reid_score: Optional[float] = None
    coord_3d: Optional[List[float]] = None
    surveyor_override: Optional[str] = None
    surveyor_notes: Optional[str] = None


def get_panel_material(class_name: str) -> str:
    c = class_name.lower()
    if "bumper" in c or "sidemirror" in c:
        return "Detachable Plastic / Composite"
    if "runningboard" in c or "quaterpanel" in c or "roof" in c:
        return "Fixed Structural Unibody (Frame)"
    if "windscreen" in c or "glass" in c:
        return "Laminated Safety Glass"
    if "tire" in c:
        return "Rubber / Alloy Wheel Assembly"
    return "Bolt-On Sheet Metal Panel"


def check_specular_glare(img_bgr: np.ndarray, bbox: List[float], config: Dict[str, Any]) -> bool:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    crop = img_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    qc = config.get("quality_checks", {})
    v_thresh = qc.get("glare_v_thresh", 245)
    s_thresh = qc.get("glare_s_thresh", 35)
    glare_mask = (hsv[:, :, 2] >= v_thresh) & (hsv[:, :, 1] <= s_thresh)
    glare_ratio = np.count_nonzero(glare_mask) / float(crop.shape[0] * crop.shape[1])
    return bool(glare_ratio > 0.55)


def validate_image_quality(img_bgr: np.ndarray, config: Dict[str, Any]) -> QualityResult:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    qc = config.get("quality_checks", {})
    min_blur = qc.get("min_blur_score", 80.0)
    min_bright = qc.get("min_brightness", 40.0)
    max_bright = qc.get("max_brightness", 225.0)
    min_contrast = qc.get("min_contrast", 25.0)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    glare_mask = (hsv[:, :, 2] >= qc.get("glare_v_thresh", 245)) & (hsv[:, :, 1] <= qc.get("glare_s_thresh", 35))
    glare_detected = bool((np.count_nonzero(glare_mask) / float(img_bgr.shape[0] * img_bgr.shape[1])) > 0.12)
    issues: List[str] = []
    if blur_var < min_blur:
        issues.append(f"Motion blur ({blur_var:.0f} < {min_blur})")
    if brightness < min_bright:
        issues.append(f"Underexposed ({brightness:.0f} < {min_bright})")
    elif brightness > max_bright:
        issues.append(f"Overexposed ({brightness:.0f} > {max_bright})")
    if contrast < min_contrast:
        issues.append(f"Low contrast ({contrast:.0f} < {min_contrast})")
    if glare_detected:
        issues.append("Severe sunlight reflections detected")
    return QualityResult(
        is_acceptable=len(issues) == 0,
        blur_score=round(blur_var, 1),
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        glare_detected=glare_detected,
        issues=issues,
    )


def estimate_severity(rel_area: float, class_name: str, config: Dict[str, Any]) -> Tuple[str, int, float]:
    st_cfg = config.get("severity_thresholds", {})
    minor_max = st_cfg.get("minor_max_ratio", 0.02)
    moderate_max = st_cfg.get("moderate_max_ratio", 0.07)
    area_pct = rel_area * 100.0
    critical_terms = {"windscreen", "glass", "tire", "headlight", "taillight"}
    is_critical = any(t in class_name.lower() for t in critical_terms)
    if is_critical:
        return "Severe", 3, round(area_pct, 2)
    if rel_area < minor_max:
        return "Minor", 1, round(area_pct, 2)
    if rel_area < moderate_max:
        return "Moderate", 2, round(area_pct, 2)
    return "Severe", 3, round(area_pct, 2)


def _extract_gipsa_labor_rate(
    location_tier: str,
    workshop_type: str,
    config: Dict[str, Any],
) -> Tuple[float, float]:
    """Resolve location-tier labor and booth rates.

    Reads base hourly denting rate and paint booth panel rate from the
    GIPSA schedule CSV (data/gipsa_bodyshop_labor_schedule.csv).
    Falls back to config location_tiers if GIPSA data is unavailable.

    Returns (effective_labor_rate_hr, effective_booth_rate_per_panel).
    """
    tier_keyword = location_tier.split("(")[0].strip() if "(" in location_tier else location_tier.strip()
    ws_cfg = config.get("workshop_types", {}).get(workshop_type, {})
    labor_markup = ws_cfg.get("labor_markup", 1.0)
    loc_cfg = config.get("location_tiers", {}).get(location_tier, {})
    loc_multiplier = loc_cfg.get("multiplier", 1.0)
    base_labor = loc_cfg.get("denting_labor_hr", 360.0)
    base_booth = loc_cfg.get("paint_booth_panel", 2200.0)
    try:
        from data.indian_price_loader import get_gipsa_labor_rate
        gipsa_rec = get_gipsa_labor_rate(tier_keyword)
        if gipsa_rec is not None:
            base_labor = gipsa_rec.hourly_denting_rate_inr
            base_booth = gipsa_rec.paint_booth_per_panel_inr
    except Exception:
        pass
    effective_labor_rate = round(base_labor * labor_markup, 2)
    effective_booth_rate = round(base_booth * loc_multiplier, 2)
    return effective_labor_rate, effective_booth_rate


def calculate_indian_repair_quote(
    class_name: str,
    severity: str,
    rel_area: float,
    model_name: str,
    location_tier: str,
    workshop_type: str,
    config: Dict[str, Any],
    vehicle_age_years: float = 0.0,
    claim_mode: bool = False,
) -> RepairQuote:
    panel_mat = get_panel_material(class_name)

    config_with_ws = dict(config)
    effective_labor_rate, effective_booth_rate = _extract_gipsa_labor_rate(location_tier, workshop_type, config_with_ws)
    ws_cfg = config.get("workshop_types", {}).get(workshop_type, {})
    parts_markup = ws_cfg.get("parts_markup", 1.0)
    gst_rate = ws_cfg.get("gst_rate", 0.18)
    v_data = config.get("vehicle_models", {}).get(model_name, {})
    model_parts = v_data.get("parts", {})
    replacement_mrp = model_parts.get(class_name)
    if replacement_mrp is None:
        try:
            from data.indian_price_loader import get_indian_pricing_catalog
            cat = get_indian_pricing_catalog()
            rec = cat.query(model_name, class_name)
            if rec is not None:
                replacement_mrp = rec.oem_part_price_inr
        except Exception:
            pass
    if replacement_mrp is None:
        replacement_mrp = model_parts.get("default_part", 3000.0)
    replacement_mrp = float(replacement_mrp) * parts_markup


    dep_rate = 0.0
    depreciated_part_cost = replacement_mrp
    if claim_mode:
        try:
            from data.indian_price_loader import get_indian_pricing_catalog
            cat = get_indian_pricing_catalog()
            dep_rate = get_indian_pricing_catalog().get_depreciated_price(
                model_name, class_name, vehicle_age_years, use_fng=(parts_markup < 1.0)
            )

            from data.indian_price_loader import get_irdai_depreciation_rate
            dep_rate = get_irdai_depreciation_rate(panel_mat, vehicle_age_years)
            depreciated_part_cost = round(replacement_mrp * (1.0 - dep_rate), 2)
        except Exception:
            dep_rate = 0.0

    is_structural = "Structural" in panel_mat
    is_plastic = "Plastic" in panel_mat
    is_glass = "Glass" in panel_mat
    if is_glass:
        action = "Windscreen / Glass Replacement"
        labor_hours = 2.0
        denting_cost = round(labor_hours * effective_labor_rate, 2)
        paint_cost = 0.0
        part_cost = round(depreciated_part_cost if claim_mode else replacement_mrp, 2)
        if claim_mode and dep_rate > 0:
            reasoning = f"Cracked laminated safety glass cannot be repaired under Indian Motor Vehicles Act; replacement mandated. IRDAI depreciation {dep_rate*100:.0f}% applied (glass = 0% depreciation by IMT norm)."
        else:
            reasoning = "Cracked laminated safety glass cannot be repaired under Indian Motor Vehicles Act; replacement mandated."
    elif severity == "Minor":
        action = "Paintless Dent Repair (PDR) / Spot Touch-up"
        labor_hours = 1.2
        denting_cost = round(labor_hours * effective_labor_rate, 2)
        paint_cost = round(effective_booth_rate * 0.35, 2)
        part_cost = 0.0
        reasoning = f"Superficial cosmetic blemish ({rel_area*100:.1f}% area). Preserves original OEM factory paint."
    elif severity == "Moderate":
        action = "Tinkering Beating & 2K Oven-Bake Paint Blend"
        labor_hours = 3.5 if not is_structural else 5.0
        denting_cost = round(labor_hours * effective_labor_rate, 2)
        panel_factor = 1.35 if any(p in class_name.lower() for p in ["bonnet", "roof"]) else 1.0
        paint_cost = round(effective_booth_rate * panel_factor, 2)
        part_cost = 0.0
        reasoning = f"Moderate surface deformation. Repairing is more cost-effective than OEM replacement (saves ₹{replacement_mrp:,.0f})."
    else:
        if is_structural:
            action = "Hydraulic Frame Pull & Structural Reconstruction"
            labor_hours = 7.5
            denting_cost = round(labor_hours * effective_labor_rate, 2)
            paint_cost = round(effective_booth_rate * 1.2, 2)
            raw_part = round(replacement_mrp * 0.4, 2)
            part_cost = round(raw_part * (1.0 - dep_rate), 2) if claim_mode else raw_part
            reasoning = "Unibody structural section cannot be unbolted. Requires body-bench pulling and alignment."
        else:
            action = "OEM Panel Replacement & Paint Booth Matching"
            labor_hours = 4.0
            denting_cost = round(labor_hours * effective_labor_rate, 2)
            paint_cost = round(effective_booth_rate, 2)
            part_cost = round(depreciated_part_cost if claim_mode else replacement_mrp, 2)
            if claim_mode and dep_rate > 0:
                reasoning = f"Severe structural deformation exceeds repair threshold. IRDAI depreciation {dep_rate*100:.0f}% applied to part cost (₹{replacement_mrp:,.0f} → ₹{part_cost:,.0f})."
            else:
                reasoning = "Severe structural deformation exceeds repair threshold. Authorized replacement recommended."
    subtotal = denting_cost + paint_cost + part_cost
    gst_amount = round(subtotal * gst_rate, 2)
    total_cost = round(subtotal + gst_amount, 2)
    return RepairQuote(
        action=action,
        panel_material=panel_mat,
        labor_hours=labor_hours,
        denting_labor_cost=round(denting_cost, 2),
        paint_booth_cost=round(paint_cost, 2),
        part_cost=round(part_cost, 2),
        gst_amount=gst_amount,
        total_cost=total_cost,
        reasoning=reasoning,
    )


# ── Re-ID & Visual Embedding Delegates (Single Source of Truth: reid) ──────
from reid import (
    extract_classical_embedding as extract_visual_embedding,
    cosine_similarity,
    hybrid_reid_match,
    spatial_only_match as diff_damage_records,
)


def extract_learned_visual_embedding(model: Any, img_bgr: np.ndarray, bbox: List[float]) -> List[float]:
    from reid import extract_reid_embedding
    return extract_reid_embedding(model, img_bgr, bbox, embedding_mode="learned")



_VEHICLE_3D_ZONES: Dict[str, Dict[str, Any]] = {
    "front": {"x_range": (0.0, 0.6), "y_range": (-1.0, 1.0), "z_range": (0.0, 1.0), "label": "Front"},
    "rear": {"x_range": (3.9, 4.5), "y_range": (-1.0, 1.0), "z_range": (0.0, 1.0), "label": "Rear"},
    "left": {"x_range": (0.6, 3.9), "y_range": (-1.0, -0.85), "z_range": (0.0, 0.9), "label": "Left Side"},
    "right": {"x_range": (0.6, 3.9), "y_range": (0.85, 1.0), "z_range": (0.0, 0.9), "label": "Right Side"},
    "bonnet": {"x_range": (0.3, 1.5), "y_range": (-0.8, 0.8), "z_range": (0.85, 1.1), "label": "Bonnet"},
    "roof": {"x_range": (1.3, 3.2), "y_range": (-0.85, 0.85), "z_range": (1.35, 1.5), "label": "Roof"},
}

_CLASS_ZONE_MAP: Dict[str, str] = {
    "bonnet-dent": "bonnet",
    "roof-dent": "roof",
    "front-bumper-dent": "front",
    "rear-bumper-dent": "rear",
    "Runningboard-Damage": "left",
    "quaterpanel-dent": "rear",
}


def project_to_3d_coordinates(
    slot: str,
    class_name: str,
    rel_center: List[float],
) -> List[float]:
    zone_key = _CLASS_ZONE_MAP.get(class_name, slot.lower())
    zone = _VEHICLE_3D_ZONES.get(zone_key, _VEHICLE_3D_ZONES.get(slot.lower(), _VEHICLE_3D_ZONES["front"]))
    cx, cy = rel_center[0], rel_center[1]
    xr = zone["x_range"]
    yr = zone["y_range"]
    zr = zone["z_range"]
    X = round(xr[0] + cx * (xr[1] - xr[0]), 3)
    Z = round(zr[1] - cy * (zr[1] - zr[0]), 3)
    Y = round((yr[0] + yr[1]) / 2.0, 3)
    return [X, Y, Z]


def build_vehicle_mesh_traces() -> List[Dict[str, Any]]:
    chassis = [
        dict(xs=[0, 4.5, 4.5, 0, 0], ys=[-1, -1, 1, 1, -1], zs=[0, 0, 0, 0, 0], name="Floor"),
        dict(xs=[1.3, 3.2, 3.2, 1.3, 1.3], ys=[-0.85, -0.85, 0.85, 0.85, -0.85], zs=[1.45, 1.45, 1.45, 1.45, 1.45], name="Roof"),
        dict(xs=[1.3, 0.6, 0.6, 1.3], ys=[-0.85, -0.85, 0.85, 0.85], zs=[1.45, 1.05, 1.05, 1.45], name="Windscreen"),
        dict(xs=[3.2, 3.8, 3.8, 3.2], ys=[-0.85, -0.85, 0.85, 0.85], zs=[1.45, 1.05, 1.05, 1.45], name="Rear Screen"),
        dict(xs=[0.6, 0.6], ys=[-0.85, -0.85], zs=[0, 1.05], name="A-Pillar L"),
        dict(xs=[3.8, 3.8], ys=[-0.85, -0.85], zs=[0, 1.05], name="C-Pillar L"),
        dict(xs=[0.6, 0.6], ys=[0.85, 0.85], zs=[0, 1.05], name="A-Pillar R"),
        dict(xs=[3.8, 3.8], ys=[0.85, 0.85], zs=[0, 1.05], name="C-Pillar R"),
        dict(xs=[0, 0.6, 0.6, 0, 0], ys=[-0.85, -0.85, 0.85, 0.85, -0.85], zs=[1.0, 1.05, 1.05, 1.0, 1.0], name="Bonnet"),
        dict(xs=[3.8, 4.5, 4.5, 3.8, 3.8], ys=[-0.85, -0.85, 0.85, 0.85, -0.85], zs=[1.05, 0.8, 0.8, 1.05, 1.05], name="Boot"),
    ]
    return chassis


def log_surveyor_feedback(
    damage_id: str,
    original_severity: str,
    corrected_severity: str,
    original_verdict: str,
    corrected_verdict: str,
    notes: str,
    audit_log_path: str = "data/surveyor_audit_log.json",
) -> None:
    Path(audit_log_path).parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "damage_id": damage_id,
        "original_severity": original_severity,
        "corrected_severity": corrected_severity,
        "original_verdict": original_verdict,
        "corrected_verdict": corrected_verdict,
        "notes": notes,
        "annotation_source": "IRDAI_Surveyor_Override",
    }
    with open(audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_surveyor_audit_log(audit_log_path: str = "data/surveyor_audit_log.json") -> List[Dict[str, Any]]:
    p = Path(audit_log_path)
    if not p.exists():
        return []
    entries = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def estimate_turnaround_and_downtime(
    damages: List["DamageRecord"],
    model_name: str,
    daily_fleet_revenue: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    if not damages:
        return {"tat_days": 0, "downtime_loss": 0.0, "availability_note": "No repairs needed.", "recommendation": ""}
    v_cfg = config.get("vehicle_models", {}).get(model_name, {})
    stock_lead = v_cfg.get("stock_lead_days", 2)
    availability_str = v_cfg.get("availability", "Standard Stock")
    has_replacement = any("Replacement" in d.quote.action for d in damages)
    total_labor_hrs = sum(d.quote.labor_hours for d in damages)
    base_work_days = math.ceil(total_labor_hrs / 5.0)
    paint_bake_days = 1 if any(d.quote.paint_booth_cost > 0 for d in damages) else 0
    if has_replacement:
        tat_days = max(base_work_days + paint_bake_days, stock_lead + 1)
    else:
        tat_days = base_work_days + paint_bake_days
    tat_days = max(1, min(tat_days, 18))
    downtime_loss = round(tat_days * daily_fleet_revenue, 2)
    recommendation = ""
    if stock_lead > 4 and has_replacement:
        recommendation = f"Part lead time is {stock_lead} days. Bodyshop recommends tinkering repair over replacement to save ₹{downtime_loss:,.0f} in fleet downtime."
    return {
        "tat_days": tat_days,
        "downtime_loss": downtime_loss,
        "availability_note": f"{availability_str} ({stock_lead} days lead)",
        "recommendation": recommendation,
    }


def evaluate_insurance_claim(
    total_repair_quote: float,
    vehicle_cc: int = 1500,
    vehicle_age_years: float = 0.0,
    current_ncb_pct: float = 35.0,
    annual_od_premium: float = 14000.0,
    claim_mode: bool = False,
) -> Dict[str, Any]:
    """Determine whether to file a motor insurance claim under IRDAI norms.

    Reads compulsory deductibles and NCB slabs from the IRDAI depreciation
    norms JSON (data/irdai_depreciation_norms.json) to ground the decision
    in actual Indian insurance tariff rules rather than hardcoded values.
    """
    try:
        from data.indian_price_loader import load_irdai_depreciation
        norms = load_irdai_depreciation()
        deductibles = norms.get("compulsory_deductibles_inr", {})
        if vehicle_cc <= 1500:
            compulsory_deductible = float(deductibles.get("Four-wheelers <= 1500cc", 1000.0))
        elif vehicle_cc > 1500:
            compulsory_deductible = float(deductibles.get("Four-wheelers > 1500cc", 2000.0))
        else:
            compulsory_deductible = 2500.0
    except Exception:
        compulsory_deductible = 2000.0 if vehicle_cc > 1500 else 1000.0

    ncb_loss = round(annual_od_premium * (current_ncb_pct / 100.0), 2)
    claim_penalty = compulsory_deductible + ncb_loss
    if total_repair_quote <= (claim_penalty + 1200.0):
        decision = "PAY OUT-OF-POCKET (SELF-FUNDED)"
        advice = (
            f"Filing an insurance claim will cost you ₹{claim_penalty:,.0f} "
            f"(₹{compulsory_deductible:,.0f} deductible + ₹{ncb_loss:,.0f} lost "
            f"{int(current_ncb_pct)}% NCB). Paying out-of-pocket saves your future renewal premium!"
        )
    else:
        decision = "FILE MOTOR INSURANCE CLAIM"
        savings = total_repair_quote - claim_penalty
        advice = (
            f"Repair bill (₹{total_repair_quote:,.0f}) significantly exceeds the claim penalty "
            f"(₹{claim_penalty:,.0f}). Net claim benefit is approximately ₹{savings:,.0f}."
        )
    return {
        "decision": decision,
        "advice": advice,
        "claim_penalty": claim_penalty,
        "compulsory_deductible_inr": round(compulsory_deductible, 2),
        "ncb_loss": ncb_loss,
        "vehicle_cc": vehicle_cc,
        "vehicle_age_years": vehicle_age_years,
        "current_ncb_pct": current_ncb_pct,
        "claim_mode": claim_mode,
    }


def extract_damage_records(
    model,
    img_bgr: np.ndarray,
    slot: str,
    conf_thresh: float,
    config: Dict[str, Any],
    model_name: str = "Maruti Suzuki (Swift, Baleno, Brezza, Dzire)",
    location_tier: str = "Tier 1 (Metros: Delhi, Mumbai, Bengaluru, Chennai, Hyd)",
    workshop_type: str = "Authorized OEM Dealer (ASS - 100% Genuine Parts)",
    vehicle_age_years: float = 0.0,
    claim_mode: bool = False,
) -> List[DamageRecord]:
    h, w = img_bgr.shape[:2]
    results = model.predict(img_bgr, conf=conf_thresh, verbose=False)[0]
    records: List[DamageRecord] = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0])
        cls_name = model.names[int(box.cls[0])]
        cx = round(((x1 + x2) / 2.0) / w, 4)
        cy = round(((y1 + y2) / 2.0) / h, 4)
        rel_area = ((x2 - x1) * (y2 - y1)) / float(w * h)
        bbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        is_glare = check_specular_glare(img_bgr, bbox, config)
        severity, rank, _ = estimate_severity(rel_area, cls_name, config)
        quote = calculate_indian_repair_quote(
            cls_name, severity, rel_area, model_name, location_tier, workshop_type, config,
            vehicle_age_years=vehicle_age_years, claim_mode=claim_mode,
        )
        embedding_mode = config.get("embedding_mode", "classical")
        if embedding_mode == "learned":
            visual_emb = extract_learned_visual_embedding(model, img_bgr, bbox)
        else:
            visual_emb = extract_visual_embedding(img_bgr, bbox)
        coord_3d = project_to_3d_coordinates(slot, cls_name, [cx, cy])
        records.append(
            DamageRecord(
                damage_id=f"{slot}_{i+1}",
                class_name=cls_name,
                slot=slot,
                rel_center=[cx, cy],
                rel_area=round(rel_area, 5),
                size_bucket=severity,
                size_rank=rank,
                confidence=round(conf, 3),
                bbox=bbox,
                quote=quote,
                is_glare_candidate=bool(is_glare),
                visual_embedding=visual_emb,
                coord_3d=coord_3d,
            )
        )
    return records


def annotate_frame(
    img_bgr: np.ndarray,
    damages: List[DamageRecord],
    colors: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    canvas = img_bgr.copy()
    for d in damages:
        c_info = colors.get(d.verdict, colors.get("BASELINE", {"bgr": [246, 130, 59]}))
        color = tuple(c_info.get("bgr", [0, 200, 0]))
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        reid_str = f" ReID:{d.reid_score:.2f}" if d.reid_score is not None else ""
        tag = f"[{d.verdict}]{reid_str} {d.class_name} | {d.size_bucket} (₹{d.quote.total_cost:,.0f})"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        ty1 = max(0, y1 - th - 6)
        cv2.rectangle(canvas, (x1, ty1), (x1 + tw + 6, ty1 + th + 6), color, -1)
        cv2.putText(
            canvas, tag, (x1 + 3, ty1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA
        )
    return canvas


def fast_video_walkaround(
    video_path: str,
    model,
    config: Dict[str, Any],
    model_name: str,
    location_tier: str,
    workshop_type: str,
    progress_callback=None,
    vehicle_age_years: float = 0.0,
    claim_mode: bool = False,
) -> Tuple[List[DamageRecord], List[Tuple[np.ndarray, str]], Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    v_cfg = config.get("video_walkaround", {})
    sample_fps = float(v_cfg.get("sample_fps", 1.5))
    step = max(1, int(fps / sample_fps))
    max_kfs = int(v_cfg.get("max_keyframes", 24))
    min_blur = float(v_cfg.get("min_blur_threshold", 65.0))
    conf_thresh = float(config.get("conf_threshold", 0.35))
    start_time = time.time()
    sampled_damages: List[DamageRecord] = []
    keyframe_previews: List[Tuple[np.ndarray, str]] = []
    frame_idx = 0
    kfs_processed = 0
    while cap.isOpened() and kfs_processed < max_kfs:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < min_blur:
            continue
        kfs_processed += 1
        if progress_callback and total_frames > 0:
            progress_callback(
                min(kfs_processed / max_kfs, 1.0),
                f"Analyzing keyframe #{kfs_processed} with Indian rate engine...",
            )
        slot_label = f"walkaround_t{int(frame_idx / fps)}s"
        kf_damages = extract_damage_records(
            model, frame, slot_label, conf_thresh, config, model_name, location_tier, workshop_type,
            vehicle_age_years=vehicle_age_years, claim_mode=claim_mode,
        )
        if kf_damages:
            annotated = annotate_frame(frame, kf_damages, config.get("colors", {}))
            keyframe_previews.append((annotated, slot_label))
            for d in kf_damages:
                is_duplicate = False
                for existing in sampled_damages:
                    if existing.class_name == d.class_name:
                        dx = existing.rel_center[0] - d.rel_center[0]
                        dy = existing.rel_center[1] - d.rel_center[1]
                        if math.hypot(dx, dy) < 0.12:
                            is_duplicate = True
                            if d.confidence > existing.confidence:
                                existing.confidence = d.confidence
                            break
                if not is_duplicate:
                    sampled_damages.append(d)
    cap.release()
    elapsed = time.time() - start_time
    stats = {
        "keyframes_analyzed": kfs_processed,
        "total_video_frames": total_frames,
        "processing_time_sec": round(elapsed, 2),
        "fps_rate": round(kfs_processed / max(elapsed, 0.001), 1),
        "unique_damages_found": len(sampled_damages),
    }
    return sampled_damages, keyframe_previews[:6], stats
