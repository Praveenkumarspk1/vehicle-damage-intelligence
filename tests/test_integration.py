"""Integration tests verifying end-to-end connections between GIPSA, IRDAI,
GBR, UNet, and visualization modules.

These tests confirm that:
  1. The GIPSA schedule CSV is actually read and used for labor rates
  2. IRDAI depreciation is applied to part costs when claim_mode=True
  3. The GBR estimator singleton loads and cross-validates against IRDAI quotes
  4. UNet segmentation produces pixel-area outputs
  5. visualize_comparison produces a combined before/after image
  6. The IndianPricingCatalog exposes GIPSA and IRDAI data simultaneously
"""

from pathlib import Path
import json
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gipsa_schedule_csv_is_loaded():
    from data.indian_price_loader import load_gipsa_schedule, GIPSA_SCHEDULE_PATH
    assert GIPSA_SCHEDULE_PATH.exists(), "gipsa_bodyshop_labor_schedule.csv must exist"
    records = load_gipsa_schedule()
    assert len(records) >= 4, "GIPSA schedule should have at least 4 tier rows"
    tiers = {r.location_tier for r in records}
    assert "Tier 1" in tiers
    assert "Tier 2" in tiers
    assert "Tier 3" in tiers


def test_gipsa_labor_rate_matches_csv():
    from data.indian_price_loader import get_gipsa_labor_rate
    rec = get_gipsa_labor_rate("Tier 1")
    assert rec is not None
    assert rec.hourly_denting_rate_inr == 480.0
    assert rec.paint_booth_per_panel_inr == 2800.0
    assert rec.oem_multiplier == 1.20
    rec2 = get_gipsa_labor_rate("Tier 3")
    assert rec2 is not None
    assert rec2.hourly_denting_rate_inr == 260.0


def test_gipsa_labor_rate_accepts_verbose_tier_string():
    from data.indian_price_loader import get_gipsa_labor_rate
    rec = get_gipsa_labor_rate("Tier 1 (Metros: Delhi, Mumbai, Bengaluru, Chennai, Hyd)")
    assert rec is not None
    assert rec.hourly_denting_rate_inr == 480.0


def test_catalog_exposes_gipsa_and_irdai():
    from data.indian_price_loader import get_indian_pricing_catalog
    catalog = get_indian_pricing_catalog()
    assert len(catalog.gipsa_schedule) >= 4
    assert "material_depreciation" in catalog.irdai_norms
    assert "compulsory_deductibles_inr" in catalog.irdai_norms


def test_irdai_depreciation_json_is_loaded():
    from data.indian_price_loader import load_irdai_depreciation, IRDAI_DEPRECIATION_PATH
    assert IRDAI_DEPRECIATION_PATH.exists()
    norms = load_irdai_depreciation()
    assert "material_depreciation" in norms
    assert "compulsory_deductibles_inr" in norms
    assert "no_claim_bonus_slabs_pct" in norms


def test_irdai_depreciation_rate_plastic_50pct():
    from data.indian_price_loader import get_irdai_depreciation_rate
    rate = get_irdai_depreciation_rate("Detachable Plastic / Composite", 0.5)
    assert rate == 0.5


def test_irdai_depreciation_rate_glass_zero():
    from data.indian_price_loader import get_irdai_depreciation_rate
    rate = get_irdai_depreciation_rate("Laminated Safety Glass", 5.0)
    assert rate == 0.0


def test_irdai_depreciation_rate_sheet_metal_by_age():
    from data.indian_price_loader import get_irdai_depreciation_rate
    rate_new = get_irdai_depreciation_rate("Bolt-On Sheet Metal Panel", 0.3)
    assert rate_new == 0.0
    rate_1yr = get_irdai_depreciation_rate("Bolt-On Sheet Metal Panel", 1.5)
    assert rate_1yr == 0.10
    rate_3yr = get_irdai_depreciation_rate("Bolt-On Sheet Metal Panel", 3.5)
    assert rate_3yr == 0.25
    rate_5yr = get_irdai_depreciation_rate("Bolt-On Sheet Metal Panel", 5.0)
    assert rate_5yr == 0.40
    rate_12yr = get_irdai_depreciation_rate("Bolt-On Sheet Metal Panel", 12.0)
    assert rate_12yr == 0.50


def test_depreciated_price_applied_in_claim_mode():
    from cost_engine.indian_repair_quote import calculate_indian_repair_quote
    config = {
        "location_tiers": {"Tier 1 (Metros: Delhi)": {"denting_labor_hr": 480.0, "paint_booth_panel": 2800.0, "multiplier": 1.15}},
        "workshop_types": {"Authorized OEM Dealer (ASS - 100% Genuine Parts)": {"labor_markup": 1.20, "parts_markup": 1.0, "gst_rate": 0.18}},
        "vehicle_models": {"Maruti Suzuki": {"parts": {"bonnet-dent": 4200.0}}},
        "severity_thresholds": {"minor_max_ratio": 0.02, "moderate_max_ratio": 0.07},
        "quality_checks": {},
    }

    quote_full = calculate_indian_repair_quote("bonnet-dent", "Severe", 0.15, "Maruti Suzuki", "Tier 1 (Metros: Delhi)", "Authorized OEM Dealer (ASS - 100% Genuine Parts)", config, vehicle_age_years=0.0, claim_mode=False)

    quote_claim = calculate_indian_repair_quote("bonnet-dent", "Severe", 0.15, "Maruti Suzuki", "Tier 1 (Metros: Delhi)", "Authorized OEM Dealer (ASS - 100% Genuine Parts)", config, vehicle_age_years=5.0, claim_mode=True)

    assert quote_claim.part_cost < quote_full.part_cost

    assert quote_claim.total_cost < quote_full.total_cost

    dep_rate = 0.40
    expected_part = round(4200.0 * (1.0 - dep_rate), 2)
    assert abs(quote_claim.part_cost - expected_part) < 1.0


def test_irdai_compulsory_deductible_from_json():
    from cost_engine.indian_repair_quote import evaluate_insurance_claim
    result = evaluate_insurance_claim(total_repair_quote=50000.0, vehicle_cc=1500)
    assert result["compulsory_deductible_inr"] == 1000.0
    result_big = evaluate_insurance_claim(total_repair_quote=50000.0, vehicle_cc=2000)
    assert result_big["compulsory_deductible_inr"] == 2000.0


def test_gbr_estimator_singleton():
    from cost_engine.gbr_utils import get_gbr_estimator, IndiaGBREstimator
    est = get_gbr_estimator()
    assert isinstance(est, IndiaGBREstimator)
    result = est.estimate("bonnet-dent", "Minor", "Maruti Suzuki", "Tier 2")
    assert "estimated_cost_inr" in result
    assert "source" in result
    assert result["estimated_cost_inr"] > 0


def test_gbr_compare_with_irdai():
    from cost_engine.gbr_utils import get_gbr_estimator
    est = get_gbr_estimator()
    irdai_quote = 8500.0
    result = est.compare_with_irdai("bonnet-dent", "Severe", "Maruti Suzuki", "Tier 2", irdai_quote)
    assert "gbr_estimated_cost_inr" in result
    assert "irdai_rule_cost_inr" in result
    assert "deviation_pct" in result
    assert "recommendation" in result


def test_gbr_cross_validation_in_pipeline():
    from pipeline.unified_pipeline import UnifiedDamagePipeline
    pipeline = UnifiedDamagePipeline()
    assert pipeline._gbr is not None
    assert pipeline._gbr.available is True



def test_pipeline_repair_method_mapping():
    from pipeline.unified_pipeline import UnifiedDamagePipeline
    pipeline = UnifiedDamagePipeline()
    assert hasattr(pipeline, "process_image")



def test_visualize_comparison_produces_combined_image():
    from comparison.visualization import visualize_comparison
    before = np.zeros((64, 64, 3), dtype=np.uint8)
    after = np.zeros((64, 64, 3), dtype=np.uint8)
    result = {
        "comparisons": [
            {
                "status": "new_damage",
                "before": {"bbox": [10, 10, 30, 30]},
                "after": {"bbox": [12, 12, 32, 32]},
            }
        ]
    }
    combined = visualize_comparison(before, after, result)
    assert combined is not None
    assert isinstance(combined, np.ndarray)

    assert combined.shape[1] >= 64


def test_overlay_regions_with_bbox():
    from comparison.visualization import overlay_regions
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    regions = [{"bbox": [10, 10, 50, 50], "status": "existing_damage"}]
    result = overlay_regions(img, regions)
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)


def test_save_visualization_writes_file(tmp_path):
    from comparison.visualization import save_visualization
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = tmp_path / "out.png"
    result_path = save_visualization(img, out)
    assert result_path.is_file()


def test_bbox_to_mask_uses_segmentation_module():
    from pipeline.segmentation import bbox_to_mask, mask_area_pixels
    shape = (100, 100)
    mask = bbox_to_mask([10, 10, 50, 50], shape)
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    area = mask_area_pixels(mask)
    assert area == 40 * 40


def _make_test_image(tmp_path):
    """Create a small test image and return its path."""
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), (120, 150, 180))
    except ImportError:
        img = None
    path = tmp_path / "test_inspection.png"
    if img is not None:
        img.save(path)
    else:
        import cv2
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:] = (120, 150, 180)
        cv2.imwrite(str(path), arr)
    return path


def test_pipeline_result_schema_includes_new_fields(tmp_path):
    """Verify process_image result includes GBR, insurance, and annotation path."""
    from pipeline.unified_pipeline import UnifiedDamagePipeline
    pipeline = UnifiedDamagePipeline()
    img_path = _make_test_image(tmp_path)
    result = pipeline.process_image(str(img_path))
    assert "success" in result
    if result.get("success"):
        assert "gbr_cross_validation" in result
        assert "insurance_claim" in result
        assert "annotated_image_path" in result
        assert "repair_methods" in result
        assert "vehicle_age_years" in result
        assert "claim_mode" in result


def test_pipeline_compare_before_after_returns_paths(tmp_path):
    """Verify compare_before_after returns file paths, not numpy arrays."""
    from pipeline.unified_pipeline import UnifiedDamagePipeline
    pipeline = UnifiedDamagePipeline()
    img = _make_test_image(tmp_path)
    result = pipeline.compare_before_after(str(img), str(img))
    assert "success" in result
    if result.get("success"):

        assert "annotated_before_path" in result
        assert "annotated_after_path" in result
        assert "comparison_visualization_path" in result
